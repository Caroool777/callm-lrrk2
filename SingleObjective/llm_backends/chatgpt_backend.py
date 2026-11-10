"""ChatGPT/OpenAI backend for CALLM molecule editing."""

from typing import Dict, List, Optional, Union
import json
import os
import re
import time
import urllib.error
import urllib.request

from rdkit import Chem

from .abstract_backend import LLMBackend
from .prompts_multi import MULTI_CONSTRAINED_DESCRIPTIONS, MULTI_TASK_DESCRIPTIONS
from .prompts_single import SINGLE_CONSTRAINED_DESCRIPTIONS, SINGLE_TASK_DESCRIPTIONS


def _load_selfies():
    try:
        import selfies as sf
    except ImportError as exc:
        raise ImportError("The 'selfies' package is required for molecule editing. Install selfies>=2.1.1.") from exc
    return sf


class ChatGPTBackend(LLMBackend):
    """OpenAI Chat Completions implementation of the CALLM molecule editor."""

    def __init__(
        self,
        num_beams: int = 5,
        num_return_sequences: int = 3,
        temperature: float = 0.8,
        do_sample: bool = True,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ) -> None:
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {num_beams}")
        if num_return_sequences < 1:
            raise ValueError(f"num_return_sequences must be >= 1, got {num_return_sequences}")
        if not 0 < temperature <= 2.0:
            raise ValueError(f"temperature should be in (0, 2.0], got {temperature}")

        super().__init__(
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=do_sample,
            device=device,
        )
        self.api_key_env_var = "OPENAI_API_KEY"
        self.api_key = api_key or os.environ.get(self.api_key_env_var)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", timeout_seconds))
        self.max_retries = max(1, int(os.environ.get("OPENAI_MAX_RETRIES", max_retries)))
        self._load_prompts()

        print(f"[{self.backend_name}] model={self.model_name}, candidates={self.num_return_sequences}, "
              f"temperature={self.temperature}")
        if not self.api_key:
            print(f"[warn] {self.api_key_env_var} not set; API calls will fail until configured.")

    @property
    def backend_name(self) -> str:
        return "ChatGPT"

    def _load_prompts(self) -> None:
        self.task2description = {
            **SINGLE_TASK_DESCRIPTIONS,
            **MULTI_TASK_DESCRIPTIONS,
        }
        self.task2constrained_description = {
            **SINGLE_CONSTRAINED_DESCRIPTIONS,
            **MULTI_CONSTRAINED_DESCRIPTIONS,
        }

    def _build_prompt(
        self,
        task_name: str,
        selfies_input: str,
        scaffold_constraint: Optional[str] = None,
    ) -> str:
        if scaffold_constraint:
            try:
                sf = _load_selfies()
                scaffold_selfies = sf.encoder(scaffold_constraint)
            except (ValueError, RuntimeError) as exc:
                print(f"[warn] Failed to encode scaffold {scaffold_constraint}: {exc}")
                scaffold_selfies = scaffold_constraint
            task_definition = self.task2constrained_description.get(
                task_name,
                self.task2description.get(task_name, ""),
            ).format(scaffold=scaffold_selfies)
        else:
            task_definition = self.task2description.get(task_name, "")

        return (
            task_definition
            + f"Now complete the following example -\nInput: <bom>{selfies_input}<eom>\nOutput: "
        )

    def _openai_headers(self) -> Dict[str, str]:
        self.api_key = self.api_key or os.environ.get(self.api_key_env_var)
        if not self.api_key:
            raise RuntimeError(
                f"{self.api_key_env_var} is not configured. Set it before running ChatGPT generation."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_chat_completion(self, payload: Dict) -> Dict:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        last_error = None

        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=data, headers=self._openai_headers(), method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 400:
                    lowered = body.lower()
                    updated_payload = dict(payload)
                    if "max_tokens" in lowered and "max_completion_tokens" not in updated_payload:
                        updated_payload["max_completion_tokens"] = updated_payload.pop("max_tokens")
                    for optional_param in ("temperature", "top_p"):
                        if optional_param in lowered and optional_param in updated_payload:
                            updated_payload.pop(optional_param, None)
                    if updated_payload != payload:
                        payload = updated_payload
                        data = json.dumps(payload).encode("utf-8")
                        last_error = body
                        continue
                if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < self.max_retries:
                    last_error = body
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenAI API failed ({exc.code}): {body}") from exc
            except urllib.error.URLError as exc:
                last_error = str(exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenAI API failed: {exc}") from exc

        raise RuntimeError(f"OpenAI API failed after {self.max_retries} retries: {last_error}")

    def _build_chatgpt_messages(
        self,
        task_prompt: str,
        input_smiles: str,
        input_selfies: str,
        scaffold_constraint: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        constraint_text = ""
        if scaffold_constraint:
            constraint_text = (
                f"\nRequired scaffold or substructure SMILES: {scaffold_constraint}\n"
                "Preserve this scaffold/substructure when chemically possible."
            )
        user_prompt = (
            f"Generate up to {max(1, int(self.num_return_sequences))} distinct candidate molecules.\n"
            f"Input canonical SMILES: {input_smiles}\n"
            f"Input SELFIES: {input_selfies}\n"
            f"{constraint_text}\n\n"
            f"Task prompt:\n{task_prompt}\n\n"
            "Rules:\n"
            "- Output valid SMILES strings, not SELFIES.\n"
            "- Keep molecules drug-like and synthetically reasonable.\n"
            "- Avoid salts, atom-mapping labels, placeholders, explanations, and markdown.\n"
            "- Return JSON only, with this exact shape: {\"smiles\": [\"...\"]}."
        )
        return [
            {
                "role": "system",
                "content": "You are a medicinal chemistry molecule editor.",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _extract_candidate_strings(self, text: str) -> List[str]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        candidates = []
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in ("smiles", "candidates", "molecules"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        candidates.extend(value)
                        break
            elif isinstance(parsed, list):
                candidates.extend(parsed)
        except json.JSONDecodeError:
            pass

        if not candidates:
            for line in cleaned.splitlines():
                item = re.sub(r"^[\-\*\d\.\)\s]+", "", line.strip().strip(",;"))
                if item:
                    candidates.append(item)

        normalized = []
        for item in candidates:
            if isinstance(item, dict):
                item = item.get("smiles") or item.get("SMILES") or item.get("selfies") or ""
            if isinstance(item, str):
                item = item.strip().strip("\"'`,;")
                if item:
                    normalized.append(item)
        return normalized

    def _candidate_to_mol(self, candidate: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            return mol
        try:
            sf = _load_selfies()
            smiles = sf.decoder(candidate)
        except (ValueError, RuntimeError):
            return None
        return Chem.MolFromSmiles(smiles)

    def _generate_candidate_mols(
        self,
        task_prompt: str,
        input_smiles: str,
        input_selfies: str,
        scaffold_constraint: Optional[str] = None,
    ) -> List[Chem.Mol]:
        payload = {
            "model": self.model_name,
            "messages": self._build_chatgpt_messages(
                task_prompt,
                input_smiles,
                input_selfies,
                scaffold_constraint,
            ),
            "temperature": self.temperature if self.do_sample else 0,
            "top_p": 0.95 if self.do_sample else 1,
            "n": 1,
            "max_tokens": 800,
        }
        response = self._post_chat_completion(payload)

        generated = []
        seen = set()
        for choice in response.get("choices", []):
            content = (choice.get("message") or {}).get("content", "")
            for candidate in self._extract_candidate_strings(content):
                mol = self._candidate_to_mol(candidate)
                if mol is None:
                    continue
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if smiles in seen:
                    continue
                seen.add(smiles)
                generated.append(mol)
                if len(generated) >= self.num_return_sequences:
                    return generated
        return generated

    def edit(
        self,
        smiles_list: List,
        scaffold_constraints: Optional[Union[str, List[str]]] = None,
    ) -> List:
        if not self.task:
            raise RuntimeError(f"Task not set for {self.backend_name}. Call set_task() first.")

        task_name = self.task[0]
        if task_name not in self.task2description:
            raise ValueError(f"Task {task_name} not found in {self.backend_name}")

        if scaffold_constraints is None:
            constraints = [None] * len(smiles_list)
        elif isinstance(scaffold_constraints, str):
            constraints = [scaffold_constraints] * len(smiles_list)
        else:
            constraints = list(scaffold_constraints)
            if len(constraints) < len(smiles_list):
                constraints.extend([None] * (len(smiles_list) - len(constraints)))
            elif len(constraints) > len(smiles_list):
                constraints = constraints[:len(smiles_list)]

        edited_molecules = []
        for index, mol in enumerate(smiles_list):
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            try:
                sf = _load_selfies()
                selfies_input = sf.encoder(smiles)
            except (ValueError, RuntimeError) as exc:
                print(f"[warn] Cannot encode SMILES {smiles}: {exc}")
                continue

            scaffold = constraints[index] if index < len(constraints) else None
            task_prompt = self._build_prompt(task_name, selfies_input, scaffold)
            try:
                edited_molecules.extend(
                    self._generate_candidate_mols(
                        task_prompt,
                        smiles,
                        selfies_input,
                        scaffold_constraint=scaffold,
                    )
                )
            except Exception as exc:
                print(f"[warn] {self.backend_name} generation failed for {smiles}: {exc}")
        return edited_molecules

    def apply_config(self, config: Dict) -> None:
        if not config:
            return
        self.num_beams = config.get("llm_num_beams", config.get("biot5_num_beams", self.num_beams))
        self.num_return_sequences = config.get(
            "llm_num_return_sequences",
            config.get("biot5_num_return_sequences", self.num_return_sequences),
        )
        self.temperature = config.get("llm_temperature", config.get("biot5_temperature", self.temperature))
        self.do_sample = config.get("llm_do_sample", config.get("biot5_do_sample", self.do_sample))
        self.model_name = config.get("openai_model", config.get("chatgpt_model", self.model_name))
        self.base_url = config.get("openai_base_url", self.base_url).rstrip("/")
        self.timeout_seconds = int(config.get("openai_timeout_seconds", self.timeout_seconds))
        self.max_retries = max(1, int(config.get("openai_max_retries", self.max_retries)))
        if config.get("openai_api_key_env"):
            self.api_key_env_var = str(config["openai_api_key_env"])
            self.api_key = os.environ.get(self.api_key_env_var)
        print(f"[{self.backend_name}] Config applied: model={self.model_name}")
