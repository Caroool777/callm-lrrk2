"""BioT5 backend for local Hugging Face molecule editing."""

from typing import Dict, List, Optional, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .abstract_backend import LLMBackend
from .prompts_multi import MULTI_CONSTRAINED_DESCRIPTIONS, MULTI_TASK_DESCRIPTIONS
from .prompts_single import SINGLE_CONSTRAINED_DESCRIPTIONS, SINGLE_TASK_DESCRIPTIONS


def _load_selfies():
    try:
        import selfies as sf
    except ImportError as exc:
        raise ImportError("The 'selfies' package is required for BioT5 generation. Install selfies>=2.1.1.") from exc
    return sf


class BioT5Backend(LLMBackend):
    """Local BioT5 text-to-molecule implementation with lazy model loading."""

    def __init__(
        self,
        num_beams: int = 5,
        num_return_sequences: int = 3,
        temperature: float = 0.8,
        do_sample: bool = True,
        device: Optional[str] = None,
        model_name: str = "QizhiPei/biot5-base-text2mol",
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
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.seed = 42
        self.scaffold_constraint_mode = "tag"
        self._generation_count = 0
        self._warned_generate_without_generator = False
        self._torch = None
        self._load_prompts()
        print(f"[{self.backend_name}] model={self.model_name}, candidates={self.num_return_sequences}")

    @property
    def backend_name(self) -> str:
        return "BioT5"

    def _load_prompts(self) -> None:
        self.task2description = {
            **SINGLE_TASK_DESCRIPTIONS,
            **MULTI_TASK_DESCRIPTIONS,
        }
        self.task2constrained_description = {
            **SINGLE_CONSTRAINED_DESCRIPTIONS,
            **MULTI_CONSTRAINED_DESCRIPTIONS,
        }

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "BioT5 generation requires torch and transformers. "
                "Install them before using llm_backend='biot5'."
            ) from exc

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(self.device).startswith("cuda") and not torch.cuda.is_available():
            print(f"[warn] Requested device {self.device} is unavailable; falling back to CPU.")
            self.device = "cpu"

        print(f"[{self.backend_name}] Loading {self.model_name} on {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self._torch = torch

    def _build_prompt(
        self,
        task_name: str,
        selfies_input: str,
        scaffold_constraint: Optional[str] = None,
    ) -> str:
        if task_name in {"scaffold_hop", "deco_hop"}:
            if scaffold_constraint:
                task_description = self.task2constrained_description[task_name].format(
                    scaffold=str(scaffold_constraint)
                )
            else:
                task_description = self.task2description[task_name]

            if task_name == "scaffold_hop":
                objective_context = (
                    "The forbidden scaffold SMARTS is "
                    "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12. "
                )
            else:
                objective_context = (
                    "The decorations to remove are matched by SMARTS "
                    "CS([#6])(=O)=O and [#7]-c1ccc2ncsc2c1. "
                )

            return (
                "Definition: You are given a molecule description in English. "
                "Your job is to generate the molecule SELFIES that fits the description.\n\n"
                "Now complete the following example -\n"
                f"Input: {task_description} The optimization parent SELFIES is "
                f"{selfies_input}. {objective_context}"
                "The reference molecule is "
                "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C.\n"
                "Output: "
            )

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

    def _matches_scaffold_constraint(self, mol: Chem.Mol, scaffold_constraint: Optional[str]) -> bool:
        if mol is None or not scaffold_constraint:
            return True
        query = Chem.MolFromSmarts(str(scaffold_constraint))
        if query is None:
            query = Chem.MolFromSmiles(str(scaffold_constraint))
        if query is None:
            return False
        try:
            if mol.HasSubstructMatch(query):
                return True
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                return scaffold.HasSubstructMatch(query) or query.HasSubstructMatch(scaffold)
        except Exception:
            return False
        return False

    def _candidate_to_mol(self, candidate_selfies: str) -> Optional[Chem.Mol]:
        sf = _load_selfies()
        raw_candidate = candidate_selfies.strip()
        candidates = []

        def add_candidate(value: str) -> None:
            value = value.strip()
            if value and value not in candidates:
                candidates.append(value)

        add_candidate(raw_candidate)
        compact_candidate = "".join(raw_candidate.split())
        add_candidate(compact_candidate)
        for fragment in compact_candidate.split("."):
            add_candidate(fragment)

        for candidate in candidates:
            try:
                smiles = sf.decoder(candidate)
            except Exception:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return mol
        return None

    def _generate_candidate_mols(
        self,
        task_prompt: str,
        input_smiles: str,
        input_selfies: str,
        scaffold_constraint: Optional[str] = None,
    ) -> List[Chem.Mol]:
        self._ensure_model_loaded()
        encoded = self.tokenizer(
            task_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generation_kwargs = {
            "max_length": 512,
            "num_beams": max(1, int(self.num_beams)),
            "num_return_sequences": max(1, int(self.num_return_sequences)),
            "do_sample": bool(self.do_sample),
            "temperature": float(self.temperature) if self.do_sample else 1.0,
            "early_stopping": True,
        }
        if self.do_sample:
            generation_kwargs.update({"top_k": 50, "top_p": 0.95})
        if generation_kwargs["num_return_sequences"] > generation_kwargs["num_beams"] and not generation_kwargs["do_sample"]:
            generation_kwargs["num_beams"] = generation_kwargs["num_return_sequences"]

        seed_value = int(self.seed) + int(self._generation_count)
        self._generation_count += 1
        try:
            self._torch.manual_seed(seed_value)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed_value)
        except Exception:
            pass

        gen_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        try:
            generator = self._torch.Generator(device=gen_device)
        except Exception:
            generator = self._torch.Generator()
        generator.manual_seed(seed_value)
        generation_kwargs["generator"] = generator

        with self._torch.no_grad():
            try:
                outputs = self.model.generate(**encoded, **generation_kwargs)
            except Exception as exc:
                message = str(exc)
                if "generator" not in generation_kwargs or "generator" not in message or "model_kwargs" not in message:
                    raise
                generation_kwargs.pop("generator", None)
                if not self._warned_generate_without_generator:
                    print("[warn] BioT5 generate() does not accept per-call generator; using global torch seed instead.")
                    self._warned_generate_without_generator = True
                outputs = self.model.generate(**encoded, **generation_kwargs)

        generated = []
        seen = set()
        for output in outputs:
            candidate = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            mol = self._candidate_to_mol(candidate)
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if not smiles or smiles in seen:
                continue
            scaffold_pass = self._matches_scaffold_constraint(mol, scaffold_constraint)
            mol.SetProp("scaffold_constraint_pass", "true" if scaffold_pass else "false")
            mol.SetProp("scaffold_constraint_mode", self.scaffold_constraint_mode)
            if scaffold_constraint and self.scaffold_constraint_mode == "filter" and not scaffold_pass:
                continue
            seen.add(smiles)
            generated.append(mol)
            if len(generated) >= self.num_return_sequences:
                break
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
        sf = _load_selfies()
        for index, mol in enumerate(smiles_list):
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            try:
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
        self.model_name = config.get("biot5_model_name", config.get("model_name", self.model_name))
        self.device = config.get("device", config.get("biot5_device", self.device))
        self.seed = int(config.get("seed", self.seed))
        self.scaffold_constraint_mode = str(config.get("scaffold_constraint_mode", self.scaffold_constraint_mode)).lower()
        print(f"[{self.backend_name}] Config applied: model={self.model_name}, device={self.device}")
