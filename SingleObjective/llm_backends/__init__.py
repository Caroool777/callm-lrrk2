"""BioT5 molecule editor factory for CALLM."""

import os
from typing import Dict, Optional

from .abstract_backend import LLMBackend
from .biot5_backend import BioT5Backend


class LLMBackendFactory:
    """Create BioT5 molecule-editing backends by name."""

    AVAILABLE_BACKENDS = {
        "biot5": BioT5Backend,
    }

    @staticmethod
    def normalize_backend_name(backend_name: Optional[str]) -> str:
        return (backend_name or "biot5").lower().strip()

    @staticmethod
    def create(backend_name: str = "biot5", **kwargs) -> LLMBackend:
        backend_name = LLMBackendFactory.normalize_backend_name(backend_name)
        if backend_name not in LLMBackendFactory.AVAILABLE_BACKENDS:
            available = ", ".join(sorted(LLMBackendFactory.AVAILABLE_BACKENDS))
            raise ValueError(
                f"Unknown LLM backend '{backend_name}'. CALLM LRRK2 supports only: {available}"
            )
        return LLMBackendFactory.AVAILABLE_BACKENDS[backend_name](**kwargs)

    @staticmethod
    def get_default_backend() -> str:
        return LLMBackendFactory.normalize_backend_name(os.environ.get("LLM_BACKEND", "biot5"))

    @staticmethod
    def backend_name_from_config(config: Dict) -> str:
        return LLMBackendFactory.normalize_backend_name(
            config.get("llm_backend") or config.get("llm_provider") or os.environ.get("LLM_BACKEND", "biot5")
        )

    @staticmethod
    def create_from_config(config: Dict) -> LLMBackend:
        backend = LLMBackendFactory.create(
            LLMBackendFactory.backend_name_from_config(config),
            num_beams=config.get("llm_num_beams", config.get("biot5_num_beams", 5)),
            num_return_sequences=config.get(
                "llm_num_return_sequences",
                config.get("biot5_num_return_sequences", 3),
            ),
            temperature=config.get("llm_temperature", config.get("biot5_temperature", 0.8)),
            do_sample=config.get("llm_do_sample", config.get("biot5_do_sample", True)),
        )
        backend.apply_config(config)
        return backend


def create_molecule_editor(
    backend: str = "biot5",
    task: Optional[str] = None,
    **kwargs,
) -> LLMBackend:
    editor = LLMBackendFactory.create(backend, **kwargs)
    if task:
        editor.set_task(task)
    return editor


class BioT5:
    """Backward-compatible wrapper for old `from biot5 import BioT5` code."""

    def __init__(self, *args, **kwargs):
        backend_name = kwargs.pop("backend", os.environ.get("LLM_BACKEND", "biot5"))
        legacy_arg_names = ("num_beams", "num_return_sequences", "temperature", "do_sample", "device")
        for name, value in zip(legacy_arg_names, args):
            kwargs.setdefault(name, value)
        self._backend = LLMBackendFactory.create(backend_name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def set_task(self, task_name: str) -> None:
        self._backend.set_task(task_name)

    def edit(self, smiles_list, scaffold_constraints=None):
        return self._backend.edit(smiles_list, scaffold_constraints=scaffold_constraints)

    def apply_config(self, config: Dict) -> None:
        self._backend.apply_config(config)


__all__ = [
    "BioT5",
    "BioT5Backend",
    "LLMBackend",
    "LLMBackendFactory",
    "create_molecule_editor",
]
