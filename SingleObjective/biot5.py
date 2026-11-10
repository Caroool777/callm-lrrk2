"""Backward-compatible BioT5 module.

The CALLM model editor is now implemented through ``SingleObjective.llm_backends``.
This module intentionally keeps only the old import surface:

    from biot5 import BioT5

New code should prefer:

    from llm_backends import create_molecule_editor
"""

from llm_backends import BioT5, create_molecule_editor

__all__ = ["BioT5", "create_molecule_editor"]
