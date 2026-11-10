"""Multi-objective molecule editing prompts for CALLM LLM backends."""

MULTI_TASK_DESCRIPTIONS = {
    "sitagliptin_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances high Sitagliptin similarity, TPSA "
        "around 80-100, LogP in range 1-3, and 2-3 fluorine atoms.\n\n"
    ),
    "zaleplon_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances high Zaleplon similarity and the "
        "target formula C19H17N3O2.\n\n"
    ),
    "amlodipine_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances high Amlodipine similarity and an "
        "appropriate ring count near 3.\n\n"
    ),
    "fexofenadine_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances Fexofenadine similarity, TPSA around "
        "90 or higher, and LogP around 4 or lower.\n\n"
    ),
    "osimertinib_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances Osimertinib similarity, TPSA around "
        "100 or higher, and LogP around 1 or lower.\n\n"
    ),
    "perindopril_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances Perindopril similarity and about 2 "
        "aromatic rings.\n\n"
    ),
    "ranolazine_mpo": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that balances Ranolazine similarity, LogP <= 7, "
        "TPSA >= 95, and one fluorine atom.\n\n"
    ),
    "lrrk2_wt_activity_mpo": (
        "Definition: You are given a molecule SELFIES. Generate a valid SELFIES "
        "molecule for LRRK2 WT stage-1 optimization. Optimize only these scored "
        "objectives: AD-penalized LRRK2 QSAR activity and a weak QED-like drug-likeness "
        "regularizer. Avoid salts, disconnected fragments, metals, and abnormal charges. "
        "Return only one valid SELFIES string.\n\n"
    ),
    "lrrk2_wt_lead_repair_mpo": (
        "Definition: You are given a molecule SELFIES from the LRRK2 WT lead-repair "
        "workflow. Generate a valid SELFIES molecule for CALLM stage-2 multi-objective "
        "optimization. Optimize exactly these four scored Pareto objectives: "
        "(1) QSAR AD-penalized LRRK2 activity, (2) BBB/CNS-likeness, "
        "(3) logP window fit, and (4) TPSA window fit. Raw predicted pIC50 is "
        "reporting metadata only and may be used only as a same-score ranking "
        "tie-breaker; it is not a Pareto objective. Do not optimize or mention "
        "unscored objectives. Avoid salts, disconnected fragments, metals, and "
        "abnormal charges. Return only one valid SELFIES string."
    ),
}

MULTI_CONSTRAINED_DESCRIPTIONS = {
    task_name: (
        description.rstrip()
        + "\n\nBelief-space scaffold constraint: {scaffold}. Preserve this scaffold "
        "or substructure when chemically possible while improving the task objectives "
        "above. Prefer side-chain or substituent edits over destroying the core.\n\n"
    )
    for task_name, description in MULTI_TASK_DESCRIPTIONS.items()
}
