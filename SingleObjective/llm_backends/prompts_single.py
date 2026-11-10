"""Single-objective molecule editing prompts for CALLM LLM backends."""

SINGLE_TASK_DESCRIPTIONS = {
    "jnk3": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that is a greater inhibitor of JNK3.\n\n"
    ),
    "qed": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule with higher drug-likeness (QED score). The molecule "
        "should have better molecular weight, lipophilicity, hydrogen bond "
        "donors/acceptors, polar surface area, and rotatable bonds.\n\n"
    ),
    "gsk3b": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that is a greater inhibitor of GSK3 beta.\n\n"
    ),
    "drd2": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that is a greater agonist of the dopamine receptor D2.\n\n"
    ),
    "lrrk2": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that is a stronger inhibitor of LRRK2 kinase while "
        "maintaining a chemically reasonable small-molecule structure.\n\n"
    ),
    "isomers_c7h8n2o2": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that matches the molecular formula C7H8N2O2.\n\n"
    ),
    "isomers_c9h10n2o2pf2cl": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that matches the molecular formula C9H10N2O2PF2Cl.\n\n"
    ),
    "deco_hop": (
        "Generate a chemically valid molecule for the TDC PMO deco_hop "
        "objective. Preserve the exact core [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])"
        "[c;h0][c;h1]c12. Only edit the N-side aryl/heteroaryl group, the "
        "substituent on [c;h0], or the alkyl chain outside the core oxygen. "
        "Remove CS([#6])(=O)=O and [#7]-c1ccc2ncsc2c1, and preserve the "
        "reference 2D pharmacophore features."
    ),
    "scaffold_hop": (
        "Generate a chemically valid molecule for the TDC PMO scaffold_hop "
        "objective. Retain the propoxy-aryl-amino-benzothiazole decoration, "
        "remove the reference quinazoline scaffold, and preserve the 2D "
        "pharmacophore features of the reference molecule."
    ),
    "valsartan_smarts": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule that contains key Valsartan pharmacophore patterns, "
        "especially the tetrazole-biphenyl motif.\n\n"
    ),
    "celecoxib_rediscovery": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to Celecoxib, preserving key "
        "features such as the pyrazole core, trifluoromethyl group, tolyl "
        "substituent, and sulfonamide-phenyl group.\n\n"
    ),
    "thiothixene_rediscovery": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to Thiothixene, preserving key "
        "features such as the thioxanthene core, piperazine ring, and "
        "dimethylsulfonamide group.\n\n"
    ),
    "troglitazone_rediscovery": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to Troglitazone, preserving key "
        "features such as the thiazolidinedione ring, chromane moiety, and "
        "phenoxy linker.\n\n"
    ),
    "albuterol_similarity": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to Albuterol, preserving key "
        "features such as the phenylethanol core, tert-butylamine group, and "
        "hydroxymethyl substituent.\n\n"
    ),
    "mestranol_similarity": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to Mestranol, preserving key "
        "features such as the steroid backbone, methoxy group at C3, and ethynyl "
        "group at C17.\n\n"
    ),
    "median1": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to both Camphor and Menthol.\n\n"
    ),
    "median2": (
        "Definition: You are given a molecule SELFIES. Your job is to generate "
        "a SELFIES molecule structurally similar to both Tadalafil and Sildenafil.\n\n"
    ),
}

SINGLE_CONSTRAINED_DESCRIPTIONS = {
    task_name: (
        "Definition: You are given a molecule SELFIES. The scaffold {scaffold} "
        "has been identified as useful belief-space knowledge for this task. "
        "Generate a new SELFIES molecule by modifying side chains or substituents "
        "while preserving the scaffold when chemically possible.\n\n"
    )
    for task_name in SINGLE_TASK_DESCRIPTIONS
}

SINGLE_CONSTRAINED_DESCRIPTIONS["deco_hop"] = (
    "Generate a chemically valid molecule for the TDC PMO deco_hop "
    "objective. Preserve the exact core {scaffold}. Only edit the N-side "
    "aryl/heteroaryl group, the substituent on [c;h0], or the alkyl chain "
    "outside the core oxygen. Remove CS([#6])(=O)=O and "
    "[#7]-c1ccc2ncsc2c1, and preserve the reference 2D pharmacophore features."
)

SINGLE_CONSTRAINED_DESCRIPTIONS["scaffold_hop"] = (
    "Generate a chemically valid molecule for the TDC PMO scaffold_hop "
    "objective. Retain the required decoration represented by {scaffold}, "
    "avoid the reference quinazoline scaffold, and preserve the 2D "
    "pharmacophore features of the reference molecule."
)
