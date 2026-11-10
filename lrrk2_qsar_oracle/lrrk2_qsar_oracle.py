from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


MODEL_FILE = "lrrk2_qsar_oracle_model.pkl"
METADATA_FILE = "lrrk2_qsar_oracle_metadata.json"
TRAIN_FP_FILE = "lrrk2_train_fingerprints.npy"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = (
    SCRIPT_DIR if (SCRIPT_DIR / MODEL_FILE).exists() else SCRIPT_DIR / "lrrk2_qsar_oracle"
)

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
MORGAN_FEATURE_NAMES = [f"Morgan_bit_{idx}" for idx in range(MORGAN_N_BITS)]
METAL_ATOMIC_NUMBERS = {
    3,
    4,
    11,
    12,
    13,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    55,
    56,
    57,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
}
ORACLE_OUTPUT_COLUMNS = [
    "Name",
    "SMILES",
    "canonical_SMILES",
    "valid_smiles",
    "pred_pIC50",
    "pred_IC50_nM",
    "activity_score",
    "qsar_activity_score_ad_penalized",
    "nearest_train_similarity",
    "applicability_domain",
    "oracle_confidence",
    "n_fragments",
    "is_multifragment",
    "contains_metal",
    "formal_charge",
    "has_abnormal_charge",
    "structure_warning",
    "MW",
    "logP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "RingCount",
    "QED",
]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def canonicalize_smiles(smiles: object) -> tuple[str, Chem.Mol | None]:
    """Parse a SMILES string and return canonical SMILES plus RDKit molecule."""
    if smiles is None or pd.isna(smiles):
        return "", None

    text = str(smiles).strip()
    if not text:
        return "", None

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return "", None

    canonical = Chem.MolToSmiles(mol, canonical=True)
    return canonical, mol


def mol_to_morgan_fp(
    mol: Chem.Mol,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_N_BITS,
) -> np.ndarray:
    """Convert an RDKit molecule to a binary Morgan fingerprint array."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fingerprints_to_model_input(fingerprints: np.ndarray) -> pd.DataFrame:
    """Represent Morgan fingerprints with stable feature names for LightGBM."""
    if fingerprints.ndim == 1:
        fingerprints = fingerprints.reshape(1, -1)
    return pd.DataFrame(fingerprints, columns=MORGAN_FEATURE_NAMES)


def calculate_descriptors(mol: Chem.Mol | None) -> dict[str, float]:
    """Calculate lightweight RDKit descriptors for reporting."""
    if mol is None:
        return {
            "MW": np.nan,
            "logP": np.nan,
            "TPSA": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "RotB": np.nan,
            "RingCount": np.nan,
            "QED": np.nan,
        }

    return {
        "MW": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotB": float(Lipinski.NumRotatableBonds(mol)),
        "RingCount": float(Lipinski.RingCount(mol)),
        "QED": float(QED.qed(mol)),
    }


def calculate_structure_flags(mol: Chem.Mol | None) -> dict[str, object]:
    """Flag structural issues that should reduce CALLM optimization reward."""
    if mol is None:
        return {
            "n_fragments": np.nan,
            "is_multifragment": False,
            "contains_metal": False,
            "formal_charge": np.nan,
            "has_abnormal_charge": False,
            "structure_warning": "invalid",
        }

    n_fragments = len(
        Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    )
    formal_charge = Chem.GetFormalCharge(mol)
    contains_metal = any(
        atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms()
    )
    is_multifragment = n_fragments > 1
    # The main rule is absolute formal charge > 1. Standalone charged ions such
    # as [NH4+] are also unsafe as generated drug-like molecules, so flag them.
    has_abnormal_charge = abs(formal_charge) > 1 or (
        mol.GetNumAtoms() == 1 and formal_charge != 0
    )

    warnings = []
    if is_multifragment:
        warnings.append("multifragment")
    if contains_metal:
        warnings.append("contains_metal")
    if has_abnormal_charge:
        warnings.append("abnormal_charge")

    return {
        "n_fragments": int(n_fragments),
        "is_multifragment": bool(is_multifragment),
        "contains_metal": bool(contains_metal),
        "formal_charge": int(formal_charge),
        "has_abnormal_charge": bool(has_abnormal_charge),
        "structure_warning": ";".join(warnings),
    }


def calculate_nearest_similarity(query_fp: np.ndarray, train_fps: np.ndarray) -> float:
    """Calculate maximum Tanimoto similarity to the training fingerprints."""
    if query_fp is None or train_fps.size == 0:
        return np.nan

    query = query_fp.astype(np.float32, copy=False)
    train = train_fps.astype(np.float32, copy=False)
    intersections = train @ query
    query_sum = float(query.sum())
    train_sums = train.sum(axis=1)
    unions = train_sums + query_sum - intersections

    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    return float(np.max(similarities))


def calculate_nearest_similarities(
    query_fps: np.ndarray,
    train_fps: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Calculate maximum Tanimoto similarity for a batch of fingerprints."""
    if len(query_fps) == 0:
        return np.array([], dtype=np.float32)

    train = train_fps.astype(np.float32, copy=False)
    train_sums = train.sum(axis=1)
    results = []

    for start in range(0, len(query_fps), chunk_size):
        query = query_fps[start : start + chunk_size].astype(np.float32, copy=False)
        intersections = query @ train.T
        query_sums = query.sum(axis=1)
        unions = query_sums[:, None] + train_sums[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float32),
            where=unions > 0,
        )
        results.append(similarities.max(axis=1))

    return np.concatenate(results).astype(np.float32, copy=False)


def pIC50_to_activity_score(pred_pic50: float) -> float:
    """Convert predicted pIC50 to CALLM's 0-1 maximization score."""
    if pred_pic50 is None or pd.isna(pred_pic50):
        return 0.0
    return float(np.clip((float(pred_pic50) - 6.0) / 3.0, 0.0, 1.0))


def ad_penalized_activity_score(
    activity_score: float,
    applicability_domain: str,
) -> float:
    """Apply applicability-domain confidence penalties to the raw QSAR score."""
    if activity_score is None or pd.isna(activity_score):
        return 0.0

    score = float(activity_score)
    if applicability_domain == "high":
        return float(np.clip(score, 0.0, 1.0))
    if applicability_domain == "medium":
        return float(np.clip(min(score * 0.85, 0.70), 0.0, 1.0))
    if applicability_domain == "low":
        return float(np.clip(min(score * 0.55, 0.45), 0.0, 1.0))
    if applicability_domain == "very_low":
        return float(np.clip(min(score * 0.30, 0.25), 0.0, 1.0))
    return 0.0


def apply_structure_penalty(score: float, flags: dict[str, object]) -> float:
    """Further penalize salts, metals, and molecules with abnormal charge."""
    if score is None or pd.isna(score):
        return 0.0

    penalized = float(score)
    if flags.get("is_multifragment", False):
        penalized = min(penalized * 0.2, 0.10)
    if flags.get("contains_metal", False):
        penalized = min(penalized * 0.2, 0.10)
    if flags.get("has_abnormal_charge", False):
        penalized = min(penalized * 0.5, 0.20)

    return float(np.clip(penalized, 0.0, 1.0))


def confidence_from_similarity(similarity: float) -> tuple[str, str]:
    """Map nearest-neighbor similarity to applicability domain and confidence."""
    if similarity is None or pd.isna(similarity):
        return "invalid", "invalid"
    if similarity >= 0.60:
        return "high", "high"
    if similarity >= 0.40:
        return "medium", "medium"
    if similarity >= 0.30:
        return "low", "low"
    return "very_low", "very_low"


def configure_model_for_scoring(model, n_jobs: int = 1):
    """Limit prediction threads for stable CALLM batch scoring.

    The trained LightGBM artifact was saved with n_jobs=-1. That is fine for
    offline modeling, but embedding the oracle inside CALLM can oversubscribe
    CPU threads or stall small scoring batches. Setting n_jobs=1 does not change
    model predictions; it only makes inference resource usage predictable.
    """
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=n_jobs)
        except (TypeError, ValueError):
            pass
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = n_jobs
        except Exception:
            pass
    return model


def validate_oracle_artifacts(model, metadata: dict, train_fps: np.ndarray) -> None:
    """Validate that model, metadata, and fingerprint artifacts agree."""
    metadata_n_bits = metadata.get("morgan_nBits")
    metadata_radius = metadata.get("morgan_radius")
    if metadata_n_bits != MORGAN_N_BITS:
        raise ValueError(
            f"Artifact mismatch: metadata morgan_nBits={metadata_n_bits}, "
            f"expected {MORGAN_N_BITS}."
        )
    if metadata_radius != MORGAN_RADIUS:
        raise ValueError(
            f"Artifact mismatch: metadata morgan_radius={metadata_radius}, "
            f"expected {MORGAN_RADIUS}."
        )
    if train_fps.ndim != 2:
        raise ValueError(
            f"Artifact mismatch: train_fps must be 2D, got shape {train_fps.shape}."
        )
    if train_fps.shape[1] != MORGAN_N_BITS:
        raise ValueError(
            f"Artifact mismatch: train_fps has {train_fps.shape[1]} columns, "
            f"expected {MORGAN_N_BITS}."
        )
    if not np.all((train_fps == 0) | (train_fps == 1)):
        raise ValueError("Artifact mismatch: train_fps must contain only 0/1 values.")

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != MORGAN_N_BITS:
        raise ValueError(
            f"Artifact mismatch: model.n_features_in_={n_features}, "
            f"expected {MORGAN_N_BITS}."
        )

    training_sample_count = metadata.get("training_sample_count")
    if (
        training_sample_count is not None
        and int(training_sample_count) != train_fps.shape[0]
    ):
        raise ValueError(
            "Artifact mismatch: metadata training_sample_count="
            f"{training_sample_count}, but train_fps has {train_fps.shape[0]} rows."
        )


def load_oracle(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict:
    """Load model, metadata, and training fingerprints."""
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILE
    metadata_path = model_dir / METADATA_FILE
    train_fp_path = model_dir / TRAIN_FP_FILE

    missing = [
        str(path)
        for path in [model_path, metadata_path, train_fp_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing oracle artifact(s). Run train_final_lrrk2_qsar_oracle.py first: "
            + ", ".join(missing)
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load LightGBM oracle model. Ensure lightgbm is installed."
        ) from exc

    model = configure_model_for_scoring(model, n_jobs=1)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_fps = np.load(train_fp_path)
    validate_oracle_artifacts(model, metadata, train_fps)

    return {
        "model": model,
        "metadata": metadata,
        "train_fps": train_fps,
        "model_dir": model_dir,
    }


def invalid_prediction_record(name: str, smiles: object) -> dict:
    """Return a complete output record for an invalid SMILES."""
    return {
        "Name": name,
        "SMILES": "" if smiles is None or pd.isna(smiles) else str(smiles),
        "canonical_SMILES": "",
        "valid_smiles": False,
        "pred_pIC50": np.nan,
        "pred_IC50_nM": np.nan,
        "activity_score": 0.0,
        "qsar_activity_score_ad_penalized": 0.0,
        "nearest_train_similarity": np.nan,
        "applicability_domain": "invalid",
        "oracle_confidence": "invalid",
        **calculate_structure_flags(None),
        **calculate_descriptors(None),
    }


def predict_lrrk2_activity(smiles: object, oracle: dict, name: str = "") -> dict:
    """Predict LRRK2 activity and oracle confidence for one molecule."""
    canonical, mol = canonicalize_smiles(smiles)
    if mol is None:
        return invalid_prediction_record(name, smiles)

    fp = mol_to_morgan_fp(mol)
    pred_pic50 = float(oracle["model"].predict(fingerprints_to_model_input(fp))[0])
    pred_ic50_nm = float(10 ** (9.0 - pred_pic50))
    raw_score = pIC50_to_activity_score(pred_pic50)
    nearest_similarity = calculate_nearest_similarity(fp, oracle["train_fps"])
    applicability_domain, oracle_confidence = confidence_from_similarity(nearest_similarity)
    structure_flags = calculate_structure_flags(mol)
    ad_score = ad_penalized_activity_score(raw_score, applicability_domain)
    ad_score = apply_structure_penalty(ad_score, structure_flags)

    return {
        "Name": name,
        "SMILES": str(smiles),
        "canonical_SMILES": canonical,
        "valid_smiles": True,
        "pred_pIC50": pred_pic50,
        "pred_IC50_nM": pred_ic50_nm,
        "activity_score": raw_score,
        "qsar_activity_score_ad_penalized": ad_score,
        "nearest_train_similarity": nearest_similarity,
        "applicability_domain": applicability_domain,
        "oracle_confidence": oracle_confidence,
        **structure_flags,
        **calculate_descriptors(mol),
    }


def generated_names(n_rows: int) -> list[str]:
    """Generate stable molecule names when no name column is available."""
    return [f"mol_{idx:06d}" for idx in range(1, n_rows + 1)]


def resolve_molecule_names(values: pd.Series, n_rows: int) -> list[str]:
    """Resolve molecule names, replacing missing or blank names with stable IDs."""
    fallback_names = generated_names(n_rows)
    resolved_names = []
    for value, fallback in zip(values.tolist(), fallback_names):
        if value is None or pd.isna(value) or not str(value).strip():
            resolved_names.append(fallback)
        else:
            resolved_names.append(str(value))
    return resolved_names


def score_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    name_col: str | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    oracle: dict | None = None,
) -> pd.DataFrame:
    """Score a dataframe of molecules while preserving input columns."""
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {smiles_col}")

    # Delay loading the model until at least one valid molecule needs scoring.
    # This lets empty batches and all-invalid batches return cleanly without
    # paying LightGBM/joblib loading cost, which is useful inside CALLM loops.
    if name_col and name_col in df.columns:
        names = resolve_molecule_names(df[name_col], len(df))
    elif "Name" in df.columns:
        names = resolve_molecule_names(df["Name"], len(df))
    else:
        names = generated_names(len(df))

    records = [None] * len(df)
    valid_items = []
    for idx, (_, row) in enumerate(df.iterrows()):
        smiles = row[smiles_col]
        canonical, mol = canonicalize_smiles(smiles)
        if mol is None:
            records[idx] = invalid_prediction_record(names[idx], smiles)
            continue

        valid_items.append(
            {
                "idx": idx,
                "name": names[idx],
                "smiles": smiles,
                "canonical": canonical,
                "mol": mol,
                "fp": mol_to_morgan_fp(mol),
            }
        )

    if valid_items:
        if oracle is None:
            oracle = load_oracle(model_dir)

        fps = np.vstack([item["fp"] for item in valid_items]).astype(np.float32)
        pred_pic50_values = oracle["model"].predict(
            fingerprints_to_model_input(fps)
        ).astype(float)
        nearest_values = calculate_nearest_similarities(fps, oracle["train_fps"])

        for item, pred_pic50, nearest_similarity in zip(
            valid_items,
            pred_pic50_values,
            nearest_values,
        ):
            applicability_domain, oracle_confidence = confidence_from_similarity(
                float(nearest_similarity)
            )
            raw_score = pIC50_to_activity_score(float(pred_pic50))
            structure_flags = calculate_structure_flags(item["mol"])
            ad_score = ad_penalized_activity_score(raw_score, applicability_domain)
            ad_score = apply_structure_penalty(ad_score, structure_flags)
            records[item["idx"]] = {
                "Name": item["name"],
                "SMILES": str(item["smiles"]),
                "canonical_SMILES": item["canonical"],
                "valid_smiles": True,
                "pred_pIC50": float(pred_pic50),
                "pred_IC50_nM": float(10 ** (9.0 - pred_pic50)),
                "activity_score": raw_score,
                "qsar_activity_score_ad_penalized": ad_score,
                "nearest_train_similarity": float(nearest_similarity),
                "applicability_domain": applicability_domain,
                "oracle_confidence": oracle_confidence,
                **structure_flags,
                **calculate_descriptors(item["mol"]),
            }

    # Keep an explicit schema even for empty inputs, and avoid pandas
    # index-alignment bugs when CALLM passes filtered/sliced DataFrames with
    # non-default indices. Assign by position so every scored row stays matched
    # to its original input row.
    scored = pd.DataFrame(
        records,
        columns=ORACLE_OUTPUT_COLUMNS,
        index=df.index,
    )
    output = df.copy()
    for column in ORACLE_OUTPUT_COLUMNS:
        output[column] = scored[column].to_numpy(copy=False)

    ordered_columns = [
        *ORACLE_OUTPUT_COLUMNS,
        *[column for column in output.columns if column not in ORACLE_OUTPUT_COLUMNS],
    ]
    return output[ordered_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score molecules with the LRRK2 QSAR activity oracle."
    )
    parser.add_argument("--input", required=True, help="Input CSV file path.")
    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="Column containing SMILES strings. Default: SMILES.",
    )
    parser.add_argument(
        "--name_col",
        default=None,
        help="Optional molecule name column. Auto-generates names if absent.",
    )
    parser.add_argument(
        "--output",
        default="lrrk2_qsar_scored.csv",
        help="Output CSV file path. Default: lrrk2_qsar_scored.csv.",
    )
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Oracle artifact directory. Default: lrrk2_qsar_oracle/.",
    )
    return parser.parse_args()


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, keep_default_na=False)
    scored = score_dataframe(
        df,
        smiles_col=args.smiles_col,
        name_col=args.name_col,
        model_dir=args.model_dir,
    )
    scored.to_csv(output_path, index=False)

    valid_count = int(scored["valid_smiles"].sum())
    invalid_count = int((~scored["valid_smiles"].astype(bool)).sum())
    print("LRRK2 QSAR oracle scoring completed")
    print(f"Input rows: {len(scored)}")
    print(f"Valid SMILES: {valid_count}")
    print(f"Invalid SMILES: {invalid_count}")
    print(f"Output path: {output_path.resolve()}")


if __name__ == "__main__":
    main()
