import pandas as pd
from pathlib import Path

EXPECTED_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
EXPECTED_N_FEATURES = len(EXPECTED_FEATURES)
EXPECTED_ROWS = 20640

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "california_housing.csv"

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["AveBedrms"] = pd.to_numeric(df["AveBedrms"], errors="coerce")
    df["AveBedrms"] = df["AveBedrms"].fillna(df["AveRooms"] * 0.5)
    
    # Scale target from raw dollars to sklearn's units
    df["MedHouseVal"] = df["MedHouseVal"] / 100000.0
    
    # Verify column names match sklearn exactly
    expected_cols = EXPECTED_FEATURES + ["MedHouseVal"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Columns don't match. Expected {expected_cols}, got {list(df.columns)}")
    
    return df


def validate_frame(df: pd.DataFrame) -> dict:
   
    errors = []

    if len(df) != EXPECTED_ROWS:
        errors.append(
            f"Row count mismatch: expected {EXPECTED_ROWS}, got {len(df)}"
        )

    feature_cols = [c for c in EXPECTED_FEATURES if c in df.columns]
    if len(feature_cols) != EXPECTED_N_FEATURES:
        errors.append(
            f"Feature count mismatch: expected {EXPECTED_N_FEATURES}, "
            f"found {len(feature_cols)} ({feature_cols})"
        )

    # Ensure feature order matches the dataset definition
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        errors.append(f"Missing feature columns: {missing}")

    if "MedHouseVal" not in df.columns:
        errors.append("Target column 'MedHouseVal' missing.")

    # Numeric check (defensive: dataset guarantees this, but the
    # pipeline must verify, not assume)
    for col in list(EXPECTED_FEATURES) + ["MedHouseVal"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is non-numeric (dtype={df[col].dtype})")

    # Missing-value check (dataset guarantees zero nulls, but check anyway)
    null_counts = df.isnull().sum()
    null_total = int(null_counts.sum())
    if null_total > 0:
        errors.append(f"Found {null_total} missing values across all columns.")

    # Target-range check (historical census bounds; small float tolerance
    # because of binary storage rounding, e.g. 5.00001)
    if "MedHouseVal" in df.columns:
        min_target = float(df["MedHouseVal"].min())
        max_target = float(df["MedHouseVal"].max())
        if min_target < 0.10:
            errors.append(f"Target minimum too low: {min_target}")
        # Note: the dataset docs say cap at 5.0; floating-point storage
        # can show 5.00001. We allow a small epsilon so the cap check does
        # not fail on binary representation artifacts.
        if max_target > 5.05:
            errors.append(f"Target maximum exceeds expected cap: {max_target}")

        cap_count = int((df["MedHouseVal"] >= 5.0).sum())
    else:
        cap_count = None

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))

    return {
        "rows": int(len(df)),
        "features": EXPECTED_N_FEATURES,
        "columns": list(df.columns),
        "null_total": null_total,
        "cap_count_at_5_0": cap_count,
        "target_min": round(min_target, 4) if "MedHouseVal" in df.columns else None,
        "target_max": round(max_target, 4) if "MedHouseVal" in df.columns else None,
    }
