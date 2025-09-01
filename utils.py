import pandas as pd
import numpy as np

def prep_dataframe(raw: pd.DataFrame, use_log_dq=False) -> pd.DataFrame:
    df = raw.copy()
    # Ensure columns exist
    required = ["r","p","e","q","us","pus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_index()
    # Build de, dq
    df["de"] = 100 * (np.log(df["e"]).diff())
    if use_log_dq:
        df["dq"] = 100 * (np.log(df["q"]).diff())
    else:
        df["dq"] = df["q"].diff()

    # Drop first NA
    df = df.dropna().reset_index(drop=True)
    # Keep only observables
    return df[["r","p","de","dq","us","pus"]]

def summarize_params(result) -> pd.DataFrame:
    d = result.params
    out = pd.DataFrame({
        "param": list(d.keys()),
        "estimate": [float(d[k]) for k in d.keys()]
    }).sort_values("param")
    return out
