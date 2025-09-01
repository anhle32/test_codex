import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dsge_model import DSGEModel, PriorConfig, EstimationResult
from utils import prep_dataframe, summarize_params

st.set_page_config(page_title="VN DSGE (COVID) • Streamlit", layout="wide")

st.title("🇻🇳 VN DSGE (COVID) — Streamlit")
st.caption("Lightweight MAP estimation + IRFs (following Economies 2022)")

with st.sidebar:
    st.header("⚙️ Options")
    use_log_dq = st.checkbox("Use log-diff for Terms of Trade (dq = 100*Δlog q)", value=False)
    irf_horizon = st.slider("IRF horizon (quarters)", 4, 20, 12)

    st.subheader("📊 Shock sizes (std units)")
    shock_covid = st.number_input("COVID shock (σ)", 1.0, 5.0, 1.49, step=0.01)
    shock_u     = st.number_input("Monetary policy shock (σ)", 0.5, 3.0, 1.12, step=0.01)
    shock_z     = st.number_input("Technology shock (σ)", 0.1, 5.0, 1.0, step=0.1)

    st.subheader("🧪 Priors (means)")
    default_priors = PriorConfig.default()
    # expose a few key priors for quick tuning
    b1_mu = st.number_input("b1 (intertemporal) mean", 0.1, 0.9, float(default_priors.b1_mu), step=0.05)
    b4_mu = st.number_input("b4 (openness) mean", 0.1, 0.99, float(default_priors.b4_mu), step=0.05)
    b7_mu = st.number_input("b7 (rate smoothing) mean", 0.1, 0.99, float(default_priors.b7_mu), step=0.05)
    b13_mu = st.number_input("b13 (COVID→x) mean", 0.1, 0.99, float(default_priors.b13_mu), step=0.05)

    # strength of prior penalty (ridge-like)
    lambda_prior = st.slider("Prior penalty strength (λ)", 10.0, 5000.0, 500.0, step=10.0)

uploaded = st.file_uploader("Upload quarterly CSV (columns: r,p,e,q,us,pus). Example in sample_data/", type=["csv"])
if uploaded is None:
    st.info("Upload your CSV to proceed. Or try the bundled sample (below).")
    with st.expander("Use sample data"):
        df_sample = pd.read_csv("sample_data/sample.csv")
        st.dataframe(df_sample.head(10))
else:
    raw = pd.read_csv(uploaded)
    st.write("### Raw data preview")
    st.dataframe(raw.head(10))

    df = prep_dataframe(raw, use_log_dq=use_log_dq)
    st.write("### Preprocessed data (observables)")
    st.dataframe(df.head(10))

    # Priors from sidebar
    pri = PriorConfig(
        b1_mu=b1_mu, b3_mu=default_priors.b3_mu, b4_mu=b4_mu, b6_mu=default_priors.b6_mu, b7_mu=b7_mu, b8_mu=default_priors.b8_mu,
        b10_mu=default_priors.b10_mu, b11_mu=default_priors.b11_mu,
        b12_mu=default_priors.b12_mu, b13_mu=b13_mu, b14_mu=default_priors.b14_mu,
        sig_z_mu=0.01, sig_dq_mu=0.01, sig_u_mu=0.01, sig_pus_mu=0.01, sig_us_mu=0.01, sig_covid_mu=0.01,
        lambda_prior=lambda_prior
    )

    # Estimate (MAP)
    st.write("### 🔧 Estimating parameters (MAP)…")
    model = DSGEModel()
    result: EstimationResult = model.estimate_map(df, priors=pri)
    st.success("Done.")

    st.write("### 📑 Parameter summary")
    st.dataframe(summarize_params(result))

    # IRFs
    st.write("### 📈 Impulse Response Functions")
    irfs = model.compute_irfs(result.params, horizon=irf_horizon,
                              shocks={"covid": shock_covid, "u": shock_u, "z": shock_z})

    cols = st.columns(3)
    for i, var in enumerate(["x", "r", "p"]):
        fig, ax = plt.subplots(figsize=(5,3))
        for shock_name, series in irfs.items():
            ax.plot(series[var], label=shock_name.upper())
        ax.axhline(0, linewidth=1)
        ax.set_title(f"IRF: {var}")
        ax.set_xlabel("quarters")
        ax.legend()
        cols[i].pyplot(fig)

    # Downloads
    st.write("### ⬇️ Download")
    buf = io.StringIO()
    pd.DataFrame({k: pd.DataFrame(v) for k, v in irfs.items()}).to_csv(buf)
    st.download_button("Download IRFs CSV", buf.getvalue(), file_name="irfs.csv", mime="text/csv")

