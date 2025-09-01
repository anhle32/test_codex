# VN DSGE (COVID) — Streamlit App

This app implements a lightweight, reproducible pipeline to estimate a small-open-economy DSGE model for Vietnam (with a COVID probability shock) and visualize impulse responses (IRFs). It follows the structure in:

Nguyen, Trung Duc; **Anh Hoang Le**; Thalassinos, E.I.; Trieu, L.K. (2022) *Economies* 10(7):159.

## ✨ Features
- Upload quarterly data (1996Q1–2020Q4 or your own)
- Auto preprocessing: build `de` (Δlog exchange rate) and `dq` (Δterms of trade)
- MAP-style estimation (fast): least-squares + prior penalties (≈ Bayesian ridge)
- Posterior-like summary (point estimates) + IRFs for shocks:
  - COVID shock (covid)
  - Monetary policy shock (u)
  - Technology shock (z)
- Download results (CSV, figures)

## 🧰 Tech
- Python 3.10+
- Streamlit, NumPy, Pandas, SciPy, Matplotlib

## 📄 Input format
CSV with quarterly rows and at least these columns:
- `r`   : SBV refinancing rate (%)
- `p`   : Vietnam inflation (q/q % or consistent)
- `e`   : USD/VND level (nominal ER)
- `q`   : Terms of Trade index (level)
- `us`  : US output gap (percent)
- `pus` : US inflation (percent)
App computes:
- `de = 100 * diff(log(e))`
- `dq = diff(q)`  (or you can switch to log-diff in the sidebar)

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
