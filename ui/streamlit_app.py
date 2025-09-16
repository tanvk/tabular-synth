import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from eval.report import build_html_report

import pandas as pd
from io import StringIO
import streamlit as st

from synth.copula import CopulaGenerator
from eval.fidelity import basic_fidelity_report
from eval.utility import utility_transfer_report
from eval.privacy import basic_privacy_report

# Try to enable CTGAN if you created synth/ctgan.py
_HAS_CTGAN = False
try:
    from synth.ctgan import CTGANGenerator
    _HAS_CTGAN = True
except Exception:
    pass

st.set_page_config(page_title="Tabular Synth — Synthetic Data Engine", layout="wide")
st.title("Tabular Synth — Synthetic Data Engine (MVP)")

with st.sidebar:
    st.header("Controls")
    n_rows = st.slider("Rows to generate", 100, 50_000, 5_000, step=100)
    model_name = st.selectbox(
        "Model",
        ["Gaussian Copula"] + (["CTGAN"] if _HAS_CTGAN else []),
        index=0
    )
    st.caption("Tip: start small for faster runs.")

    if model_name == "CTGAN":
        epochs = st.slider("CTGAN epochs", 50, 1000, 150, step=25)
        batch  = st.selectbox("Batch size", [100, 200, 300, 500], index=2)
        pac    = st.selectbox("Pac", [1, 2, 5, 10], index=0)

uploaded = st.file_uploader("Upload a CSV (a small sample is fine to start)", type=["csv"])

# @st.cache_resource
# def train_model(model_name: str, df: pd.DataFrame):
#     """Train & return a fitted synthesizer, cached by (model_name, data signature)."""
#     # Invalidate cache if data changes by using a simple signature (shape + column hash)
#     _sig = (model_name, df.shape, tuple(df.columns))
#     if model_name == "Gaussian Copula":
#         gen = CopulaGenerator().fit(df)
#     else:
#         # Safe default if someone selects CTGAN without the file present
#         # in ui/streamlit_app.py inside train_model()
#         # from synth.ctgan import CTGANGenerator
#         # gen = CTGANGenerator(epochs=150, batch_size=256, pac=1).fit(df)   # pac=1 avoids the divisibility issue
#         try:
#             from synth.ctgan import CTGANGenerator
#             gen = CTGANGenerator(epochs=150, batch_size=256, pac=1).fit(df)
#         except Exception as e:
#             st.exception(e)  
#             st.warning("CTGAN not available; falling back to Gaussian Copula.")
#             gen = CopulaGenerator().fit(df)

#     return gen
@st.cache_resource
def train_model(model_name: str, df: pd.DataFrame, **kwargs):
    """
    Train & return a fitted synthesizer, cached by (model, data signature, params).
    Passing kwargs (epochs/batch/pac) ensures different settings get their own cache.
    """
    # simple signature so cache invalidates when data/params change
    _sig = (model_name, tuple(df.columns), df.shape, tuple(sorted(kwargs.items())))

    if model_name == "Gaussian Copula":
        gen = CopulaGenerator().fit(df)
    else:
        from synth.ctgan import CTGANGenerator
        gen = CTGANGenerator(
            epochs=kwargs.get("epochs", 150),
            batch_size=kwargs.get("batch", 300),
            pac=kwargs.get("pac", 1),
        ).fit(df)
    return gen

if uploaded is None:
    st.info("Upload a CSV to start.")
else:
    real = pd.read_csv(uploaded)
    st.success(f"Loaded real data: {real.shape[0]} rows × {real.shape[1]} cols")
    st.dataframe(real.head(10), use_container_width=True)

    # Gentle guard for very large uploads
    if real.shape[0] > 100_000:
        st.warning("Large dataset — consider sampling for the demo to keep things fast.")

    # Target column (optional) — you can still pick any column from the dropdown
    target = st.selectbox("Target column (optional, for utility eval)",
                          options=["(none)"] + list(real.columns), index=0)

    # Generate
    if st.button("🚀 Generate Synthetic Data"):
        # with st.spinner("Training and sampling…"):
        #     gen = train_model(model_name, real)
        #     synth = gen.sample(n_rows)
        with st.spinner("Training and sampling…"):
            if model_name == "CTGAN":
                gen = train_model(model_name, real, epochs=epochs, batch=batch, pac=pac)
            else:
                gen = train_model(model_name, real)
            synth = gen.sample(n_rows)

        st.subheader("Preview (Synthetic)")
        st.dataframe(synth.head(10), use_container_width=True)

        # Tabs for metrics
        tab1, tab2, tab3 = st.tabs(["Fidelity", "Utility", "Privacy"])

        with tab1:
            with st.spinner("Computing fidelity…"):
                fid = basic_fidelity_report(real, synth)
            st.subheader("Fidelity (headline)")
            st.json(fid["headline"])
            with st.expander("Univariate similarity (KS/TVD)"):
                st.dataframe(pd.DataFrame(fid["univariate"]), use_container_width=True)
            with st.expander("Correlation deltas (best/worst)"):
                st.write("Worst pairs")
                st.dataframe(pd.DataFrame(fid["corr_worst"]), use_container_width=True)
                st.write("Best pairs")
                st.dataframe(pd.DataFrame(fid["corr_top"]), use_container_width=True)

        with tab2:
            if target != "(none)" and target in real.columns and target in synth.columns:
                with st.spinner("Evaluating utility (synth→real vs real→real)…"):
                    util = utility_transfer_report(real, synth, target_col=target)
                st.subheader("Utility Transfer")
                st.json(util)
            else:
                st.info("Select a target column to run utility evaluation.")

        with tab3:
            with st.spinner("Running privacy checks…"):
                prv = basic_privacy_report(real, synth)
            st.subheader("Privacy")
            st.json(prv)
            ok = prv["flags"]["exact_match_ok"] and prv["flags"]["knn_min_ok"]
            if ok:
                st.success("Privacy checks passed (no exact matches; kNN distances > 0).")
            else:
                st.error("Privacy check failed, inspect exact_match_rate and kNN distances.")

        # Full HTML report
        settings = {
            "model": model_name,
            "n_rows_generated": n_rows,
            "target": None if target == "(none)" else target,
        }
        report_html = build_html_report(
            dataset_name=getattr(uploaded, "name", "uploaded.csv"),
            n_real=real.shape[0],
            n_cols=real.shape[1],
            model_name=model_name,
            settings=settings,
            fidelity=fid,
            utility=(util if target != "(none)" else None),
            privacy=prv,
        )

        from utils.run_artifacts import save_run
        # run_dir, csv_p, html_p = save_run(synth, report_html)
        # st.success(f"Saved to {run_dir}")
        
        # Save artifacts (local path inside container or host if mounted)
        res = save_run(synth, report_html)  # defaults to base_dir="artifacts"

        st.success(f"Artifacts saved to: {res['run_dir']}")
        st.write(f"- CSV: `{res['csv_path']}`")
        st.write(f"- HTML report: `{res['html_path']}`")

        st.download_button(
            "Download run report (HTML)",
            data=report_html.encode("utf-8"),
            file_name="tabular_synth_report.html",
            mime="text/html",
        )


        # Download
        csv_buf = StringIO()
        synth.to_csv(csv_buf, index=False)
        st.download_button(
            "Download synthetic.csv",
            csv_buf.getvalue().encode("utf-8"),
            file_name="synthetic.csv",
            mime="text/csv",
        )