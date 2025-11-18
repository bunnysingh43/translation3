# app_safe.py
import streamlit as st
import traceback
import time

st.set_page_config(page_title="App Safe - debug", layout="wide")

st.title("Safe app (non-blocking) — debug mode")
st.caption("This file lazy-loads heavy libs and prevents import-time hangs.")

st.write("DEBUG: UI rendered successfully ✅")

# Simple debug markers so you can see where things run
st.write("DEBUG: timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))

# Helper: lazy import functions
def lazy_import_fitz():
    try:
        import fitz  # pymupdf
        return fitz, None
    except Exception as e:
        return None, traceback.format_exc()

def lazy_import_pytesseract():
    try:
        import pytesseract
        return pytesseract, None
    except Exception as e:
        return None, traceback.format_exc()

# A safe wrapper that executes the heavy pipeline only on demand
def run_heavy_pipeline(show_output=True):
    st.info("Starting heavy pipeline — lazy imports happening now...")
    out = {"steps": []}

    fitz, err_fitz = lazy_import_fitz()
    if err_fitz:
        out["steps"].append(("pymupdf import failed", err_fitz))
        st.error("pymupdf import failed — see details below.")
        st.code(err_fitz)
        return out

    pyt, err_pyt = lazy_import_pytesseract()
    if err_pyt:
        out["steps"].append(("pytesseract import failed", err_pyt))
        st.error("pytesseract import failed — see details below.")
        st.code(err_pyt)
        return out

    # If we reach here, both imports worked.
    out["steps"].append(("imports_ok", "pymupdf & pytesseract imported"))

    # Example safe usage (do NOT run heavy loops at import-time)
    try:
        # placeholder: replace with the smallest reproducible snippet of your real code
        out["steps"].append(("example_action", "running small example"))
        # For example, create an empty PDF doc or just check versions
        st.write("pymupdf version:", getattr(fitz, "__doc__", "")[:200])
        st.write("pytesseract version:", getattr(pyt, "__version__", "unknown"))
        out["steps"].append(("example_action_complete", "done"))
    except Exception as e:
        tb = traceback.format_exc()
        out["steps"].append(("runtime_exception", tb))
        st.error("Runtime exception during pipeline. See details:")
        st.code(tb)
        return out

    if show_output:
        st.success("Heavy pipeline finished (simulated). See steps below.")
        for s in out["steps"]:
            st.write("•", s[0], ":", s[1] if len(s) > 1 else "")
    return out

# UI: button to run heavy work
st.write("---")
st.subheader("Manual run controls")
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Run OCR / PDF pipeline (lazy)"):
        run_heavy_pipeline()
with col2:
    st.write("This run will only start heavy imports when you press the button.")
    st.write("If imports fail, the page will show the full traceback — no shimmer.")

st.write("---")
st.subheader("Quick checks")
st.write("- Repository path: check that `app.py` still exists in repo root or `main/app.py`.")
st.write("- If you need the actual app UI, I can patch `app.py` similarly to be lazy and non-blocking.")
