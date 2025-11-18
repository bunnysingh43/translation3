# app_diagnose.py
import traceback
import streamlit as st

st.title("Diagnosis for Streamlit startup error")
st.write("This page will show import and startup errors below (if any).")

error_text = None

try:
    # === Put the actual app import or start logic here ===
    # If your original app is app.py with a function like `main()`,
    # do: from app import main; main()
    # Otherwise safely import the original file:
    import importlib, sys, types
    # Attempt to import the real app module (adjust path if your app is in main/app.py)
    # Try a few common module paths:
    tried = []
    success = False
    for modpath in ("app", "main.app", "main.app as app", "main.app", "main"):
        try:
            # attempt to import each; ignore those that raise
            if modpath == "app":
                import app as user_app
            elif modpath == "main.app":
                from main import app as user_app
            elif modpath == "main":
                import main as user_app
            else:
                continue
            # if import succeeded, try to call main() if exists (safe guard)
            if hasattr(user_app, "main"):
                user_app.main()
            elif hasattr(user_app, "run"):
                user_app.run()
            success = True
            st.success(f"Imported module: {user_app.__name__}")
            break
        except Exception as e:
            tried.append((modpath, repr(e)))
    if not success:
        raise ImportError("Failed to import any candidate app modules: " + str(tried))
except Exception as e:
    error_text = traceback.format_exc()

if error_text:
    st.error("Startup or import error (full traceback):")
    st.code(error_text)
else:
    st.info("No import errors detected — either the app started, or main() returned without error.")
