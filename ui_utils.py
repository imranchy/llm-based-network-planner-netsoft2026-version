# ui_utils.py
try:
    import streamlit as st  # type: ignore
except Exception:  # allow non-UI usage
    st = None  # type: ignore
import time

def timed(label="Operation"):
    """
    Decorator to measure execution time of a Streamlit function.
    Example:
        @timed("Query Execution")
        def run_query(): ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            st.info(f"⏱️ {label} took {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator

def progress_update(step, total, message="Processing..."):
    """
    Display a progress bar in Streamlit.
    Example:
        for i in range(total):
            progress_update(i, total, "Loading data")
    """
    progress = st.progress(step / total)
    st.caption(f"{message} ({step}/{total})")
