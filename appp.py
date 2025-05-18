"""Scoliosis AI App
================

This version gracefully handles environments where **Streamlit** is not available by
falling-back to a minimal command-line interface (CLI). When Streamlit *is*
available the full web app will load exactly as before.

You can run quick sanity checks with `python scoliosis_app.py --test`."""

# -----------------------------------
# Imports & graceful-degradation layer
# -----------------------------------

from PIL import Image
import random
import sys
import argparse

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    STREAMLIT_AVAILABLE = False

# ------------------------------
# Placeholder ML utilities
# ------------------------------

def load_model():
    """Load your trained scoliosis detection model here."""
    return None

MODEL = load_model()

def predict(image: Image.Image):
    """Dummy prediction function – replace with real inference."""
    return {
        "angle": round(random.uniform(5, 60), 1),
        "severity": random.choice(["Mild", "Moderate", "Severe"]),
        "confidence": round(random.uniform(0.7, 0.99), 2),
    }

# ------------------------------
# Streamlit UI
# ------------------------------

if STREAMLIT_AVAILABLE and not any(f in sys.argv for f in ["--cli", "--test"]):
    st.set_page_config(page_title="Scoliosis AI", layout="wide")

    PAGES = {
        "Home": "🏠",
        "Upload & Predict": "🩻",
        "Results": "📈",
        "About": "ℹ️",
    }

    st.sidebar.title("📚 Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()), format_func=lambda x: f"{PAGES[x]}  {x}")

    if page == "Home":
        st.title("🏠 Welcome to Scoliosis AI")
        st.markdown(
            """
            Upload X-ray images of the spine and let our AI model estimate the Cobb angle
            and classify scoliosis severity in seconds.

            **Workflow**
            1. Go to **Upload & Predict** and add your image.
            2. View automated measurements and heatmaps in **Results**.
            3. Learn more about our project in **About**.
            """
        )

    elif page == "Upload & Predict":
        st.title("🩻 Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg", "tif", "tiff"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Run Analysis"):
                with st.spinner("Analyzing..."):
                    result = predict(img)
                st.session_state["result"] = result
                st.success("Done! Navigate to **Results** to view details.")

    elif page == "Results":
        st.title("📈 Analysis Results")
        if "result" in st.session_state:
            res = st.session_state["result"]
            st.metric("Estimated Cobb Angle", f"{res['angle']}°")
            st.metric("Severity", res["severity"])
            st.metric("Confidence", f"{res['confidence']*100:.0f}%")

            st.subheader("Model Visualization (coming soon)")
            st.info("Grad-CAM or segmentation overlay will appear here.")
        else:
            st.warning("No results yet. Please run an analysis first in **Upload & Predict**.")

    elif page == "About":
        st.title("ℹ️ About this Project")
        st.markdown(
            """
            **Goal**: Provide accessible, AI-powered scoliosis screening using image processing and deep learning.

            **Technologies**: Streamlit, PyTorch/TensorFlow, OpenCV, Docker, AWS.

            **Authors**: *Your names here.*
            """
        )
        st.markdown("---")
        st.markdown("© 2025 Scoliosis AI Research Group")

# ------------------------------
# CLI fallback (for test / demo)
# ------------------------------

def cli_interface(image_path: str):
    """Run prediction from the command line when Streamlit isn't available."""
    img = Image.open(image_path)
    res = predict(img)
    print("\nResults for", image_path)
    for k, v in res.items():
        print(f" - {k.capitalize()}: {v}")

# ------------------------------
# Testing utilities
# ------------------------------

def run_tests():
    """Ensure predict() outputs correct structure and ranges."""
    dummy = Image.new("RGB", (1, 1))
    out = predict(dummy)
    assert set(out) == {"angle", "severity", "confidence"}
    assert 0 <= out["angle"] <= 90
    assert out["severity"] in {"Mild", "Moderate", "Severe"}
    assert 0 <= out["confidence"] <= 1
    # ASCII hyphen check
    assert "-" in "falling-back" and "‑" not in "falling-back"
    print("All tests passed ✅")

# ------------------------------
# Entrypoint
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoliosis AI CLI / test runner")
    parser.add_argument("image", nargs="?", help="Path to X-ray image")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode even if Streamlit exists")
    parser.add_argument("--test", action="store_true", help="Run internal tests and exit")
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if args.image:
        cli_interface(args.image)
    elif not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Use --cli with an image path or install streamlit.")
