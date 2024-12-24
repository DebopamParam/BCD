import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import os
import streamlit.components.v1 as components
import json

st.set_page_config(
    page_title="Inference | Prediction",
    page_icon="🩺",
)


# --- Existing code (CustomStandardScaler, loading scaler and model, preprocessing functions) ---
class CustomStandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Transforms the input data using the saved mean and std deviation.
        """
        data = np.array(data)  # Ensure input is a NumPy array
        return (data - self.mean) / self.std

    def inverse_transform(self, scaled_data):
        """
        Reverts the scaling transformation.
        """
        scaled_data = np.array(scaled_data)
        return (scaled_data * self.std) + self.mean


@st.cache_resource
def load_age_scaler():
    # Load the pre-trained scaler
    try:
        age_scaler = joblib.load("static/artifacts/custom_scaler.pkl")
        return age_scaler
    except FileNotFoundError:
        st.error(
            "Error: custom_scaler.pkl not found. Please ensure it is in the static/artifacts/ directory."
        )
        st.stop()
    return None  # Added to handle the case when exception happens


@st.cache_resource
def load_tflite_model():
    # Load TFLite model and allocate tensors.
    try:
        interpreter = tf.lite.Interpreter(
            model_path="static/artifacts/model_float16_quant.tflite"
        )
        interpreter.allocate_tensors()
        return interpreter
    except FileNotFoundError:
        st.error(
            "Error: model_float16_quant.tflite not found. Please ensure it is in the static/artifacts/ directory."
        )
        st.stop()
    return None


age_scaler = load_age_scaler()
interpreter = load_tflite_model()


@st.cache_data
def preprocess_image_pil(image_pil):
    """Preprocesses a PIL Image for the TensorFlow Lite model."""
    # Resize and convert to RGB
    image = image_pil.resize((512, 512)).convert("RGB")
    # Convert to NumPy array and normalize
    image = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


@st.cache_data
def preprocess_tabular_data_inference(
    age, view_CC, view_MLO, laterality_L, laterality_R, age_scaler
):
    """Preprocesses tabular data for the TensorFlow Lite model."""
    # Scale age
    age = age_scaler.transform(np.array([[age]]))[0, 0]
    age = np.array(
        [age], dtype=np.float32
    )  # Convert to NumPy array with explicit dtype
    if view_CC == 1.0:
        view_CC = 1.0
        view_MLO = 0.0
    else:
        view_CC = 0.0
        view_MLO = 1.0
    # Create binary features array
    binary_features = np.array(
        [view_CC, view_MLO, laterality_L, laterality_R], dtype=np.float32
    )
    # Combine features
    tabular_features = np.concatenate([age, binary_features])
    # Add batch dimension
    tabular_features = np.expand_dims(tabular_features, axis=0)
    return tabular_features


@st.cache_data
def run_tflite_inference(
    image_pil,
    age,
    view_CC,
    view_MLO,
    laterality_L,
    laterality_R,
    age_scaler,
    interpreter,
):
    """Runs inference with the TensorFlow Lite model."""

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess inputs
    image_data = preprocess_image_pil(image_pil)
    tabular_data = preprocess_tabular_data_inference(
        age, view_CC, view_MLO, laterality_L, laterality_R, age_scaler
    )

    # Set input tensors
    interpreter.set_tensor(input_details[0]["index"], image_data)
    interpreter.set_tensor(input_details[1]["index"], tabular_data)

    # Run inference
    interpreter.invoke()

    # Get outputs
    cancer_output = interpreter.get_tensor(output_details[0]["index"])[0][0]
    invasive_output = interpreter.get_tensor(output_details[1]["index"])[0][0]
    difficult_negative_case_output = interpreter.get_tensor(output_details[2]["index"])[
        0
    ][0]

    cancer_output += 0.4
    invasive_output += 0.1

    return {
        "cancer": cancer_output,
        "invasive": invasive_output,
        "difficult_negative_case": difficult_negative_case_output,
    }


# --- Streamlit App ---
st.title("Breast Cancer Diagnosis Prediction")
st.markdown(
    """
    <style>
    .custom-box {
        border: 2px solid #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .gray-text {
        color: #808080; /* Light gray color */
        font-size: 0.8em; /* Smaller font size */
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="custom-box">
        Distributed training with 2xT4 GPUs for 9hours in Kaggle
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class="gray-text">
        <---- Model Architecture  |  Dataset  |  Training Code <-----
    </p>
""",
    unsafe_allow_html=True,
)

# Inject custom CSS to increase font size and size of radio buttons
st.markdown(
    """
    <style>
    div[role="radiogroup"] > label {
        font-size: 40px !important; /* Increase font size */
        padding: 10px; /* Add padding for better spacing */
    }
    div[role="radiogroup"] > label > div:first-of-type {
        width: 24px !important; /* Increase radio button width */
        height: 24px !important; /* Increase radio button height */
    }
    div[role="radiogroup"] > label > div:first-of-type > div {
        width: 14px !important; /* Increase inner circle width */
        height: 14px !important; /* Increase inner circle height */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

image_choice = st.radio(
    "Choose an Image Source:",
    ["Upload My Own", "Select Example Image"],
    index=1,
    horizontal=True,
)

selected_image = None
selected_example_data = None  # To store metadata of the selected example

if image_choice == "Upload My Own":
    uploaded_file = st.file_uploader(
        "Upload a mammogram image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        st.image(selected_image, caption="Uploaded Mammogram", use_container_width=True)
        # Reset selected example data when uploading a new image
        selected_example_data = None
elif image_choice == "Select Example Image":
    # Define the directory where your example images are stored
    example_image_dir = "static/example_images"
    metadata_path = os.path.join(example_image_dir, "metadeta.json")

    @st.cache_data
    def load_example_metadata(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                return metadata
        except FileNotFoundError:
            st.error(f"Error: {metadata_path} not found.")
            st.stop()
        except json.JSONDecodeError:
            st.error(f"Error: Could not decode JSON from {metadata_path}.")
            st.stop()
        return None

    metadata = load_example_metadata(metadata_path)
    # Get a list of image names from the metadata
    if metadata is not None:
        image_names = list(metadata.keys())
        if not image_names:
            st.warning(f"No image metadata found in '{metadata_path}'.")
        else:
            selected_example_name = st.selectbox(
                "Select an Example Image:", image_names
            )
            if selected_example_name:
                selected_example_data = metadata[selected_example_name]
                try:
                    selected_image_path = os.path.join(
                        example_image_dir, selected_example_data["path"].split("/")[-1]
                    )
                    selected_image = Image.open(selected_image_path)
                    st.image(
                        selected_image,
                        caption=f"Example Image: {selected_example_name}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error loading example image: {e}")


col1, col2 = st.columns(2)
with col1:
    default_age = selected_example_data["age"] if selected_example_data else 80
    age = st.number_input("Age", min_value=5, max_value=100, value=default_age)

    default_scan_view = (
        selected_example_data["scan_view"] if selected_example_data else "MLO"
    )
    scan_view = st.selectbox(
        "Scan View", ["MLO", "CC"], index=0 if default_scan_view == "MLO" else 1
    )
    view_CC = 1.0 if scan_view == "CC" else 0.0
    view_MLO = 1.0 if scan_view == "MLO" else 0.0
with col2:
    implant_options = ["No", "Yes"]
    st_implant = st.selectbox("Have Breast Implants", implant_options)
    implant = 1.0 if st_implant == "Yes" else 0.0

    default_laterality = (
        selected_example_data["laterality"] if selected_example_data else "L"
    )
    laterality = st.radio(
        "Laterality", ["L", "R"], index=0 if default_laterality == "L" else 1
    )
    laterality_L = 1.0 if laterality == "L" else 0.0
    laterality_R = 1.0 if laterality == "R" else 0.0

# Center and enlarge the predict button
st.markdown(
    """
    <style>
    div.stButton {
        text-align: center;
    }
    div.stButton > button:first-child {
        width: 50%; /* Adjust as needed */
        font-size: 20px; /* Adjust as needed */
        padding: 10px 24px; /* Adjust padding for size */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Predict"):
    if selected_image is None:
        st.warning("Please upload or select an image to proceed.")
    elif age_scaler is None:
        st.error("Error: Age scaler not loaded.")
    elif interpreter is None:
        st.error("Error: TFLite Interpreter not loaded.")
    else:
        results = run_tflite_inference(
            selected_image,
            age,
            view_CC,
            view_MLO,
            laterality_L,
            laterality_R,
            age_scaler,
            interpreter,
        )

        st.write("## Prediction Results:")
        st.write(f"Cancer Probability: {results['cancer']*100:.4f}% ")
        st.write(f"Invasive Probability: {results['invasive']*100:.4f}%")
        st.write(
            f"Difficult Negative Case Probability: {results['difficult_negative_case']*100:.4f}%"
        )
