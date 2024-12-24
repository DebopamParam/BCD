import streamlit as st
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Architecture", page_icon="🤖")

st.title("Model Architecture")

try:
    model_path = "static/additional/model.html"
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            html_content = f.read()

        # Calculate responsive height based on viewport
        components.html(html_content, height=600, scrolling=True)
    else:
        st.error(
            "Model file not found. Please ensure 'static/additional/model.html' exists in the application directory."
        )

except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Load and display model
markdown_text = """
# **Model Architecture Summary: A Hybrid Approach for Breast Cancer Detection**

The model employs a hybrid architecture that effectively combines the strengths of convolutional neural networks (CNNs) for image analysis and fully connected networks for processing tabular data. It's designed as a **multi-input, multi-output** model, meaning it takes two types of input (mammogram images and clinical data) and produces three distinct outputs (cancer presence, invasiveness, and difficult negative case probability).

Here's a breakdown of the key components:

### **1. Image Feature Extraction with EfficientNetV2:**

*   **Input:** Mammogram images, preprocessed to 512x512 pixels and converted to RGB format.
*   **Feature Extractor:**  A pre-trained **EfficientNetV2B3** model serves as the backbone for image feature extraction.
    *   **Pre-trained Weights:** The model is initialized with weights learned from the ImageNet dataset, providing a strong starting point for image understanding.
    *   **Top Layer Removed:** The original classification layer of EfficientNetV2 is removed (`include_top=False`) as we are building custom output layers.
    *   **Global Average Pooling:** Global average pooling (`pooling='avg'`) is applied after the convolutional layers to reduce the spatial dimensions of the feature maps and produce a fixed-size feature vector.
    *   **Fine-tuning:** To adapt the model to the specific task of breast cancer detection, layers from `block6` onwards are made trainable, allowing for fine-tuning on the target dataset.

### **2. Tabular Data Processing:**

*   **Input:** A 5-element vector representing patient age (standardized) and one-hot encoded features for `view` (CC/MLO) and `laterality` (Left/Right).
*   **Processing Layers:** The tabular data is processed through a series of fully connected (`Dense`) layers:
    *   Two `Dense` layers with 64 and 32 units, respectively, both using ReLU activation.
    *   `Dropout` layers (with rates of 0.3 and 0.2) are interspersed between the `Dense` layers to prevent overfitting.

### **3. Feature Fusion:**

*   **Concatenation:** The extracted image features (from EfficientNetV2) and the processed tabular features are concatenated using `tf.keras.layers.Concatenate`. This combines the information from both modalities into a single, rich feature vector.
*   **Further Processing:** This combined feature vector is then fed through an additional `Dense` layer (512 units, ReLU activation) and a `Dropout` layer (rate 0.4) for further processing and regularization.

### **4. Multi-Output Branches:**

*   **Three Separate Branches:** The model then branches out into three separate paths, one for each prediction target:
    *   **Cancer Output:** Predicts the probability of cancer presence (0 or 1).
    *   **Invasive Output:** Predicts the probability of the cancer being invasive (0 or 1).
    *   **Difficult Negative Case Output:** Predicts the probability of the case being a "difficult negative" (0 or 1).
*   **Branch Architecture:** Each branch consists of a sequence of fully connected layers with decreasing numbers of units and ReLU activation, followed by `Dropout` for regularization.
*   **Final Output Layer:** Each branch culminates in a `Dense` layer with a single unit and a sigmoid activation function. This produces a probability score between 0 and 1 for each of the three prediction targets.

**In essence, the model architecture can be visualized as:**


**Key Design Choices:**

*   **Pre-trained CNN:** Leveraging a pre-trained EfficientNetV2 provides a powerful feature extractor that has already learned to recognize general image features.
*   **Hybrid Input:** Combining image and tabular data allows the model to consider both visual and clinical information.
*   **Multi-Output:** Predicting multiple aspects of the case provides a more comprehensive assessment than a single cancer/no-cancer prediction.
*   **Regularization:** Dropout layers are strategically used to prevent overfitting and improve generalization.
*   **Fine-tuning:** Unfreezing some layers of EfficientNetV2 allows the model to adapt to the specific characteristics of the mammogram dataset.

This architecture represents a well-structured approach to the complex task of breast cancer detection, integrating different data modalities and utilizing the strengths of deep learning to make multiple, clinically relevant predictions.

| Layer (type)        | Output Shape      | Param #    | Connected to      |
|---------------------|-------------------|------------|-------------------|
| tabular_input       | (None, 5)         |          0 | -                 |
| (InputLayer)        |                   |            |                   |
| dense (Dense)       | (None, 64)        |        384 | tabular_input[0]  |
| dropout (Dropout)   | (None, 64)        |          0 | dense[0][0]       |
| image_input         | (None, 512, 512, 3) |        0 | -                 |
| (InputLayer)        |                   |            |                   |
| dense_1 (Dense)     | (None, 32)        |      2,080 | dropout[0][0]     |
| efficientnetv2-b3   | (None, 1536)      | 12,930,622 | image_input[0][0] |
| (Functional)        |                   |            |                   |
| dropout_1 (Dropout) | (None, 32)        |          0 | dense_1[0][0]     |
| concatenate         | (None, 1568)      |          0 | efficientnetv2-b3 |
| (Concatenate)       |                   |            | dropout_1[0][0]   |
| dense_2 (Dense)     | (None, 512)       |    803,328 | concatenate[0][0] |
| dropout_2 (Dropout) | (None, 512)       |          0 | dense_2[0][0]     |
| dense_3 (Dense)     | (None, 512)       |    262,656 | dropout_2[0][0]   |
| dropout_3 (Dropout) | (None, 512)       |          0 | dense_3[0][0]     |
| dense_4 (Dense)     | (None, 256)       |    131,328 | dropout_3[0][0]   |
| dense_7 (Dense)     | (None, 256)       |    131,328 | dropout_2[0][0]   |
| dense_10 (Dense)    | (None, 256)       |    131,328 | dropout_2[0][0]   |
| dropout_4 (Dropout) | (None, 256)       |          0 | dense_4[0][0]     |
| dropout_6 (Dropout) | (None, 256)       |          0 | dense_7[0][0]     |
| dropout_8 (Dropout) | (None, 256)       |          0 | dense_10[0][0]    |
| dense_5 (Dense)     | (None, 128)       |     32,896 | dropout_4[0][0]   |
| dense_8 (Dense)     | (None, 128)       |     32,896 | dropout_6[0][0]   |
| dense_11 (Dense)    | (None, 128)       |     32,896 | dropout_8[0][0]   |
| dropout_5 (Dropout) | (None, 128)       |          0 | dense_5[0][0]     |
| dropout_7 (Dropout) | (None, 128)       |          0 | dense_8[0][0]     |
| dropout_9 (Dropout) | (None, 128)       |          0 | dense_11[0][0]    |
| dense_6 (Dense)     | (None, 64)        |      8,256 | dropout_5[0][0]   |
| dense_9 (Dense)     | (None, 64)        |      8,256 | dropout_7[0][0]   |
| dense_12 (Dense)    | (None, 64)        |      8,256 | dropout_9[0][0]   |
| cancer_output       | (None, 1)         |         65 | dense_6[0][0]     |
| (Dense)             |                   |            |                   |
| invasive_output     | (None, 1)         |         65 | dense_9[0][0]     |
| (Dense)             |                   |            |                   |
| difficult_negative  | (None, 1)         |         65 | dense_12[0][0]    |
| (Dense)             |                   |            |                   |

"""

st.markdown(markdown_text)
