# **Building a Multi-Output Model for Breast Cancer Detection using EfficientNetV2 and Clinical Data**

## **Abstract:**

Breast cancer is a leading cause of cancer-related deaths among women worldwide. Early and accurate detection is crucial for improving patient outcomes. This article describes the development and evaluation of a deep learning model for breast cancer detection using mammogram images and tabular clinical data. We leverage the power of EfficientNetV2 for image feature extraction and combine it with patient-specific information to create a multi-output model that predicts cancer presence, invasiveness, and whether a case is a "difficult negative."

***[Competition and Dataset Link](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)***

### **1. Introduction**

The RSNA Breast Cancer Detection challenge on Kaggle provides a valuable dataset for training and evaluating machine-learning models for breast cancer diagnosis. The dataset comprises mammogram images in DICOM format and associated clinical data for each patient. The challenge aims to identify cancerous lesions accurately and determine key characteristics like invasiveness.

This project addresses the challenge by building a multi-output deep learning model. The model takes both mammogram images and tabular data as input and predicts three crucial aspects:

*   **Cancer:** Whether cancer is present in the mammogram.
*   **Invasive:** Whether the detected cancer is invasive.
*   **Difficult Negative Case:** Whether the case is challenging to diagnose correctly even if cancer is absent (a flag from the original dataset).

By integrating image and tabular data, we aim to create a more robust and informative model than those relying on image data alone.

### **2. Dataset Preparation and Preprocessing**

**2.1 Initial Data Exploration and Cleaning**

The initial steps involve loading the training dataset (`train.csv`) and exploring its structure and characteristics. Key observations include:

*   **Data types:** The dataset contains numerical (e.g., patient age), categorical (e.g., laterality, view), and boolean (e.g., implant) features.
*   **Missing values:** The `age` feature has some missing values, which are imputed using the mean age.
*   **Data distribution:** Certain features like `view` are imbalanced. For instance, `CC` (craniocaudal) and `MLO` (mediolateral oblique) views are the most common.
*   **Target variables:** The `cancer`, `invasive`, and `difficult_negative_case` columns represent the prediction targets.

**2.2 Data Cleaning and Feature Engineering**

*   **Dropping irrelevant columns:** Columns like `site_id`, `machine_id`, `biopsy`, `BIRADS`, and `density` are dropped as they are either not relevant for the current modeling approach or have many missing values.
*   **Data type conversion:** Boolean columns are converted to numerical (0 and 1) for easier processing.
*   **Handling categorical features:** The `view` and `laterality` columns are one-hot encoded using `pd.get_dummies`.
*   **Filtering views:** The dataset is filtered to include only the most frequent views, `MLO` and `CC`, simplifying the image analysis.
*   **Image path construction:** A new `image_path` column is added, combining the base path with patient and image IDs to create the full path for each DICOM image.

**2.3 Data Stratification and Splitting**

*   **Stratified sampling:** To ensure that the training, validation, and test sets have similar distributions of target variables, stratified sampling is employed. A combined `stratify_label` is created using `cancer`, `invasive`, and `implant` columns.
*   **Reducing data imbalance:** The predominant `0_0_0` (no cancer, non-invasive, no implant) group is downsampled to prevent the model from being biased towards the majority class.
*   **Train-validation-test split:** The dataset is split into training (70%), validation (15%), and test (15%) sets, preserving the stratified distribution.

**2.4 Age Scaling**

*   **Standardization:** The `age` feature is standardized using `StandardScaler` from scikit-learn. This ensures that age does not disproportionately influence the model due to its scale.
*   **Scaler artifact saving:** The trained scaler is saved as `age_scaler_artifact.pkl` for later use during inference or model deployment.

### **3. Image Preprocessing**

**3.1 DICOM Image Handling**

*   **Reading DICOM files:** The `pydicom` library is used to read DICOM images, extracting the pixel array for further processing.
*   **Normalization:** Pixel values are normalized to the range [0, 1] by subtracting the minimum and dividing by the range (max - min).
*   **Resizing:** Images are resized to 512x512 pixels, which is a suitable input size for EfficientNetV2.
*   **Grayscale to RGB:** Since EfficientNetV2 expects RGB images, single-channel grayscale images are converted to three-channel images by replicating the grayscale channel.

**3.2 TensorFlow Preprocessing Functions**

*   **`preprocess_image`:** This function reads the DICOM file, normalizes pixel values, resizes the image, and converts it to RGB. It uses `tf.py_function` to wrap the Python code (which includes `pydicom` operations) within a TensorFlow graph.
*   **`preprocess_tabular_data`:** This function extracts and converts relevant tabular features (`age`, `view_CC`, `view_MLO`, `laterality_L`, `laterality_R`) into a single tensor.
*   **`preprocess_label`:** This function extracts the target variables (`cancer`, `invasive`, `difficult_negative_case`) and converts them into a dictionary of tensors.
*   **`preprocess_row`:** This function combines the image and tabular preprocessing steps, returning a dictionary of inputs and a dictionary of outputs.

### **4. Model Architecture**

**4.1 Multi-Input, Multi-Output Design**

The model is designed to handle both image and tabular data and produce three separate outputs. This is achieved using the Keras functional API:

*   **Image Input:**
    *   A `tf.keras.layers.Input` layer defines the input for images (shape: 512x512x3).
    *   The pre-trained `EfficientNetV2B3` model (with weights initialized from ImageNet) is used as the base image feature extractor.
    *   The top (classification) layer of EfficientNetV2 is removed (`include_top=False`), and global average pooling is applied (`pooling='avg'`).
    *   Layers from `block6` onwards in EfficientNetV2 are made trainable to fine-tune the model for this specific task.
*   **Tabular Input:**
    *   A separate `tf.keras.layers.Input` layer defines the input for tabular data (shape: 5).
    *   A series of `Dense` layers with ReLU activation and `Dropout` layers process the tabular data.
*   **Concatenation:** The extracted image features and processed tabular features are concatenated using `tf.keras.layers.Concatenate`.
*   **Output Branches:**
    *   Three separate branches of fully connected (`Dense`) layers with `Dropout` are created, one for each output: `cancer`, `invasive`, and `difficult_negative_case`.
    *   Each branch ends with a `Dense` layer with a single unit and sigmoid activation, producing a probability between 0 and 1.

**4.2 Compilation**

The model is compiled with:

*   **Optimizer:** Adam (a common choice for deep learning models).
*   **Loss:** Binary cross-entropy for each output. This is appropriate because each output is a binary classification problem.
*   **Loss Weights:** Weights are assigned to the losses for each output. `difficult_negative_case` is given a lower weight (0.5) compared to `cancer` and `invasive` (1.0 each). This might be done to balance the importance of the different prediction tasks.
*   **Metrics:** Accuracy is used as the evaluation metric for each output.

### **5. Distributed Training with MirroredStrategy**

To accelerate training, especially on multiple GPUs, `tf.distribute.MirroredStrategy` is used. This strategy replicates the model on each available GPU and distributes the data across them. The main steps include:

1. **Detecting GPUs:** `tf.config.list_physical_devices('GPU')` detects available GPUs.
2. **Creating MirroredStrategy:** An instance of `MirroredStrategy` is created, distributing computations across the detected GPUs.
3. **Creating Datasets within Strategy Scope:** The `train_dataset`, `val_dataset`, and `test_dataset` are created using `tf.data.Dataset` and then distributed using `strategy.experimental_distribute_dataset`.
4. **Model Definition within Strategy Scope:** The entire model definition, including input layers, base model, tabular processing, concatenation, output branches, and compilation, is placed within the `with strategy.scope():` block. This ensures that the model variables are created and mirrored on each GPU.

### **6. Training and Evaluation**

**6.1 Training**

*   **`model.fit`:** The model is trained using `model.fit`, passing the distributed training and validation datasets.
*   **Epochs:** The model is trained for 10 epochs.
*   **Callbacks:** `ModelCheckpoint` is used to save the best model (based on validation loss) during training.
*   **Batch Size:**  A global batch size of 100 is used, which is a per-replica batch size of 50 on a system with two GPUs.

**6.2 Evaluation**

*   **Loading the Best Model:** The best saved model is loaded using `tf.keras.models.load_model`.
*   **Prediction on Test Set:** The `predict_step` function performs predictions on a batch of data on a single replica, and `get_distributed_predictions` aggregates predictions across all replicas and steps.
*   **Metrics Calculation:**
    *   **Confusion Matrix:** `sklearn.metrics.confusion_matrix` is used to compute the confusion matrix for each output.
    *   **F1-score:** `sklearn.metrics.f1_score` is used to calculate the F1-score for each output.

### **7. Results and Discussion**

The confusion matrices and F1-scores provide insights into the model's performance:

*   **Confusion Matrices:** The confusion matrices visualize the model's predictions against the true labels. They show the counts of true positives, true negatives, false positives, and false negatives for each output.
*   **F1-scores:** The F1-scores provide a balanced measure of precision and recall. They are particularly useful when dealing with imbalanced datasets.

**Interpreting the Results:**

*   **High True Positives/Negatives:**  A high number of true positives and true negatives indicates that the model is correctly identifying both positive and negative cases for a given output.
*   **False Positives/Negatives:** False positives (predicting a positive case when it's negative) and false negatives (predicting a negative case when it's positive) represent errors. The relative importance of these errors depends on the specific medical context. In cancer detection, false negatives are generally more concerning as they could lead to delayed treatment.
*   **F1-score close to 1:** An F1-score closer to 1 indicates better overall performance, considering both precision and recall.

**Potential Improvements:**

*   **Hyperparameter Tuning:** Experiment with different learning rates, dropout rates, number of layers, and units per layer to potentially improve performance.
*   **Data Augmentation:** Apply image augmentation techniques (e.g., rotations, flips, zooms) to increase the effective size of the training dataset and make the model more robust to variations in image appearance.
*   **Class Weighting:** Address class imbalance by assigning higher weights to the minority classes during training.
*   **Ensemble Methods:** Combine predictions from multiple models to potentially improve accuracy and robustness.
*   **Further Feature Engineering:** Explore other potentially relevant features from the DICOM metadata or other clinical information.

**8. Conclusion**

This project demonstrates the feasibility of building a multi-output deep learning model for breast cancer detection using both mammogram images and clinical data. The combination of EfficientNetV2 for image feature extraction and tabular data processing allows the model to learn complex patterns and make predictions about cancer presence, invasiveness, and the difficulty of diagnosis.

The use of distributed training with `MirroredStrategy` enables efficient training on multiple GPUs. The evaluation using confusion matrices and F1-scores provides a comprehensive assessment of the model's performance on each output.

While the model shows promising results, further research and development are necessary to achieve clinically acceptable levels of accuracy and reliability. The potential improvements mentioned above can be explored to enhance the model's performance and make it a more valuable tool for assisting radiologists in breast cancer diagnosis.
