```markdown

# Linux Log Template Classification using Deep Learning

This repository contains an end-to-end Machine Learning and Deep Learning pipeline designed to classify raw Linux log messages into standard, structured system event templates (`EventId`). The project combines text vectorization, structural feature handling, and a multi-layer Sequential Neural Network built using TensorFlow and Keras.

## 🚀 Features
* **Hybrid Preprocessing Pipeline**: Integrates structural data (`Component`, `PID`) with TF-IDF Vectorization on raw unstructured log content (`Content`).
* **Robust Target Encoding**: Dynamic integer mapping of log categories using `LabelEncoder`.
* **Deep Learning Classifier**: Multi-layer Dense Neural Network optimized for multi-class classification using Softmax activation.
* **Error Diagnosis Utilities**: Custom evaluation routines to pinpoint exact semantic log messages that confuse the network.

---

## 📂 Project Structure

```text
├── linux-logs.ipynb     # Main Jupyter notebook containing data pipeline and model training
├── README.md            # Project documentation
└── data/                # Directory containing raw Linux log benchmarks

```

---

## 🛠️ Installation & Setup

1. Clone this repository to your local machine or upload the notebook to Kaggle/Google Colab.
2. Install the required dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow seaborn matplotlib plotly

```



---

## ⚙️ Model Pipeline Workflow

### 1. Feature Preprocessing

Log messages contain structural components alongside free text. Features are systematically prepared using a `ColumnTransformer`:

* **`Content`**: Converted into numerical vectors using a `TfidfVectorizer`.
* **`Component` & `PID**`: Passed through natively as supplementary numerical/structural information.

### 2. Network Architecture

The classification engine is built using Keras' `Sequential` API:

* **Input Layer**: Structured dynamically to match the absolute size of the generated TF-IDF vocabulary matrix shape.
* **Hidden Layers**: Fully connected `Dense` layers (128 ➔ 64 ➔ 32 nodes) utilizing `ReLU` activations.
* **Output Layer**: A dense layer scaled to `num_classes` (dynamically inferred using `np.max(y) + 1`) paired with **Softmax** activation to output precise event category probabilities.

```python
model = Sequential([
    keras.layers.Input(shape=(input_dimensions,)), 
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])

```

### 3. Compilation Configuration

* **Optimizer**: `Adam` (Adaptive Moment Estimation)
* **Loss Function**: `sparse_categorical_crossentropy` (ideal for multi-class targets encoded as integers)

---

## 📊 Evaluation & Diagnostics

The pipeline prints a full classification performance breakdown. While overall accuracy reaches **94%**, the project notes structural data imbalances where rare system event logs are heavily out-represented by routine system entries.

### Error Analysis Script

To debug performance bottlenecks without cluttering the notebook with dense, unreadable confusion matrix graphs, the project relies on an explicit text-unrolling lookup to reveal direct semantic mismatches:

```python
errors = (y_test != pred_classes)
output = pd.DataFrame({
    'actual_template': comp_label.inverse_transform(y_test[errors]),
    'predicted_template': comp_label.inverse_transform(pred_classes[errors])
})
print(output)

```

```

```
