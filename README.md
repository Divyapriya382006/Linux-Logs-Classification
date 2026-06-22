
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

## 📊 Evaluation & Metrics

### 1. Classification Report

The neural network model achieves an overall accuracy of **94%** across the test split. Below is the detailed precision, recall, and F1-score breakdown for select classes displaying sample distribution imbalances:

```text
             precision    recall  f1-score   support

           7       1.00      1.00      1.00        11
           9       1.00      1.00      1.00         2
          10       1.00      1.00      1.00         4
          12       0.50      1.00      0.67         1
          41       1.00      1.00      1.00         6
          80       0.96      1.00      0.98        23
          81       1.00      1.00      1.00         4
          82       1.00      1.00      1.00        63
          84       1.00      1.00      1.00        20
          85       1.00      1.00      1.00       187
          86       1.00      1.00      1.00         2
         100       0.83      1.00      0.91         5
         110       1.00      1.00      1.00        24
         111       1.00      1.00      1.00        20

    accuracy                           0.94       400
   macro avg       0.33      0.34      0.33       400
weighted avg       0.94      0.94      0.94       400

```

> **Note on Performance Imbalance**: While the model performs flawlessly on high-frequency root logs (such as classes `82` and `85`), the lower `macro avg` performance reflects extreme dataset long-tail distributions where certain rare system occurrences appear only once in the raw text corpus.

### 2. Confusion Matrix

A standard confusion matrix mapping is plotted via Seaborn heatmaps to evaluate spatial multi-class predictions:

```python
cm = confusion_matrix(y_test, pred_classes)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Log Template Classification Confusion Matrix')
plt.show()

```

Due to the presence of over 100+ unique categorical event slots, standard graphical heatmaps condense heavily. Visual evaluation confirms a dominant true-positive diagonal line alongside negligible spread across off-diagonal coordinate pairs.

---

## 🔍 Error Analysis & Diagnostics

To explicitly break down misclassifications without relying on crowded visual matrices, the code isolates instances where predictions conflict with actual labels.

The tracking script captures the exact semantic mismatches where the model struggled to differentiate overlapping text structures:

```python
errors = (y_test != pred_classes)
output = pd.DataFrame({
    'actual': comp_label.inverse_transform(y_test[errors]),
    'predicted': comp_label.inverse_transform(pred_classes[errors])
})

```

### Mismatch Profiling Log

| Index | Actual True Template Class | Predicted Misclassified Target |
| --- | --- | --- |
| **0** | `audit: initializing netlink socket (disabled)` | `audit(<*>.<*>:<*>): initialized` |
| **1** | `DMA zone: <*> pages, LIFO batch:<*>` | `ACPI: ACPI tables contain no PCI IRQ routing e...` |
| **2** | `ACPI disabled because your bios is from <*> an...` | `CPU: Intel Pentium III (Coppermine) stepping <*>` |
| **3** | `sdpd startup succeeded` | `SELinux: Starting in permissive mode` |
| **4** | `mapped 4G/4G trampoline to <*>.` | `ACPI: Subsystem revision <*>` |
| **5** | `Transparent bridge - <*>` | `usbcore: registered new driver usbfs` |
| **6** | `User unknown timed out after <*> seconds at <*...` | `Calibrating delay loop... <*>.<*> BogoMIPS` |
| **7** | `Enabling fast FPU save and restore... done.` | `Enabling unmasked SIMD FPU exception support.....` |
| **8** | `Couldn't authenticate user` | `notify question section contains no SOA` |
| **9** | `rpc.idmapd startup succeeded` | `restart.` |
| **10** | `Bringing up loopback interface: succeeded` | `Checking 'hlt' instruction... OK.` |
| **11** | `portmap startup succeeded` | `rpc.statd startup succeeded` |
| **12** | `Using tsc for high-res timesource` | `audit(<*>.<*>:<*>): initialized` |
| **13** | `Version <*>.<*>.<*> Starting` | `authentication failure; logname= uid=0 euid=0 ...` |
| **14** | `BIOS-provided physical RAM map:BIOS-e820: <*> - <*> (usable)` | `PCI: Using configuration type <*>` |
| **15** | `PCI: Using IRQ router PIIX/ICH [<*>/<*>] at <*>` | `PCI: PCI BIOS revision <*>.<*> entry at <*>, l...` |
| **16** | `PCI: Probing PCI hardware` | `Initializing random number generator: succeeded` |
| **17** | `kernel.core_uses_pid = <*>` | `Kernel command line: ro root=LABEL=/ rhgb quiet` |
| **18** | `ACPI: Subsystem revision <*>` | `zapping low mappings.` |
| **19** | `audit(<*>.<*>:<*>): initialized` | `irqbalance startup succeeded` |
| **20** | `klogd <*>.<*>.<*>, log source = /proc/kmsg sta...` | `Initializing CPU#<*>` |
| **21** | `ACPI: ACPI tables contain no PCI IRQ routing e...` | `audit: initializing netlink socket (disabled)` |

```

```
