<<<<<<< HEAD
# 🧠 NDT Flaw Detection – Streamlit Frontend

A complete Machine Learning workflow for **Non-Destructive Testing (NDT)** flaw detection using the **Koomas dataset**.  
This project combines data extraction, training, prediction, and an interactive **Streamlit dashboard** for visualization, analysis, and model evaluation.

---

## 🔍 Short Description

This app predicts whether an ultrasonic or X-ray image strip contains a **flaw** or **no flaw**, trained on the [Koomas/NDT_ML_Flaw](https://github.com/koomas/NDT_ML_Flaw) dataset.  
It provides:
- A **Streamlit UI** to visualize and predict flaws
- **Threshold tuning** with precision, recall, F1, and AUC
- **Dataset statistics** (class balance, shard sizes)
- **Training log viewer** for metrics and progress tracking

---

## 📁 Project Structure

NDT_ML_Flaw/
│
├── data_extraction.py # Extracts image strips & labels into .npy shards
├── train_model.py # Trains the CNN model, saves to models/
├── predict_any.py # CLI prediction on single or batch data
├── app.py # Base Flask/CLI version (legacy)
├── streamlit_app.py # 🔹 Streamlit frontend for visual prediction & stats
├── test_utilities.py # Test utilities and helpers
├── requirements.txt # Default environment requirements
├── requirements_streamlit.txt # Streamlit + TensorFlow environment
├── extracted_data/ # Folder containing manifest.json and shards
│ ├── manifest.json
│ ├── batch_XXX_images.f16.npy
│ └── batch_XXX_labels.u1.npy
├── models/ # Trained model folder
│ ├── best.keras
│ ├── final.keras
│ ├── threshold.txt


---

## ⚙️ Installation

1. **Clone this project**
   ```bash
   git clone https://github.com/koomas/NDT_ML_Flaw.git
   cd NDT_ML_Flaw

* Create and activate virtual environment *

py -m venv venv
.\venv\Scripts\activate
│ ├── training_log.csv
│ └── metrics.txt
└── uploads/ # Temporary uploads from Streamlit UI

Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements_streamlit.txt

🚀 Running the Streamlit App

Start the Streamlit frontend:

streamlit run streamlit_app.py

🖥️ App Pages Overview
🧩 Overview

Displays environment setup summary.

Verifies TensorFlow, dataset, and model availability.

⚡ Predict

Upload Image: Predict flaw/no-flaw for PNG/JPG.

Pick from Shard (.npy): Choose a global row from the extracted dataset for inference.

🎚️ Threshold Tuning

Explore precision, recall, F1, AUC vs threshold.

Shows confusion matrix and ROC curve for validation split.

📊 Validate (Stats)

Summarizes dataset class distribution.

Plots shard-wise row counts and flaw/no-flaw ratios.

🔍 Explore Dataset

Browse sample rows directly from .npy shards.

View image strips and ground truth labels.

🧾 Training Logs

Displays training_log.csv and metrics.txt saved by train_model.py.

*📈 Training Workflow*

Data Extraction

python data_extraction.py


Creates extracted_data/manifest.json and .npy shards.

*Model Training*

python train_model.py


Saves model to models/best.keras or models/final.keras
and threshold info to models/threshold.txt.

Command-line Prediction (optional)

python predict_any.py --input sample.png


*Streamlit Visualization*

streamlit run streamlit_app.py
=======
# NDT_ML_Flaw
This is a data set for teaching a machine learning learning model with different types of flaws. The data is zipped in an .xz format, which can be opened for example with Python LZMA package. The data size is 480x7168 and the flaw area 1100-3100. Each file contains 1000 images with roughly 50% flaws and 50% no flaws. 2xx batches are simulated flaws.

The .xz files contain the actual images and .txt files metadata regarding the image.
Metadata example
Flaw (1 flaw, 0 no flaw), Amount of augmentation (between 0.4-1), Flaw depth  Flaw location (scan axis)  Original flaw size, index line (always one), Flaw type
1	                        0.472899	                              1681.461469	210.526275	               2.0	              1	                         P41_01

>>>>>>> 1111d43 (Initial commit)
