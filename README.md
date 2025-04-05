# SQL Injection Classifier using DistilBERT

This project uses a fine-tuned `distilbert-base-uncased` model to detect potential SQL injection attacks in input text.

---

## Model

We use a pretrained **DistilBERT** model from Hugging Face and fine-tune it on a labeled dataset of:

- ✅ Safe SQL inputs
- 🚨 SQL Injection attempts

---

## Project Structure

```
project_folder/
│
├── train_sql_injection_classifier.py   # Script to train and save the model
├── predict_sql_injection.py            # Script to load model and classify new input
├── sql_injection_dataset.csv           # Dataset of safe & malicious SQL statements
├── sql_injection_model/                # (Ignored by Git) Folder containing saved model files
├── results/                            # (Ignored by Git) Trainer logs & checkpoints
└── README.md
```

---

## Setup

```bash
# Create virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install required packages
pip install -r requirements.txt
```

You can also manually install:

```bash
pip install transformers datasets scikit-learn pandas accelerate torch
```

---

## Train the Model

```bash
python train_sql_injection_classifier.py
```

This script:

- Loads the dataset
- Fine-tunes DistilBERT
- Saves the trained model to `./sql_injection_model/`

---

## Make Predictions

```bash
python predict_sql_injection.py
```

You'll be prompted to enter input text, and the model will return:

- ✅ `"Safe Input"` or
- 🚨 `"SQL Injection Detected"`

---

## 🔬 Example

```
Enter input text to classify: SELECT * FROM users WHERE username = 'admin' --
Prediction: SQL Injection Detected
```

---


