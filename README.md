# 🧠 Medical QA Text Classifier with distilbert

This project demonstrates how to fine-tune a pre-trained distilbert model to classify medical and general knowledge yes/no/maybe questions. It includes preprocessing, model training, evaluation, and prediction phases, all implemented in a single Jupyter Notebook.

---

## 📁 Project Structure

- `code.ipynb`: Main notebook containing code for training and evaluating the model.
- `data/`: Contains CSV or text files used for training (if applicable).
- `README.md`: Project documentation.

---

## 🚀 Model Overview

- **Base model**: `distilbert-base`
- **Task**: Text classification (3 classes: Yes / No / Maybe)
- **Library**: HuggingFace Transformers
- **Tokenizer**: `distilbertTokenizer`

---

## 📊 Dataset

The dataset consists of questions related to medical and general knowledge topics, each labeled with one of the following:
- `"yes"`
- `"no"`
- `"maybe"` (relatively underrepresented)

The data was split into training and testing sets using `train_test_split`.

---

## ✅ Evaluation Metrics

The model is evaluated using:
- Accuracy
- Classification Report (Precision, Recall, F1-score)

### Sample results:

```text
Accuracy: 57%
Model performs well on 'yes', poorly on 'no', and completely misses 'maybe' due to class imbalance.
