
````markdown
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
````

---

## 💬 Sample Questions and Predictions

| Question                     | Prediction |
| ---------------------------- | ---------- |
| Is smoking harmful?          | no         |
| Can exercise improve health? | no         |
| Is climate change real?      | no         |
| Can stress affect health?    | no         |
| Can we prevent all diseases? | no         |

> Note: Due to class imbalance and insufficient training, model initially predicts "no" for all inputs. Improvements are needed (see below).

---

## 🔧 How to Improve

* ✅ Balance the dataset using oversampling or class weights
* ✅ Use more training data
* ✅ Experiment with hyperparameters (`learning_rate`, `epochs`, etc.)
* ✅ Use early stopping or validation loss monitoring
* ✅ Try other models

---

## 🛠️ Installation & Requirements

```bash
pip install transformers datasets scikit-learn nltk rouge-score
```

---

## 🧪 Running the Notebook

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/Medical-QA-Classifier.git
   cd Medical-QA-Classifier
   ```

2. Launch the notebook:

   ```bash
   jupyter notebook finetuned_medical_model.ipynb
   ```

3. Follow along cell-by-cell to preprocess, fine-tune, and evaluate the model.

---

## 🙋 Author

* 🔗 GitHub: [@Khojoii](https://github.com/Khojoii)
* 📧 Contact: [m.khojoii@gmail.com](mailto:m.khojoii@gmail.com)






