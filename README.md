# 💳 Credit Card Fraud Detection Using Machine Learning

An end-to-end machine learning project that detects potentially fraudulent credit card transactions using classification and anomaly detection techniques. It includes an interactive Streamlit dashboard for transaction analysis, fraud prediction, and model evaluation.

## 📌 Project Overview

Credit card fraud is a major challenge in digital payments. Since fraudulent transactions represent a very small percentage of all transactions, detecting them accurately is a highly imbalanced classification problem.

This project explores multiple machine learning algorithms and develops a Random Forest model to identify potentially fraudulent transactions.

## 🎯 Project Objectives

- Explore and understand credit card transaction data.
- Identify patterns and characteristics associated with fraudulent transactions.
- Handle severe class imbalance.
- Train and compare multiple machine learning models.
- Evaluate performance using precision, recall, F1-score, and Average Precision.
- Build an interactive dashboard for fraud detection.

## 📊 Dataset

**Dataset:** European Credit Card Fraud Detection

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Original transactions: 284,807
- Input features: 30
- Target variable: `Class`
- `0` = Legitimate transaction
- `1` = Fraudulent transaction

The dataset contains anonymized PCA-transformed features (`V1`–`V28`), along with transaction time and amount.

The dataset contains historical transactions from September 2013.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 🤖 Machine Learning Models

Three algorithms were explored and compared:

1. Logistic Regression
2. Random Forest Classifier
3. Isolation Forest

Random Forest achieved the strongest overall performance in the experiments and was selected for the final dashboard.

## 📈 Final Model Performance

The final Random Forest model was evaluated using a chronological train-validation-test split to simulate predicting future transactions.

| Metric | Result |
|---|---:|
| Accuracy | 99.96% |
| Precision | 91.80% |
| Recall | 75.68% |
| F1-Score | 82.96% |
| Average Precision | 79.78% |

**Decision threshold:** 0.20 (20%)

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,667 | 5 |
| Actual Fraud | 18 | 56 |

The model detected 56 out of 74 fraudulent transactions in the chronological test set, while incorrectly flagging 5 legitimate transactions.

## 🖥️ Interactive Streamlit Dashboard

The application includes:

- **Analysis Overview:** Transaction statistics and fraud distribution.
- **Model Evaluation:** Performance metrics and confusion matrix.
- **Individual Transaction Investigation:** Inspect a transaction's predicted fraud probability.
- **High-Risk Transaction Alerts:** View transactions flagged for further review.
- **Prediction Results:** Download transaction predictions as a CSV file.

## 📸 Dashboard Screenshots

### Dashboard Home
![Dashboard Home](screenshots/dashboard-home.png)

### Analysis Overview
![Analysis Overview](screenshots/dashboard-overview.png)

### Model Evaluation
![Model Evaluation](screenshots/model-evaluation.png)

### Individual Transaction Investigation
![Transaction Investigation](screenshots/transaction-investigation.png)

### High-Risk Transaction Alerts
![High-Risk Alerts](screenshots/high-risk-alerts.png)

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── models/
│   └── fraud_detection_model.joblib
│
├── screenshots/
│   ├── dashboard-home.png
│   ├── dashboard-overview.png
│   ├── model-evaluation.png
│   ├── transaction-investigation.png
│   └── high-risk-alerts.png
│
├── app.py
├── .gitignore
└── README.md
```

**Note:** The original dataset is not included in the GitHub repository because of its large file size. Download it from Kaggle and place it inside the `data/` folder before running the application.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/bilallukman/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 3. Download the dataset

Download `creditcard.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it inside the `data/` folder.

### 4. Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser at:

`http://localhost:8501`

## ⚠️ Important Note

This project is developed for educational and portfolio purposes using a historical credit card transaction dataset from 2013.

The results should not be interpreted as proof of production-level fraud detection performance. Real-world deployment would require additional validation, monitoring, and updated transaction data.

## 👨‍💻 Author

**Bilal Lukman**

BCA Data Science Student | Aspiring Data Analyst

GitHub: [bilallukman](https://github.com/bilallukman)

---

 If you find this project interesting, feel free to explore the repository!