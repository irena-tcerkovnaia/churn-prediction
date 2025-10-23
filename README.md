# Telephone Subscription Churn Prediction

This project predicts customer churn using demographic, billing, and usage data.  
It was developed as a final data science project (Seneca College) to explore machine learning approaches for telecom churn modeling.

---

## Project Overview

Customer churn prediction helps telecom companies identify clients likely to cancel their subscription, allowing proactive retention strategies.  
This project applies multiple models — from logistic regression to advanced gradient boosting — to classify customers as **Churn** or **No Churn**.

---

## Machine Learning Approach

- **Data preprocessing:** Missing value handling, categorical encoding, feature scaling  
- **Imbalance handling:** SMOTE oversampling  
- **Model training:** Logistic Regression, Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost  
- **Hyperparameter tuning:** GridSearchCV, RandomizedSearch, Optuna  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, LogLoss

---

## Key Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 | LogLoss |
|--------|-----------|----------|------------|--------|------|----------|
| **Baseline** | 0.788 | 0.838 | 0.620 | 0.519 | 0.565 | 0.425 |
| **ClassWeight** | 0.753 | 0.838 | 0.523 | **0.789** | 0.629 | 0.499 |
| **CatBoost (Optuna)** | **0.792** | **0.845** | 0.647 | 0.476 | 0.549 | **0.416** |
| **CatBoost (Base)** | 0.747 | 0.841 | 0.515 | 0.810 | **0.630** | 0.492 |

📈 **Best performing model:** CatBoost (ROC-AUC ≈ 0.845, Recall ≈ 0.81)

---

## Project Structure

churn-prediction/
├── data/ # Sample dataset (no confidential data)
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── src/ # Python scripts for modular code
│ ├── EDA_functions.py
│ ├── Model_functions.py
│ └── Evaluation_Metrics.py
├── requirements.txt # Project dependencies
├── README.md # Project documentation

---

## How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/churn-prediction.git
   cd churn-prediction
   
2. Create & activate actual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Open notebooks

jupyter lab

Run the notebooks inside the notebooks/ folder step by step.

 # Technologies Used
Python 3.9+,
Pandas, NumPy, Scikit-learn
CatBoost, LightGBM, XGBoost
Optuna (for hyperparameter optimization)
Imbalanced-learn (for SMOTE)
Matplotlib, Seaborn
JupyterLab, PyCharm

 # Business Impact
Accurate churn prediction enables telecom operators to:

Reduce churn by identifying at-risk customers early

Target retention campaigns efficiently

Improve customer lifetime value and forecasting accuracy


# Author
Irena Tcerkovnaia

Data Analyst / Aspiring Data Scientist

Toronto, Canada

📧 [tcerkovnaiairena@gmail.com]







