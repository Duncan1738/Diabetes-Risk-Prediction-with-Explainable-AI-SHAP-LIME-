# Diabetes Risk Prediction with Explainable AI (SHAP & LIME)

This project builds and evaluates machine learning models to predict diabetes risk based on key health indicators. It also integrates **explainable AI tools** like **SHAP** and **LIME** to interpret and visualize model predictions — helping both researchers and practitioners understand the model's decision-making process.

---

## 📌 Objectives

- Train and evaluate predictive models using the PIMA Diabetes dataset
- Apply logistic regression, random forest, and XGBoost classifiers
- Use SHAP for global feature importance and LIME for local interpretability
- Demonstrate how machine learning can assist in medical decision support

---

## 🧾 Dataset

- **Name**: PIMA Indians Diabetes Dataset  
- **Source**: [Kaggle - UCI PIMA Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0: No diabetes, 1: Diabetes)

---

## ⚙️ Technologies & Libraries

| Category        | Tools Used                         |
|----------------|-------------------------------------|
| Programming     | Python 3.x                          |
| Notebook        | Jupyter Notebook / Google Colab     |
| ML Models       | LogisticRegression, RandomForest, XGBoost |
| Explainability  | SHAP, LIME                          |
| Visualization   | Matplotlib, Seaborn                 |
| Others          | scikit-learn, pandas, numpy, joblib |

---

## 📊 Model Performance

| Model              | Accuracy | ROC AUC |
|-------------------|----------|---------|
| Logistic Regression | ~78%     | ~82%    |
| Random Forest       | ~85%     | ~88%    |
| XGBoost             | ✅ Best | ✅ Best |

> 🧠 SHAP shows that **Glucose**, **BMI**, and **Age** are the most influential predictors.

---

## 📁 Project Structure
diabetes-risk-prediction/ ├── diabetes_prediction.ipynb # Main notebook ├── xgb_diabetes_model.pkl # Saved XGBoost model (optional) ├── README.md └── requirements.txt # Optional


## ▶️ How to Run

1. Clone the repository https://github.com/Duncan1738/Diabetes-Risk-Prediction-with-Explainable-AI-SHAP-LIME-.git
2. Upload the dataset if needed (CSV)
3. Install requirements:
   ```bash
   pip install shap lime xgboost scikit-learn

👤 Author
Duncan Kibet
PhD Student in Big Data, Chosun University

📜 License
MIT License
