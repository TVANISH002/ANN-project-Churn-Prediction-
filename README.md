## 🏦 Customer Churn Prediction (Banking)
End-to-end machine learning project to predict whether a bank customer is likely to churn.

### Why this project
Customer churn directly impacts revenue in the banking sector.  
Accurately identifying customers at risk of leaving enables proactive retention strategies,
improves customer experience, and reduces avoidable revenue loss.  
This project demonstrates how tabular customer data can be transformed into a
**production-style machine learning workflow** for churn risk prediction.

---

### What I built
- Built a binary classification system to estimate customer churn risk  
- Implemented clean preprocessing for mixed tabular data (categorical and numerical features)  
- Trained an Artificial Neural Network (ANN) to generate churn probability scores  
- Saved inference-ready artifacts (model, encoders, scaler) for reproducible predictions  

---

### How it works
- Removed non-predictive identifiers (RowNumber, CustomerId, Surname)  
- Encoded categorical variables:
  - Gender → Label Encoding  
  - Geography → One-Hot Encoding  
- Scaled numerical features using StandardScaler  
- Trained an ANN using TensorFlow/Keras with EarlyStopping to reduce overfitting  
- Generated churn probability scores (0–1) and converted them into final churn decisions  

---

### Key outcome
- Produces interpretable churn risk probabilities to support decision-making  
- Uses a consistent inference pipeline through persisted preprocessing artifacts  
- Demonstrates deployment-ready ML design (training → persistence → reproducible inference)  

---

### Limitations
- The dataset represents customers from a limited set of countries, so predictions
  may not generalize well to new geographic regions without retraining.
- The model relies on static customer attributes and does not include transaction-level
  or time-based behavioral data.
- Churn is treated as a binary outcome without modeling underlying causes.
- Class imbalance may influence decision thresholds depending on business objectives.

---

### Tools
Python, Pandas, NumPy, scikit-learn, TensorFlow/Keras

---

### Dataset
Source: `Churn_Modelling.csv`  

Each record represents a bank customer with attributes such as:
- Credit score, age, tenure, balance  
- Geography, gender  
- Product usage and activity indicators  

---

### Project structure
- `experiments.ipynb` – Data preprocessing, feature engineering, and model training  
- `prediction.ipynb` – Inference using saved artifacts  
- `app.py` – Inference application logic  
- `Churn_Modelling.csv` – Dataset  
- `model.h5` – Trained ANN model  
- `scaler.pkl` – StandardScaler  
- `label_encoder_gender.pkl` – LabelEncoder  
- `onehot_encoder_geo.pkl` – OneHotEncoder  
- `requirements.txt` – Project dependencies  

---

### Future improvements
- Incorporate transaction-level and time-series customer behavior  
- Extend the model to new geographic regions and evaluate generalization  
- Add explainability techniques to understand churn drivers  
- Deploy inference service via FastAPI and cloud infrastructure  

---
