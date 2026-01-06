## 🏦 Customer Churn Prediction (Banking)
End-to-end machine learning project to predict whether a bank customer is likely to churn.

### Why this project
Customer churn directly impacts revenue in the banking industry.  
Predicting churn early helps teams prioritize retention efforts, improve customer experience,
and reduce avoidable revenue loss. This project demonstrates how tabular customer data can
be transformed into a production-style ML workflow for churn risk prediction.

### What I built
- Built a binary classification system to predict churn risk using customer attributes  
- Implemented clean preprocessing for mixed tabular data (categorical + numerical features)  
- Trained an Artificial Neural Network (ANN) model for churn probability estimation  
- Saved inference-ready artifacts (model + encoders + scaler) for reproducible predictions  

### How it works
- Removed non-predictive identifiers (RowNumber, CustomerId, Surname)  
- Encoded categorical features:
  - Gender → Label Encoding
  - Geography → One-Hot Encoding  
- Scaled numeric features using StandardScaler  
- Trained an ANN (TensorFlow/Keras) and used EarlyStopping to reduce overfitting  
- Output is a churn probability (0–1), converted into a final churn decision  

### Key outcome
- Produces an interpretable churn risk probability for decision support  
- Uses a consistent inference pipeline via saved preprocessing artifacts  
- Demonstrates deployment-ready ML design (training → persistence → reproducible inference)  

### Tools
Python, Pandas, NumPy, scikit-learn, TensorFlow/Keras

