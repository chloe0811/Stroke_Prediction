import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from prediction_pipeline import load_and_preprocess_data, print_metrics, store_metrics

# 1. Load dataset
df = pd.read_csv('/Users/sptsai/Documents/GitHub/Stroke_Prediction/stroke_data.csv')

# 2. One-hot encode categorical features
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# 3. Drop 'id' column if exists
df.drop(columns=['id'], inplace=True, errors='ignore')

# 4. Split features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 5. Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 6. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 8. Train Logistic Regression with class_weight='balanced'
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 9. Predict probabilities and apply custom threshold
threshold = 0.4  # You can adjust this
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# 10. Evaluation
print("\n=== Logistic Regression (Custom Threshold) ===")
print(f"Threshold: {threshold}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("AUC Score:", roc_auc_score(y_test, y_proba))
