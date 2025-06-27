import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.ensemble import BalancedBaggingClassifier

# 1. Load dataset
df = pd.read_csv('/Users/sptsai/Documents/GitHub/Stroke_Prediction/stroke_data.csv')

# 2. One-hot encode categorical features
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# 3. Drop ID column if present
df.drop(columns=['id'], inplace=True, errors='ignore')

# 4. Split features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 5. Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 6. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 8. Initialize Balanced Bagging Classifier
model = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(),
    sampling_strategy='auto',
    replacement=False,
    random_state=42,
    n_estimators=50,
    n_jobs=-1
)

# 9. Fit the model
model.fit(X_train, y_train)

# 10. Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 11. Evaluation
print("\n=== Balanced Bagging Classifier ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("AUC Score:", roc_auc_score(y_test, y_proba))
