import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv('/Users/sptsai/Documents/GitHub/Stroke_Prediction/stroke_data.csv')

# 2. One-hot encode categorical features
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# 3. Drop 'id' column if present
df.drop(columns=['id'], inplace=True, errors='ignore')

# 4. Separate features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 5. Impute missing values (use median)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 6. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 8. SMOTE oversampling on training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 9. Train Logistic Regression
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg.fit(X_train_resampled, y_train_resampled)
y_pred_lr = logreg.predict(X_test)

# 10. Train Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test)

# 11. Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n=== {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

# 12. Evaluate models
evaluate_model(y_test, y_pred_lr, "Logistic Regression (SMOTE)")
evaluate_model(y_test, y_pred_rf, "Random Forest (SMOTE)")

# 13. AUC Scores
auc_lr = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"\nAUC Score (LogReg): {auc_lr:.4f}")
print(f"AUC Score (Random Forest): {auc_rf:.4f}")
