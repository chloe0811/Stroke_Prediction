from prediction_pipeline import load_and_preprocess_data, print_metrics, store_metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedBaggingClassifier
from focal_loss_nn import run_focal_loss_nn
from smote_model import run_smote_lr
import xgboost as xgb

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
results = []

# === Method 1: Logistic Regression + Threshold ===
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# predict_proba gives [P(class=0), P(class=1)]
y_prob = logreg.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

print_metrics(f"LogReg + Threshold={threshold}", y_test, y_pred, y_prob)
store_metrics("LogReg+Threshold", y_test, y_pred, y_prob, results)

# === Method 2: Cost-sensitive Logistic Regression ===
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

cs_logreg = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
cs_logreg.fit(X_train, y_train)

y_pred = cs_logreg.predict(X_test)
y_prob = cs_logreg.predict_proba(X_test)[:, 1]

print_metrics("Cost-sensitive LR", y_test, y_pred, y_prob)
store_metrics("CostSensitiveLR", y_test, y_pred, y_prob, results)

# === Method 3: XGBoost Weighted ===
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)  # ratio of negative/positive

xgb_clf = xgb.XGBClassifier(scale_pos_weight=pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
y_prob = xgb_clf.predict_proba(X_test)[:, 1]

print_metrics("XGBoost Weighted", y_test, y_pred, y_prob)
store_metrics("XGBoostWeighted", y_test, y_pred, y_prob, results)

# === Method 4: Balanced Bagging + Logistic Regression ===
bagger = BalancedBaggingClassifier(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    sampling_strategy='auto',
    replacement=False,
    random_state=42,
    n_estimators=10
)
bagger.fit(X_train, y_train)

y_pred = bagger.predict(X_test)
if hasattr(bagger, "predict_proba"):
    y_prob = bagger.predict_proba(X_test)[:, 1]
else:
    y_prob = y_pred  # fallback

print_metrics("BalancedBagging LR", y_test, y_pred, y_prob)
store_metrics("BalancedBaggingLR", y_test, y_pred, y_prob, results)

# === Method 5: Focal Loss NN ===
run_focal_loss_nn(results)

# === Method 6: SMOTE + Logistic Regression ===
run_smote_lr(results)

# === Summary ===
print("\n\n=== Summary of Results ===")
import pandas as pd
summary = pd.DataFrame(results)
print(summary)
