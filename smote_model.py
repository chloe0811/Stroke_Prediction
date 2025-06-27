# smote_model.py

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from prediction_pipeline import load_and_preprocess_data, print_metrics, store_metrics

def run_smote_lr(results):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print_metrics("SMOTE + LogisticRegression", y_test, y_pred, y_prob)
    store_metrics("SMOTE_LogReg", y_test, y_pred, y_prob, results)
