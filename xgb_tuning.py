import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, roc_auc_score
import xgboost as xgb
from prediction_pipeline import load_and_preprocess_data

def auc_scorer(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)

def tune_xgboost():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # You can adjust this parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, (len(y_train) - sum(y_train)) / sum(y_train)],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring=make_scorer(auc_scorer, needs_proba=True),
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    print("Best AUC:", grid_search.best_score_)

    # Optionally test best estimator on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    print("Test AUC:", roc_auc_score(y_test, y_prob))
    print("Test recall:", recall_score(y_test, y_pred))

if __name__ == "__main__":
    tune_xgboost()
