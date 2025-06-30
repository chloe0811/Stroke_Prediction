import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, recall_score, roc_auc_score
from prediction_pipeline import load_and_preprocess_data

def auc_scorer(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)

def tune_logreg():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = [{0: weights[0], 1: weights[1]}, None]  # Also try no class weight

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'class_weight': class_weights,
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }

    logreg = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring=make_scorer(auc_scorer, needs_proba=True),
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    print("Best AUC:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    print("Test AUC:", roc_auc_score(y_test, y_prob))
    print("Test recall:", recall_score(y_test, y_pred))

if __name__ == "__main__":
    tune_logreg()
