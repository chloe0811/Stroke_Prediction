import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    df = pd.read_csv('stroke_data.csv')
    df = pd.get_dummies(df, columns=[
        'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
    ], drop_first=True)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Fill missing values
    imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
    X = imputer.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

def print_metrics(name, y_test, y_pred, y_prob):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))


def store_metrics(name, y_test, y_pred, y_prob, results):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    results.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    })
    return results