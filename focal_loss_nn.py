import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# 1. Load dataset
df = pd.read_csv('/Users/sptsai/Documents/GitHub/Stroke_Prediction/stroke_data.csv')

# 2. One-hot encode categorical features
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# 3. Drop 'id' column if it exists
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 8. Define focal loss function
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss_val = -alpha * tf.math.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_val)
    return loss

# 9. Build neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 10. Compile model
model.compile(
    loss=focal_loss(gamma=2., alpha=0.25),
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 11. Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 12. Predict
y_probs = model.predict(X_test).flatten()
y_pred = (y_probs >= 0.5).astype(int)

# 13. Evaluation
print("\n=== Focal Loss Neural Network ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 14. AUC with NaN check
if np.isnan(y_probs).any():
    print("⚠️ Warning: NaNs found in predicted probabilities!")
    mask = ~np.isnan(y_probs)
    auc = roc_auc_score(y_test[mask], y_probs[mask])
else:
    auc = roc_auc_score(y_test, y_probs)

print("AUC:", auc)
