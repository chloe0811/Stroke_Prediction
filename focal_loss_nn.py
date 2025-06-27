import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
import numpy as np
from prediction_pipeline import load_and_preprocess_data, print_metrics, store_metrics

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.backend.mean(alpha * keras.backend.pow(1. - pt_1, gamma) * keras.backend.log(pt_1)) \
               -keras.backend.mean((1 - alpha) * keras.backend.pow(pt_0, gamma) * keras.backend.log(1. - pt_0))
    return focal_loss_fixed

def run_focal_loss_nn(results):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Apply SMOTE to balance minority class in training data
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Build a deeper neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train_res.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile model with focal loss
    model.compile(
        optimizer='adam',
        loss=focal_loss(gamma=2, alpha=0.25),
        metrics=['accuracy']
    )

    # Train the model for 50 epochs
    model.fit(X_train_res, y_train_res, epochs=50, batch_size=32, verbose=0)

    # Predict probabilities on test set
    y_prob = model.predict(X_test).flatten()

    # Use a lower threshold for classification to increase recall
    threshold = 0.3
    y_pred = (y_prob >= threshold).astype(int)

    # Print metrics and store results
    print_metrics("Focal Loss NN + SMOTE", y_test, y_pred, y_prob)
    store_metrics("FocalLossNN_SMOTE", y_test, y_pred, y_prob, results)
