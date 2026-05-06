import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np



wine = load_wine()
X = wine.data  
y = wine.target  
print("Feature sample (raw):", X[0])


scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature sample (normalized):", X[0])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(13,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(X_train, y_train, epochs=20, batch_size=8)


model.evaluate(X_test, y_test)


sample = np.array([[13.2, 2.77, 2.51, 18.5, 96, 2.5, 2.3, 0.3, 1.8, 5.0, 1.05, 3.2, 1050]])  
sample = scaler.transform(sample) 
pred_logits = model.predict(sample)

print("Logits:", pred_logits)

pred_class = np.argmax(pred_logits, axis=1)
print("Predicted class:", pred_class, "->", wine.target_names[pred_class][0])