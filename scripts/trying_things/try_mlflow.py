import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tf_example")

# Prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 32)

    # Train
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=32
    )

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", acc)

    # Log model
    mlflow.tensorflow.log_model(model, artifact_path="model")
