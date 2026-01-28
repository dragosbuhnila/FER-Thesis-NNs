import os
import mlflow
from tensorflow import keras


def main():
    print("Starting MLflow TensorFlow example...", flush=True)

    # Link to MLflow tracking server
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set. Export it before running training.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("tf_example")
    print(f"Set MLflow tracking URI to {tracking_uri} and experiment.")

    # Prepare data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print("Loaded and preprocessed MNIST data.")

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
    print("Built and compiled the model.")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 32)
        print("Logged hyperparameters.")

        # Train
        history = model.fit(
            x_train,
            y_train,
            validation_split=0.1,
            epochs=5,
            batch_size=32
        )
        print("Model training completed.")

        # Evaluate
        loss, acc = model.evaluate(x_test, y_test)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", acc)
        print("Evaluated model and logged metrics.")

        # Log model
        mlflow.tensorflow.log_model(model, artifact_path="model")
        print("Logged the trained model to MLflow.")



if __name__ == "__main__":
    main()