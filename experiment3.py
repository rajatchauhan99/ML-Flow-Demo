import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("experiment3")
n_estimators = 100

# Start an MLflow run
with mlflow.start_run():
    # Define and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the heatmap as an image
    cm_path = "confusion_matrix_heatmap.png"
    plt.savefig(cm_path)
    plt.close()

    # Log parameters, metrics, and artifacts
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    mlflow.log_artifact(cm_path)

    mlflow.log_artifact(__file__)

    print(f"Model and artifacts saved in run {mlflow.active_run().info.run_uuid}")

    # Remove the confusion matrix file after logging
    if os.path.exists(cm_path):
        os.remove(cm_path)
