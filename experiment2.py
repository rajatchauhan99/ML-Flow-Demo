import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 2

mlflow.set_experiment("experiment 2")

# Start an MLflow run using context manager
with mlflow.start_run(run_name = "max_depth 2"):
    # Define and train the model
    model = DecisionTreeClassifier( max_depth= max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # Log parameters and metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")

