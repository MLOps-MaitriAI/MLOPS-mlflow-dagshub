import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os

# Initialize DagsHub and set MLflow tracking URI
dagshub.init(repo_owner='MLOps-MaitriAI', repo_name='MLOPS-mlflow-dagshub', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MLOps-MaitriAI/MLOPS-mlflow-dagshub.mlflow")

# Load the IRIS dataset
data = load_iris()

# Split the data into training and test sets. (0.8, 0.2) split.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

max_depth = 13

# Try to set the experiment name in MLflow
try:
    mlflow.set_experiment('builds')
except mlflow.exceptions.MlflowException as e:
    print(f"Warning: Could not set experiment. Using default experiment. Error: {e}")

# Enable autologging
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run() as run:
    # Build and Train the Model
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # # Create a confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix for Iris Dataset')

    # # Save the confusion matrix plot
    # plt.savefig("confusion_matrix.png")
    # plt.close()  # Close the plot to free up memory

    # # Log the confusion matrix plot as an artifact
    # mlflow.log_artifact("confusion_matrix.png")

    # # Log metrics
    # mlflow.log_metric("accuracy", accuracy)

    # # Clean up the confusion matrix image file
    # os.remove("confusion_matrix.png")

    # Register the model
    model_name = "DAG"
    model_uri = f"runs:/{run.info.run_id}/model"

    try:
        # Register the model
        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Model '{model_name}' registered successfully.")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error registering model: {e}")

    # Optionally, transition the model to a specific stage (e.g., Production) 
    client = mlflow.tracking.MlflowClient()

    try:
        # Get latest version of the model
        latest_version_info = client.get_latest_versions(model_name, stages=["None"])[0]
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_info.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {latest_version_info.version} transitioned to Production stage.")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error transitioning model version stage: {e}")


