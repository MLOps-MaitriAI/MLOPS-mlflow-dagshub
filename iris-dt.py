import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='MLOps-MaitriAI', repo_name='MLOPS-mlflow-dagshub', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/MLOps-MaitriAI/MLOPS-mlflow-dagshub.mlflow")
# Load the IRIS dataset
data = load_iris()

# Split the data into training and test sets. (0.8, 0.2) split.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

max_depth=20

mlflow.set_experiment('iris')
# Log model hyperparameters and metrics to the MLflow server
with mlflow.start_run():
    
    # Build and Train the Model
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    
    y_pred=model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)
    
    # Log the accuracy score
    mlflow.log_metric("accuracy", accuracy)

    print('accuracy:',accuracy)
    # Create a confusion matrix plot 
# Predict the labels for the test set
   # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Iris Dataset')
    
    plt.savefig("confusion_matrix.png")
    #log model
    mlflow.sklearn.log_model(model,"decision_tree")
#log tag
    mlflow.set_tag("author","Pradeep")
    mlflow.set_tag("model","decision tree")

    # Log the parameters
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.log_param("max_depth", max_depth)
    

    # mlflow.sklearn.log_model(model, "iris_rf_model")
    # # Save the MLflow run ID
    # run_id = mlflow.active_run().info.run_id
    # print("MLflow Run ID:", run_id)