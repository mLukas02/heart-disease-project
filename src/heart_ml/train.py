import click
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import datetime

def create_pipeline(model_name, random_state):
    if model_name == 'logistic_regression':
        model = LogisticRegression(random_state=random_state)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
    elif model_name == 'xgboost':
        model = XGBClassifier(random_state=random_state)
    elif model_name == 'svm':
        model = SVC(random_state=random_state)
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    return model

def get_param_grid(model_name):
    if model_name == 'logistic_regression':
        return {'max_iter': [100, 200, 300], 'C': [0.1, 1.0, 10.0]}
    elif model_name == 'random_forest':
        return {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
    elif model_name == 'xgboost':
        return {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
    elif model_name == 'svm':
        return {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
    elif model_name == 'knn':
        return {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    return {}

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/processed",
    type=click.Path(exists=True, file_okay=False),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models/model.joblib",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--model-name",
    default='logistic_regression',
    type=click.Choice(['logistic_regression', 'random_forest', 'xgboost', 'svm', 'knn']),
    show_default=True,
)
def train(
    dataset_path: str,
    save_model_path: str,
    random_state: int,
    model_name: str,
) -> None:
    X_train = pd.read_csv(f"{dataset_path}/train_features.csv")
    y_train = pd.read_csv(f"{dataset_path}/train_target.csv")
    X_test = pd.read_csv(f"{dataset_path}/test_features.csv")
    y_test = pd.read_csv(f"{dataset_path}/test_target.csv")
    
     # Ensure y_train is 1-dimensional
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()  # Convert DataFrame to Series if it's a single column DataFrame
    y_train = y_train.values.ravel()  # Convert to numpy array and then ravel
    
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()  
    y_test = y_test.values.ravel() 
    
    # Set the tracking URI to the local directory
    mlflow.set_tracking_uri("file:./mlruns")

    # Ensure experiment exists
    experiment_name = "Heart Disease Prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        
    # Set the run name
    run_name = f"{model_name} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        model = create_pipeline(model_name, random_state)
        param_grid = get_param_grid(model_name)
        
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        accuracy = best_model.score(X_test, y_test)
        
        print("accuracy", accuracy)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        
        click.echo(f"Best Parameters: {grid_search.best_params_}")
        click.echo(f"Accuracy: {accuracy}.")
        
        # Get additional metrics
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)
        
        # Log additional metrics
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1-score", report['1']['f1-score'])
        
        # Log confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=best_model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        dump(best_model, save_model_path)
        mlflow.sklearn.log_model(best_model, "model")
        click.echo(f"Model is saved to {save_model_path}.")

if __name__ == "__main__":
    train()