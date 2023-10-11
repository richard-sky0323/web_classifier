#libreries de terceros
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import typing
import pickle
import spacy as sp
import re
import en_core_web_sm
import argparse

#ML libreries
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer

#MLflow libreries
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

#Libreries URL
from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import os

#MLflow libreries
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient



def fetch_web_data():
    web_df = pd.read_csv('C:/Users/Richard Alejandro/Workspace/project-MLflow/Mlflow_project_Linkscribe/website_classification.csv')
    #dataset.head()
    dataset = pd.DataFrame(web_df)
    df = dataset[['website_url','cleaned_website_text','Category']].copy()

    df['category_id'] = df['Category'].factorize()[0]
    #category_id_df = df[['Category', 'category_id']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)
    #id_to_category = dict(category_id_df[['category_id', 'Category']].values)

    return df

def split_dataset(data: pd.DataFrame , test_size=0.25, random_state=42):
    X = data['cleaned_website_text'] # Collection of text
    y = data['category_id'] # Target or the labels we want to predict    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def generate_confusion_matrix(model_name, model):
    """
    Train a model using a pipeline and generate a confusion matrix figure
    :param model_name: Name of the model
    :param model: The model
    :param X_train: Training data features
    :param y_train: Training data labels
    :param X_test: Test data features
    :param y_test: Test data labels
    :return: Figure containing the confusion matrix
    """
    data = fetch_web_data()
    X_train, X_test, y_train, y_test = split_dataset(data)

    category_id_df = data[['Category', 'category_id']].drop_duplicates()
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Category']].values)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    fitted_vectorizer = tfidf.fit(X_train)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)    

    try:
        check_is_fitted(model)
    except NotFittedError:
        m = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
        model=CalibratedClassifierCV(estimator=m, cv="prefit").fit(tfidf_vectorizer_vectors, y_train)
        #model.fit(X_train, y_train) 
    
    predictions = model.predict(tfidf.transform(X_test))
    accuracy = metrics.accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, predictions, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    #conf_mat = confusion_matrix(y_test, predictions, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap="OrRd", fmt='d',
                xticklabels=category_id_df.Category.values, 
                yticklabels=category_id_df.Category.values)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f"Confusion Matrix for {model_name}")
    #sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    
    return fig

def run_experiment(experiment_id, n_splits=5):

    data = fetch_web_data()
    X_train, X_test, y_train, y_test = split_dataset(data)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    fitted_vectorizer = tfidf.fit(X_train)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

    m = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
    model = CalibratedClassifierCV(estimator=m, cv="prefit").fit(tfidf_vectorizer_vectors, y_train)

    model_name = 'LinearSVC'

    print(f'Training {model_name}...')

    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_id = f'{model_name}-{run_id}'

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_id) as run:

        mlflow.log_param("model_name", model_name)

        # Entrena el modelo en todo el conjunto de entrenamiento
        model.fit(tfidf_vectorizer_vectors, y_train)
        predicted = model.predict(tfidf.transform(X_test))
        print(metrics.accuracy_score(y_test, predicted))

        #mlflow.log_metric("accuracy", accuracy_score())

        #y_pred = model.predict(tfidf.transform(X_test))
        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted, average='macro')
        recall = recall_score(y_test, predicted, average='macro')
        f1 = f1_score(y_test, predicted, average='macro')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        signature = infer_signature(tfidf_vectorizer_vectors, model.predict(tfidf_vectorizer_vectors))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature
        )

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        fig = generate_confusion_matrix(model_name, model)
        mlflow.log_figure(fig, f"{model_name}-confusion-matrix.png")


# def run_experiment(experiment_id, n_splits=5):

#     data = fetch_web_data()
#     #X, y = data['cleaned_website_text'], data['Category_id']
#     X_train, X_test, y_train, y_test = split_dataset(data)

#     tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
#     fitted_vectorizer = tfidf.fit(X_train)
#     tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

#     m = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
#     #model_lv= CalibratedClassifierCV(estimator=m, cv="prefit").fit(tfidf_vectorizer_vectors, y_train)

#     models = {
#         'LinearSVC': CalibratedClassifierCV(estimator=m, cv="prefit")
#     }

#     for model_name, model in models.items():
#         print(f'Running {model_name}...')

#         run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
#         run_id = f'{model_name}-{run_id}'

#         with mlflow.start_run(experiment_id=experiment_id, run_name=run_id) as run:

#             kfold = model_selection.KFold(n_splits=n_splits, random_state=7, shuffle=True)
#             cv_results = model_selection.cross_val_score(model, tfidf_vectorizer_vectors, y_train, cv=kfold, scoring='accuracy')
#             print(f"Accuracy: {cv_results.mean():.3f} ({cv_results.std():.3f})")


#             #print(f"Accuracy: {accuracy_mean:.3f} ({accuracy_std:.3f})")
#             #print(f"Precision: {precision_mean:.3f} ({precision_std:.3f})")
#             #print(f"Recall: {recall_mean:.3f} ({recall_std:.3f})")
#             #print(f"F1-Score: {f1_mean:.3f} ({f1_std:.3f})")

#             #print(f"Accuracy: {cv_results.mean():.3f} ({cv_results.std():.3f})")
#             #accuracy = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

#             mlflow.log_metric("accuracy", cv_results.mean())
#             mlflow.log_metric("std", cv_results.std())
#             #mlflow.log_metric("accuracy_mean", accuracy_mean)
#             #mlflow.log_metric("accuracy_std", accuracy_std)
#             #mlflow.log_metric("precision_mean", precision_mean)
#             #mlflow.log_metric("precision_std", precision_std)
#             #mlflow.log_metric("recall_mean", recall_mean)
#             #mlflow.log_metric("recall_std", recall_std)
#             #mlflow.log_metric("f1_mean", f1_mean)
#             #mlflow.log_metric("f1_std", f1_std)
#             mlflow.log_param("model_name", model_name)
#             mlflow.log_param("n_splits", n_splits)

#             for fold_idx, kflod_result in enumerate(cv_results):
#                 mlflow.log_metric(key="crossval", value=kflod_result, step=fold_idx)

            
#             x_train, x_test, y_train, y_test = split_dataset(data)
#             model.fit(tfidf_vectorizer_vectors, y_train)
#             signature = infer_signature(tfidf_vectorizer_vectors, model.predict(tfidf_vectorizer_vectors))
#             mlflow.sklearn.log_model(
#                 sk_model=model,
#                 artifact_path=model_name,
#                 signature=signature
#             )
#             print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
#             # log artifacts
#             fig = generate_confusion_matrix(model_name, model)
#             mlflow.log_figure(fig, f"{model_name}-confusion-matrix.png")

def get_best_run(experiment_id, metric):
    """
    Get the best run for the experiment
    :param experiment_id:  id of the experiment
    :param metric:  metric to use for comparison
    :return:
    """
    client = MlflowClient()

    # Get all the runs for the experiment
    runs = client.search_runs(experiment_id)

    # Find the run with the highest accuracy metric
    best_run = None
    best_metric_value = 0
    for run in runs:
        metric_value = run.data.metrics[metric]
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_run = run
    # Return the best run
    return best_run

def eval_best_model(experiment_id, metric):
    """
    Evaluate the best model
    :param experiment_id:  id of the experiment
    :param metric:  metric to use for comparison when selecting the best model
    :return:
    """
    model = get_best_model(experiment_id, metric)
    # Get the test dataset
    data = fetch_web_data()
    X_train, X_test, y_train, y_test = split_dataset(data)
    # Evaluate the model
    calibrated_svc = CalibratedClassifierCV(estimator=model, cv="prefit")
    predicted = calibrated_svc.predict(X_test)
    #predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f"Best model accuracy: {accuracy:.3f}")

def get_best_model(experiment_id, metric):
    """
    Get the best model for the experiment
    :param experiment_id:
    :param metric:
    :return:
    """
    # Get the best run
    best_run = get_best_run(experiment_id, metric)
    # Get the model artifact URI
    model_uri = f"runs:/{best_run.info.run_id}/{best_run.data.params['model_name']}"
    # Load the model as a PyFuncModel
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def find_model_by_name(registered_model_name):
    """
    Check if a model with the given name already exists in the model registry
    :param registered_model_name:
    :return:
    """
    client = MlflowClient()
    model = client.get_registered_model(registered_model_name)
    return model


def promote_model_to_stage(registered_model_name, stage, version=None):
    """
    Promote the latest version of a model to the given stage
    :param registered_model_name:
    :param stage:
    :return:
    """
    client = MlflowClient()
    model = client.get_registered_model(registered_model_name)
    if version is not None:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=stage,
        )
        return
    latest_versions = [mv.version for mv in model.latest_versions]
    client.transition_model_version_stage(
        name=registered_model_name,
        version=max(latest_versions),
        stage=stage,
    )

def register_best_model(experiment_id, metric, registered_model_name):
    """
    Register the best model in the experiment as a new model in the MLflow Model Registry
    :param experiment_id:
    :param metric:
    :param registered_model_name:
    :return:
    """
    # Get the best run
    best_run = get_best_run(experiment_id, metric)
    # Get the model artifact URI
    model_uri = f"runs:/{best_run.info.run_id}/{best_run.data.params['model_name']}"
    # registered_model = find_model_by_name(registered_model_name)
    # if registered_model is None:
    registered_model = mlflow.register_model(model_uri, registered_model_name)
    return registered_model  


def rollback_model_version(registered_model_name, stage, version):
    """
    Rollback the model version to the given version
    :param registered_model_name:
    :param stage:
    :param version:
    :return:
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage=stage
    )

def get_list_of_models():
    """"
    Obtain the list of models in the registry
    """
    client = MlflowClient()
    for rm in client.search_registered_models():
        print(rm.name)

def call_model_at_stage(registered_model_name, stage, data):
    """
    Call the production model to get predictions
    :param registered_model_name:
    :param stage:
    :param data:
    :return:
    """
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{registered_model_name}/{stage}"
    )
    # Evaluate the model
    predictions = model.predict(data)
    return predictions


def move_model_to_production_form_runname(experiment_id, run_name, model_name):
    """
    Move the model to production from the run name
    :param model_name:
    :param stage:
    :return:
    """
    client = MlflowClient()
    # Get the run ID from the run name
    found_run = None
    for run in client.search_runs(experiment_id):
        if run.info.run_name == run_name:
            found_run = run
            break

    if found_run is None:
        raise Exception(f"Run {run_name} not found")
    
    model_uri = f"runs:/{found_run.info.run_id}/{found_run.data.params['model_name']}"
    try:
        model = client.get_registered_model(model_name)
        if model is not None:
            print(f"Model {model_name} already exists")
    except:
        model = mlflow.register_model(model_uri, model_name)
    return model

def list_experiments_artifacts(experiment_id):
    """
    List the artifacts for all the runs in the experiment
    :param experiment_id:
    :return:
    """
    client = MlflowClient()

    # Get all the runs for the experiment
    runs = client.search_runs(experiment_id)

    # Get the artifacts for each run
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            print(f" - {artifact.path}")

def list_experiment_models(experiment_id):
    """
    List the models for all the runs in the experiment
    :param experiment_id:
    :return:
    """
    client = MlflowClient()

    # Get all the runs for the experiment
    runs = client.search_runs(experiment_id)

    # Get the artifacts for each run
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            if artifact.path.endswith(".pkl"):
                print(f" - {artifact.path}")

def create_experiment(experiment_name):
    """
    Create a new experiment in MLflow
    :param experiment_name:
    :return:
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

def get_best_model_uri(experiment_id, metric):
    """
    Get the best model URI for the experiment
    :param experiment_id:
    :param metric:
    :return:
    """
    # Get the best run
    best_run = get_best_run(experiment_id, metric)
    # Get the model artifact URI
    model_uri = f"runs:/{best_run.info.run_id}/{best_run.data.params['model_name']}"
    return model_uri

if __name__ == '__main__':
    #mlflow.set_tracking_uri("http://myserver.com/mlflow:5000")
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsplits', type=int, default=5)
    parser.add_argument('--nephocs', type=int, default=500)
    args = parser.parse_args()

    # Create a new experiment in MLflow and get experiment ID
    experiment_name = f"Web Classifier"
    experiment_id = create_experiment(experiment_name)

    # move_model_to_production_form_runname(experiment_id, run_name="LDA-20230908-200035", model_name= "modelo-iris")
    # promote_model_to_stage("modelo-iris", "Production")

    #run_experiment(experiment_id, n_splits=args.nsplits)


    #list_experiment_models(experiment_id)
    # run = get_best_run(experiment_id, metric="accuracy")
    # print(run.info.run_name,
    #       run.data.metrics["accuracy"],
    #       run.data.params["model_name"])

    # model_uri = get_best_model_uri(experiment_id, metric="accuracy")
    # print(model_uri)
    
    model_name = "web-classifier-uao"
    stage = "Production"
    #register_best_model(experiment_id, "accuracy", model_name)
    promote_model_to_stage(model_name, stage)
    
    # #rollback_model_version(model_name, stage, 2)

    # data = [{
    #     "sepal-length": 6.9,
    #     "sepal-width": 3.1,
    #     "petal-length": 5.1,
    #     "petal-width": 2.3
    # }]
    # predictions = call_model_at_stage(model_name, stage, data)
    # print(predictions)



