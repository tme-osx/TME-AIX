"""Data science workflow for revenue assurance."""

import os

import kfp.compiler
from dotenv import load_dotenv
from kfp import dsl


load_dotenv(override=True)

kubeflow_endpoint = os.getenv("KUBEFLOW_ENDPOINT", "http://ds-pipeline-dspa.tme-aix.svc.cluster.local:8888")
base_image = os.getenv("BASE_IMAGE", "image-registry.openshift-image-registry.svc.cluster.local:5000/tme-aix/environment:latest")

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def data_prep(
    x_train_file: dsl.Output[dsl.Dataset],
    x_test_file: dsl.Output[dsl.Dataset],
    y_train_file: dsl.Output[dsl.Dataset],
    y_test_file: dsl.Output[dsl.Dataset],
):

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import lzma
    import shutil

    def get_data() -> pd.DataFrame:
        # Extract the .xz file
        with lzma.open('data/telecom_revass_data.csv.xz', 'rb') as f_in:
            with open('data/telecom_revass_data.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Load the synthetic telecom data
        data_path = "data/telecom_revass_data.csv"
        data = pd.read_csv(data_path)

        # Display basic information about the dataset
        data.info()

        print("Initial Dataset:")
        print(data.head())

        return data

    def create_training_set(dataset: pd.DataFrame, test_size: float = 0.3):
        # Check for missing values
        missing_values = dataset.isnull().sum()
        print("Missing values in each column:", missing_values)

        # Convert categorical variables to numeric
        dataset = pd.get_dummies(dataset, columns=['Plan_Type'], drop_first=True)

        # Split the data into features and target variable
        X = dataset.drop('Fraud', axis=1)
        y = dataset['Fraud']

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        return x_train, x_test, y_train, y_test

    def save_df(object_file, target_object: pd.DataFrame):
        target_object.to_pickle(object_file)

    dataset = get_data() 
    x_train, x_test, y_train, y_test = create_training_set(dataset)

    save_df(x_train_file.path, x_train)
    save_df(x_test_file.path, x_test)
    save_df(y_train_file.path, y_train)
    save_df(y_test_file.path, y_test)


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def validate_data():
    pass


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def train_model(
    x_train_file: dsl.Input[dsl.Dataset],
    y_train_file: dsl.Input[dsl.Dataset],
    model_file: dsl.Output[dsl.Model],
):
    import pickle
    import pandas as pd
    from imblearn.ensemble import BalancedRandomForestClassifier

    def load_df(object_file):
        return pd.read_pickle(object_file)

    def save_model(object_file, x_train: pd.DataFrame, model):
        with open(object_file, "wb") as f:
            pickle.dump((model, x_train.columns.tolist()), f)

    def train_default(x_train: pd.DataFrame, y_train: pd.DataFrame):
        # Initialize and train the BalancedRandomForestClassifier with Fine-tuned hyperparameters 
        model = BalancedRandomForestClassifier(
            random_state=42,
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=2,
            max_features='sqrt',
            max_depth=100,
            bootstrap=True,
            sampling_strategy='all',  # Set to 'all' to adopt future behavior
            replacement=True  # Set to 'True' to silence the warning
        )

        model.fit(x_train, y_train)

        return model

    x_train = load_df(x_train_file.path)
    y_train = load_df(y_train_file.path)

    model = train_default(x_train, y_train)

    save_model(model_file.path, x_train, model)


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def evaluate_model(
    x_test_file: dsl.Input[dsl.Dataset],
    y_test_file: dsl.Input[dsl.Dataset],
    model_file: dsl.Input[dsl.Model],
    mlpipeline_metrics_file: dsl.Output[dsl.Metrics],
):
    import pickle
    import json
    import pandas as pd

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    def load_df(object_file):
        return pd.read_pickle(object_file)

    def load_model(object_file):
        with open(object_file, "rb") as f:
            target_object, feature_names = pickle.load(f)
        return target_object

    x_test = load_df(x_test_file.path)
    y_test = load_df(y_test_file.path)
    model = load_model(model_file.path)

    # Make predictions on the test set with BalancedRandomForestClassifier
    y_pred = model.predict(x_test)

    # Evaluate the model
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    print("---------------")
    print("BalancedRandomForestClassifier Results:")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print("\nAccuracy Score:")
    print(acc_score)
    print("---------------")

    metrics = {
        "metrics": [
            {
                "name": "accuracy-score",
                "numberValue": acc_score,
                "format": "PERCENTAGE",
            },
        ]
    }

    with open(mlpipeline_metrics_file.path, "w") as f:
        json.dump(metrics, f)


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas"],
)
def validate_model(model_file: dsl.Input[dsl.Model]):
    import pickle
    import pandas as pd

    with open(model_file.path, 'rb') as model_file:
        model, feature_names = pickle.load(model_file)

    data = {
        "Call_Duration": 5.6,
        "Data_Usage": 150.3,
        "Sms_Count": 20,
        "Roaming_Indicator": 0,
        "MobileWallet_Use": 1,
        "Plan_Type": "prepaid",
        "Cost": 50.75,
        "Cellular_Location_Distance": 1.2,
        "Personal_Pin_Used": 1,
        "Avg_Call_Duration": 4.3,
        "Avg_Data_Usage": 120.6,
        "Avg_Cost": 45.7
    }
    # Convert the JSON data to a DataFrame
    input_values = pd.DataFrame([data])

    # Ensure the columns match the training data
    input_values = input_values.reindex(columns=feature_names, fill_value=0)

    # Make predictions using the loaded model
    prediction = model.predict(input_values)

    # Map the prediction to a more user-friendly response
    prediction_label = 'Fraud' if prediction[0] == 1 else 'Non-Fraud'

    print(f"Prediction: {prediction_label}")


@kfp.dsl.pipeline(
    name="Pipeline",
)
def pipeline(model_obc: str = "model"):
    data_prep_task = data_prep()

    train_model_task = train_model(
        x_train_file=data_prep_task.outputs["x_train_file"],
        y_train_file=data_prep_task.outputs["y_train_file"],
    )

    evaluate_model_task = evaluate_model(
        x_test_file=data_prep_task.outputs["x_test_file"],
        y_test_file=data_prep_task.outputs["y_test_file"],
        model_file=train_model_task.output,
    )

    validate_model_task = validate_model(model_file=train_model_task.output,)


if __name__ == "__main__":
    print(f"Connecting to kfp: {kubeflow_endpoint}")

    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            bearer_token = f.read().rstrip()
    else:
        bearer_token = os.environ["BEARER_TOKEN"]

    # Check if the script is running in a k8s pod
    # Get the CA from the service account if it is
    # Skip the CA if it is not
    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    if os.path.isfile(sa_ca_cert):
        ssl_ca_cert = sa_ca_cert
    else:
        ssl_ca_cert = None

    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )
    result = client.create_run_from_pipeline_func(pipeline, arguments={}, experiment_name="revass")
    print(f"Starting pipeline run with run_id: {result.run_id}")
    client.wait_for_run_completion(result.run_id, 300) #timeout after 5 mn
