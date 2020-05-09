import os
import zipfile

import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import CategoricalParameter, HyperparameterTuner
from tqdm import tqdm

from Loader.data_loader import get_image_paths
from train.train_progan import DATA_PATH


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


UPLOAD_DATA = True

sagemaker_session = sagemaker.Session(default_bucket="duck-and-cover")

if __name__ == "__main__":
    if UPLOAD_DATA:
        data_dir = []
        print("Uploading data in {0} to S3/duck-and-cover".format(DATA_PATH),)
        paths = get_image_paths(DATA_PATH)
        zipf = zipfile.ZipFile(DATA_PATH + ".zip", "w", zipfile.ZIP_DEFLATED)
        zipdir(DATA_PATH, zipf)
        zipf.close()
        for path in tqdm(paths):
            key_prefix, _ = os.path.split(path)
            sagemaker.Session().upload_data(
                path=path, key_prefix=key_prefix,
            )

    estimator = TensorFlow(
        entry_point="train/train_hpo.sh",
        source_dir=source_dir,
        train_instance_type=config.get("train_instance_type", "local"),
        train_instance_count=1,
        role="AmazonSageMaker-ExecutionRole-20180822T000617",
        framework_version="1.15.2",
        dependencies=["base", "data", "projects", "ssh"],
        py_version="py3",
        base_job_name=config.get("experiment_name", None),
        metric_definitions=[{"Name": "validation:accuracy", "Regex": "top1 accuracy = (.*?) with",}],
        enable_sagemaker_metrics=True,
    )

    emb_file = config.get("data_input")
    if config.get("train_instance_type", "local") == "local":
        inputs = f"file://{emb_file[0]}"
    elif len(emb_file) > 1:
        inputs = "s3://eb7-datascience/embeddings"
    else:
        emb_file = emb_file[0].split("/")[-1]
        inputs = "s3://eb7-datascience/embeddings/{0}".format(emb_file)

    hyperparameter_ranges = {
        "lstm_size": CategoricalParameter(values=[50, 100]),
        "lr": CategoricalParameter(values=[0.001, 0.005, 0.01]),
    }

    objective_metric_name = "validation:accuracy"
    objective_type = "Maximize"
    metric_definitions = [{"Name": "validation:accuracy", "Regex": "top1 accuracy = (.*?) with"}]

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        strategy="Random",
        max_jobs=6,
        max_parallel_jobs=2,
        objective_type=objective_type,
        base_tuning_job_name=config.get("experiment_name", None) + "-hpo",
        early_stopping_type="Auto",
    )

    tuner.fit(inputs)
