import json
from multiprocessing.connection import wait

import sagemaker
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig
from sagemaker.inputs import FileSystemInput

sess = sagemaker.Session()
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

if __name__ == "__main__":
    # metric definition to extract the results
    metric_definitions = [
        {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
        {"Name": "train_samples_per_second", "Regex": "train_samples_per_second.*=\D*(.*?)$"},
        {"Name": "epoch", "Regex": "epoch.*=\D*(.*?)$"},
        {"Name": "f1", "Regex": "f1.*=\D*(.*?)$"},
        {"Name": "exact_match", "Regex": "exact_match.*=\D*(.*?)$"},
    ]

    # instance configurations
    instance_type = "ml.p3.16xlarge"
    instance_count = 1
    volume_size = 200

    with open("config.json") as f:
        config = json.loads(f.read())

    role = config["aws_config"]["role"]
    smp_options = config["smp_options"]
    mpi_options = config["mpi_options"]
    hyperparameters = config["hyperparameters"]
    distribution = {"smdistributed": {"modelparallel": smp_options}, "mpi": mpi_options}
    file_sytem_id = config["fsx_options"]["file_system_id"]
    directory_path = config["fsx_options"]["directory_path"]



    file_system_input = FileSystemInput(file_system_id=file_sytem_id,
                                        file_system_type="FSxLustre",
                                        directory_path=directory_path,
                                        file_system_access_mode='ro')


    native_huggingface_estimator = HuggingFace(
        entry_point="run_mlm.py",
        metrics_definition=metric_definitions,
        instance_type=instance_type,
        instance_count=instance_count,
        subnets=['xxxxxxxxxxxxxx'],
        security_group_ids=['xxxxxxxxxxxxxx'],
        volume_size=volume_size,
        role=role,
        transformers_version="4.17",
        pytorch_version="1.10",
        py_version="py38",
        distribution=distribution,
        hyperparameters=hyperparameters,
        disable_profiler=True,
        debugger_hook_config=False,
    )
    native_huggingface_estimator.fit(inputs={"training": file_system_input}, wait=False)


    # Initialize the Amazon Training Compiler
    compiler_config = TrainingCompilerConfig()

    # To use SageMaker Training Compiler in a Distributed setting, please use a wrapper script to invoke your training script
    LAUNCH_SM_TRAINING_COMPILER = "launch_sm_training_compiler.py"
    SORCE_DIR = "./optimized"
    hyperparameters["training_script"] = "optimized/run_mlm.py"

    optimized_huggingface_estimator = HuggingFace(
        entry_point=LAUNCH_SM_TRAINING_COMPILER,
        source_dir=SORCE_DIR,
        compiler_config=compiler_config,
        metrics_definition=metric_definitions,
        instance_type=instance_type,
        instance_count=instance_count,
        subnets=['xxxxxxxxxxxxxx'],
        security_group_ids=['xxxxxxxxxxxxxx'],
        volume_size=volume_size,
        role=role,
        transformers_version="4.17",
        pytorch_version="1.10",
        py_version="py38",
        hyperparameters=hyperparameters,
        disable_profiler=True,
        debugger_hook_config=False,
    )
    optimized_huggingface_estimator.fit(inputs={"training": file_system_input}, wait=False)