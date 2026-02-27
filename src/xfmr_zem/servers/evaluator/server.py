import os
import sys
import re
import opik
import requests
from opik.evaluation import evaluate as opik_evaluate
from opik import Opik

from xfmr_zem.server import ZemServer
from loguru import logger
from typing import Any
from dotenv import load_dotenv

# Import our engines
from xfmr_zem.servers.evaluator.factory.models import ModelFactory
from xfmr_zem.servers.evaluator.factory.dataset import DatasetFactory
from xfmr_zem.servers.evaluator.factory.evaluate import EvaluateModelFactory
from xfmr_zem.servers.evaluator.factory.metric import MetricFactory
from xfmr_zem.servers.evaluator.factory.task import TaskFactory

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("evaluator")

def get_access_token():
    usage = [
        "Set OPIK_TOKEN_URL, OPIK_USERNAME, OPIK_PASSWORD in .env or the environment.",
        "Call get_access_token() only after the above settings are confirmed.",
        "Ensure the Opik token endpoint is reachable from this host."
    ]
    try:
        # Đường dẫn hoàn chỉnh để xin cấp token từ Keycloak
        token_url = os.getenv("OPIK_TOKEN_URL")
        username = os.getenv("OPIK_USERNAME")
        password = os.getenv("OPIK_PASSWORD")

        missing = [
            name for name, value in [
                ("OPIK_TOKEN_URL", token_url),
                ("OPIK_USERNAME", username),
                ("OPIK_PASSWORD", password),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing)}. "
                "See usage instructions."
            )

        request_headers = {"Content-Type": "application/x-www-form-urlencoded"}

        request_payload = {
            "client_id": "opik-cli",
            "grant_type": "password",
            "username": username,
            "password": password,
        }

        response = requests.post(url=token_url, headers=request_headers, data=request_payload)
        response.raise_for_status()
        response_data = response.json()
        access_token = response_data.get("access_token")

        if not access_token:
            raise ValueError("Response did not contain an 'access_token' value.")

        return access_token

    except requests.exceptions.RequestException as network_error:
        raise RuntimeError(
            f"Unable to reach token endpoint: {network_error}. Usage: {' | '.join(usage)}"
        ) from network_error
    except Exception as error:
        raise RuntimeError(
            f"{error}. Usage: {' | '.join(usage)}"
        ) from error

@server.tool()
def build_opik_dataset(
    dataset_path: str = "data/MCQ_dataset.json",
    dataset_name: str = "mmlu",
    dataset_type: str = "multiple_choice",
    use_access_token: bool = False,
    workspace: str = None,
    project: str = None,
    limit: int = 10,
    reset: bool = False,
) -> Any:
    """
    Loads a local dataset (JSON) and uploads it to Opik.
    Pre-process your data into `dataset_path` using `preprocess.py` first.
    
    Args:
        dataset_path: Path to the local dataset file
        dataset_name: Name of the dataset in Opik
        dataset_type: Type of task (e.g., 'multiple_choice')
        limit: Maximum number of items to load
        reset: If True, delete existing dataset and create new one
    """
    _ = load_dotenv(override=True)

    # 1. Configure Opik
    logger.info("Configuring Opik...")

    host_url = os.getenv("OPIK_BASE_URL", None)
    api_key = os.getenv("OPIK_API_KEY", None)

    logger.info(f"your host_url: {host_url}")
    logger.info(f"your api_key: {api_key}")
    if (use_access_token):
        access_token = get_access_token()
        api_key = f"Bearer {access_token}"

    if (not api_key):
        raise ValueError("api_key not found.")

    if (not host_url):
        raise ValueError("host_url not found.")

    client = Opik(
        host=host_url,
        api_key=api_key,
        workspace=workspace,
        project_name = project
    )

    trace = client.trace(
        name="sdk_client_test",
        input={"test": "client_initialization"},
        output={"status": "success"}
    )
    logger.info(f"[+] Successfully created trace in project '{project}'")

    # 2. Load and Upload Dataset
    logger.info(f"Loading Dataset from local file: {dataset_path} (type={dataset_type}, reset={reset})")
    try:
        # Handle reset - delete existing dataset if needed
        if reset:
            try:
                existing_dataset = client.get_dataset(name=dataset_name)
                logger.warning(f"Reset=True, deleting existing dataset '{dataset_name}'")
                client.delete_dataset(name=dataset_name)
                logger.info(f"Successfully deleted existing dataset")
            except Exception:
                logger.info(f"No existing dataset to delete or deletion failed")

        # Load data from file
        dataset_obj = DatasetFactory.get_dataset(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            limit=limit
        )
        dataset_items = dataset_obj.load_data()
        
        # Create dataset (will create new or get existing if not reset)
        dataset = client.get_or_create_dataset(name=dataset_name)
        dataset.insert(dataset_items)
        
        action = "reset_and_created" if reset else "created_or_updated"
        logger.info(f"Successfully {action} dataset '{dataset_name}' with {len(dataset_items)} items")
        
        return server.save_output([{"dataset_name": dataset_name}])
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return server.save_output([{
            "status": "error",
            "dataset_name": dataset_name,
            "error": str(e),
            "reset": reset
        }])


@server.tool()
def evaluate(
    data: Any,
    dataset_name: str = "mmlu",
    test_model_engine: str = "huggingface",  
    test_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",

    evaluate_model_engine: str = "local",    
    evaluate_provider: str = "huggingface",
    evaluate_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",

    task_type: str = "generative",     
    project_name: str = "Evaluate LM",
    metrics: list = None,
    limit: int = 1,
    custom_context: list[str] = None,
    experiment_name: str = "Run Evaluation",
) -> Any:
    """
    Evaluates a model on an existing Opik dataset.
    """   
    _ = load_dotenv(override=True)

    # 1. Configure Opik
    logger.info("Configuring Opik...")
    opik.configure(
        url=os.getenv("OPIK_BASE_URL", None),
        api_key=os.getenv("OPIK_API_KEY", None)
    )
    client = Opik()

    # 2. Get the Model
    logger.info(f"Initializing Model: {test_model_id} ({test_model_engine})")
    model = ModelFactory.get_model(test_model_engine, test_model_id)

    # 2.5 Get Evaluation Model
    logger.info(f"Initializing Evaluation Model: {evaluate_model_id} (Engine: {evaluate_model_engine}, Provider: {evaluate_provider})")
    eval_model = EvaluateModelFactory.get_model(engine = evaluate_model_engine, 
                                                provider = evaluate_provider, 
                                                model_id = evaluate_model_id)

    # 3. Get the Dataset
    logger.info(f"Loading Dataset from Opik: {dataset_name}")
    
    try:
        dataset = client.get_dataset(name=dataset_name)
        dataset_items = dataset.get_items()
        logger.info(f"Dataset '{dataset_name}' found with {len(dataset_items)} items")
    except Exception as e:
        logger.error(f"Failed to get dataset '{dataset_name}': {e}")
        raise ValueError(f"Dataset '{dataset_name}' not found. Please run load_opik_dataset first.")




    if not dataset:
        logger.warning("Dataset is empty. Aborting evaluation.")
        return {"error": "Dataset is empty."}

    # 4. Get Metrics
    if metrics is None:
        metrics = []
    logger.info(f"Metrics: {len(metrics)}")
    scoring_metrics = MetricFactory.get_metrics(metrics, model=eval_model)

    # 5. Define the Task
    task_runner = TaskFactory.get_task(task_type, model, custom_context=custom_context)

    # # 6. Run Opik Evaluation
    logger.info("Starting Opik Evaluation...")
    results = opik_evaluate(
        dataset=dataset,
        task=task_runner.run,
        nb_samples=limit,
        scoring_metrics=scoring_metrics,
        project_name=project_name,
        experiment_name=experiment_name
    )    
    return results


if __name__ == "__main__":
    server.run()
    
