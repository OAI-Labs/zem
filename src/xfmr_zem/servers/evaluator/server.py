import os
import sys
import opik
from opik.evaluation import evaluate as opik_evaluate
from opik import Opik

from xfmr_zem.server import ZemServer
from loguru import logger
from typing import Any
from dotenv import load_dotenv

# Import our engines
from xfmr_zem.servers.evaluator.factory.models_factory import ModelFactory
from xfmr_zem.servers.evaluator.factory.dataset_factory import DatasetFactory
from xfmr_zem.servers.evaluator.factory.evaluate_factory import EvaluateModelFactory
from xfmr_zem.servers.evaluator.factory.metric_factory import MetricFactory
from xfmr_zem.servers.evaluator.factory.task_factory import TaskFactory

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("evaluator")
@server.tool()
def evaluate(
    dataset_path: str = "data/MCQ_dataset.json",
    dataset_name: str = "mmlu",
    task_type: str = "multiple_choice",
    test_model_engine: str = "huggingface",
    test_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    evaluate_model_engine: str = "gemini",
    evaluate_model_id: str = "gemini-2.5-flash",
    limit: int = 3,
    project_name: str = "Evaluate LM",
    experiment_name: str = "Run Evaluation"
):
    """
    Evaluates a model on a local dataset (JSON) using Opik.
    Pre-process your data into `dataset_path` using `preprocess.py` first.
    """
    
    _ = load_dotenv(override=True)

    # 1. Configure Opik
    logger.info("Configuring Opik...")
    opik.configure(api_key=os.getenv("OPIK_API_KEY", None))
    client = Opik()

    # 2. Get the Model
    logger.info(f"Initializing Model: {test_model_id} ({test_model_engine})")
    model = ModelFactory.get_model(test_model_engine, test_model_id)

    # 2.5 Get Evaluation Model
    logger.info(f"Initializing Evaluation Model: {evaluate_model_id} ({evaluate_model_engine})")
    eval_model = EvaluateModelFactory.get_model(evaluate_model_engine, evaluate_model_id)

    # # 3. Get the Dataset
    logger.info(f"Loading Dataset from local file: {dataset_path} (type={task_type})")
    
    try:
        # Thử lấy dataset đã tồn tại
        dataset = client.get_dataset(name=dataset_name)
        logger.info(f"Dataset '{dataset_name}' already exists, using existing dataset")
        logger.info(f"Dataset has {len(dataset.get_items())} items")
        
    except Exception as e:
        # Dataset chưa tồn tại, tạo mới từ file
        logger.info(f"Dataset '{dataset_name}' not found, creating new dataset from file")
        try:
            dataset_obj = DatasetFactory.get_dataset(
                dataset_type=task_type,
                dataset_path=dataset_path,
                limit=limit
            )
            dataset_items = dataset_obj.load_data()
            dataset = client.get_or_create_dataset(name=dataset_name)
            dataset.insert(dataset_items)
            logger.info(f"Created new dataset with {len(dataset_items)} items")
        
        except Exception as create_error:
            logger.error(f"Failed to create dataset from file: {create_error}")
            raise


    if not dataset:
        logger.warning("Dataset is empty. Aborting evaluation.")
        return {"error": "Dataset is empty."}

    # 4. Get Metrics
    metrics = MetricFactory.get_metrics(task_type, model=eval_model)

    # 5. Define the Task
    task = TaskFactory.get_task(task_type, model)

    # # 6. Run Opik Evaluation
    logger.info("Starting Opik Evaluation...")
    results = opik_evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=metrics,
        project_name=project_name,
    )
    
    return results


if __name__ == "__main__":
    server.run()