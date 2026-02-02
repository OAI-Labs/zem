import time
from typing import TYPE_CHECKING, Dict, List, Optional, Type
from uuid import uuid4

from zenml.enums import ExecutionMode
from zenml.logger import get_logger
from zenml.orchestrators import (
    BaseOrchestrator,
    BaseOrchestratorConfig,
    BaseOrchestratorFlavor,
)
from zenml.orchestrators.dag_runner import ThreadedDagRunner
from zenml.utils import string_utils

if TYPE_CHECKING:
    from zenml.models import PipelineRunResponse, PipelineSnapshotResponse
    from zenml.stack import Stack

logger = get_logger(__name__)

class ParallelLocalOrchestrator(BaseOrchestrator):
    """Orchestrator responsible for running pipelines locally in parallel."""
    _orchestrator_run_id: Optional[str] = None

    def submit_pipeline(
        self,
        snapshot: "PipelineSnapshotResponse",
        stack: "Stack",
        base_environment: Dict[str, str],
        step_environments: Dict[str, Dict[str, str]],
        placeholder_run: Optional["PipelineRunResponse"] = None,
    ) -> None:
        """Submits a pipeline to the orchestrator."""
        self._orchestrator_run_id = str(uuid4())
        start_time = time.time()
        
        # Build DAG
        dag = {
            step_name: step.spec.upstream_steps
            for step_name, step in snapshot.step_configurations.items()
        }

        def run_step_wrapper(step_name: str) -> None:
            step = snapshot.step_configurations[step_name]
            self.run_step(step=step)

        # Use ThreadedDagRunner for parallel execution
        dag_runner = ThreadedDagRunner(
            dag=dag,
            run_fn=run_step_wrapper
        )
        
        logger.info("Starting parallel local execution...")
        dag_runner.run()

        run_duration = time.time() - start_time
        logger.info(
            "Parallel pipeline run has finished in `%s`.",
            string_utils.get_human_readable_time(run_duration),
        )
        self._orchestrator_run_id = None

    def get_orchestrator_run_id(self) -> str:
        """Returns the active orchestrator run id."""
        if not self._orchestrator_run_id:
            raise RuntimeError("No run id set.")
        return self._orchestrator_run_id

class ParallelLocalOrchestratorConfig(BaseOrchestratorConfig):
    """Parallel local orchestrator config."""
    @property
    def is_local(self) -> bool:
        return True

class ParallelLocalOrchestratorFlavor(BaseOrchestratorFlavor):
    """Class for the `ParallelLocalOrchestratorFlavor`."""
    @property
    def name(self) -> str:
        return "parallel_local"

    @property
    def config_class(self) -> Type[ParallelLocalOrchestratorConfig]:
        return ParallelLocalOrchestratorConfig

    @property
    def logo_url(self) -> str:
        """A URL to represent the flavor in the dashboard."""
        return "https://public-flavor-logos.s3.eu-central-1.amazonaws.com/orchestrator/local.png"

    @property
    def implementation_class(self) -> Type[ParallelLocalOrchestrator]:
        return ParallelLocalOrchestrator
