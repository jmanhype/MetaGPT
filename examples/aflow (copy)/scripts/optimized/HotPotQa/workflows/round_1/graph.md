# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi (updated for HotpotQA Alternate History)
# @Desc    : Workflow Classes for HotpotQA Alternate History and Basic Tasks

from typing import Literal, Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager
import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
import scripts.optimized.HotPotQa.workflows.template.operator as operator
import scripts.optimized.HotPotQa.workflows.round_1.prompt as prompt_custom
from scripts.optimized.HotPotQa.workflows.template.operator import (
    get_historical_fact_extractor,
    get_alternate_scenario_generator,
    get_plausibility_checker,
    get_narrative_coherence_enhancer,
    get_historical_accuracy_verifier
)

# Import the logging module and configure logging
import logging
import traceback

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("workflow_errors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define Pydantic models for operator responses
class HistoricalFactsResponse(BaseModel):
    response: str = Field(default="")

class AlternateScenarioResponse(BaseModel):
    alternate_scenario: str = Field(default="")

class PlausibilityResponse(BaseModel):
    plausibility_score: float = Field(default=0.0)
    reasoning: str = Field(default="")

class CoherenceResponse(BaseModel):
    enhanced_scenario: str = Field(default="")
    changes_made: List[str] = Field(default_factory=list)

class AccuracyResponse(BaseModel):
    accuracy_score: float = Field(default=0.0)
    inaccuracies: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "Trading"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm: Any,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = llm
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str) -> Tuple[str, float]:
        """
        Implementation of the basic workflow
        
        Args:
            problem (str): The input problem to solve.
        
        Returns:
            Tuple[str, float]: The solution and the total cost.
        """
        solution = await self.custom(input=problem, instruction="")
        return solution['response'], self.llm.cost_manager.total_cost

class HotpotQAWorkflow(Workflow):
    def __init__(
        self,
        name: str,
        llm: Any,
        dataset: DatasetType,
    ) -> None:
        super().__init__(name, llm, dataset)
        # Initialize operators using factory functions
        self.historical_fact_extractor = get_historical_fact_extractor(self.llm)
        self.alternate_scenario_generator = get_alternate_scenario_generator(self.llm)
        self.plausibility_checker = get_plausibility_checker(self.llm)
        self.narrative_coherence_enhancer = get_narrative_coherence_enhancer(self.llm)
        self.historical_accuracy_verifier = get_historical_accuracy_verifier(self.llm)

    async def execute_alternate_history_workflow(
        self, question_input: str
    ) -> Tuple[Dict[str, Any], float]:
        """
        Implementation of the alternate history workflow.
        
        Args:
            question_input (str): Original HotpotQA question.
        
        Returns:
            Tuple[Dict[str, Any], float]: Evaluation results and total cost.
        """
        try:
            # Step 1: Extract historical facts
            extracted_facts_response = await self.historical_fact_extractor(
                input=question_input,
                instruction=prompt_custom.HISTORICAL_FACT_EXTRACTION_PROMPT
            )
            logger.debug(f"Extracted facts response: {extracted_facts_response}")
            extracted_facts = HistoricalFactsResponse(**self._ensure_dict(extracted_facts_response))
            logger.debug(f"Parsed extracted facts: {extracted_facts}")

            # Step 2: Generate alternate scenarios
            generated_scenario_response = await self.alternate_scenario_generator(
                facts=extracted_facts.response,
                instruction=prompt_custom.ALTERNATE_SCENARIO_GENERATION_PROMPT
            )
            logger.debug(f"Generated scenario response: {generated_scenario_response}")
            generated_scenario = AlternateScenarioResponse(**self._ensure_dict(generated_scenario_response))
            logger.debug(f"Parsed generated scenario: {generated_scenario}")

            # Step 3: Check plausibility
            plausibility_response_data = await self.plausibility_checker(
                scenario=generated_scenario.alternate_scenario,
                original_question=question_input,
                instruction=prompt_custom.PLAUSIBILITY_CHECK_PROMPT
            )
            logger.debug(f"Plausibility response data: {plausibility_response_data}")
            plausibility_response = PlausibilityResponse(**self._ensure_dict(plausibility_response_data))
            plausibility_score = plausibility_response.plausibility_score
            logger.debug(f"Plausibility score: {plausibility_score}")

            # Step 4: Enhance narrative coherence
            coherent_scenario_response_data = await self.narrative_coherence_enhancer(
                scenario=generated_scenario.alternate_scenario,
                original_question=question_input,
                instruction=prompt_custom.NARRATIVE_COHERENCE_ENHANCEMENT_PROMPT
            )
            logger.debug(f"Coherent scenario response data: {coherent_scenario_response_data}")
            coherent_scenario_response = CoherenceResponse(**self._ensure_dict(coherent_scenario_response_data))
            coherence_score = len(coherent_scenario_response.changes_made)  # Example coherence metric
            logger.debug(f"Coherence score (number of changes made): {coherence_score}")

            # Step 5: Verify historical accuracy
            accuracy_response_data = await self.historical_accuracy_verifier(
                scenario=coherent_scenario_response.enhanced_scenario,
                original_question=question_input,
                original_facts=extracted_facts.response,
                instruction=prompt_custom.HISTORICAL_ACCURACY_VERIFICATION_PROMPT
            )
            logger.debug(f"Accuracy response data: {accuracy_response_data}")
            accuracy_response = AccuracyResponse(**self._ensure_dict(accuracy_response_data))
            accuracy_score = accuracy_response.accuracy_score
            logger.debug(f"Accuracy score: {accuracy_score}")

            evaluation_results = {
                "plausibility": plausibility_score,
                "coherence": coherence_score,
                "accuracy": accuracy_score,
                "alternate_scenario": generated_scenario.alternate_scenario,
                "enhanced_scenario": coherent_scenario_response.enhanced_scenario
            }

            logger.debug(f"Evaluation results: {evaluation_results}")

            return evaluation_results, self.llm.cost_manager.total_cost
        except Exception as e:
            logger.error(f"Error in execute_alternate_history_workflow: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return {"error": str(e)}, 0.0

    def _ensure_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Ensures that the provided object is a dictionary.
        If it's a Pydantic model, convert it to a dict.
        If it's a list, convert each element.
        Otherwise, convert the object to a string.

        Args:
            obj (Any): The object to ensure is a dictionary.

        Returns:
            Dict[str, Any]: The standardized dictionary.
        """
        if isinstance(obj, dict):
            return {k: self._ensure_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, 'dict'):
            return {k: self._ensure_serializable(v) for k, v in obj.dict().items()}
        elif isinstance(obj, list):
            return {f"item_{i}": self._ensure_serializable(item) for i, item in enumerate(obj)}
        else:
            return {"value": str(obj)}

    def _ensure_serializable(self, obj: Any) -> Any:
        """
        Recursively ensures that an object is JSON serializable.
        Converts non-serializable objects to strings.

        Args:
            obj (Any): The object to serialize.

        Returns:
            Any: The JSON serializable object.
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return self._ensure_dict(obj)
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        else:
            return str(obj)

    async def __call__(self, problem: str) -> Tuple[Dict[str, Any], float]:
        """
        Override the basic workflow call method to use the alternate history workflow.
        
        Args:
            problem (str): The input HotpotQA question.
        
        Returns:
            Tuple[Dict[str, Any], float]: The evaluation results and the total cost.
        """
        return await self.execute_alternate_history_workflow(problem)

# Example usage
if __name__ == "__main__":
    llm_config = {
        "model": "gpt-3.5-turbo",
        "api_key": "your-api-key-here"
    }
    
    # Basic workflow
    basic_workflow = Workflow("BasicWorkflow", llm_config, "HumanEval")
    
    # HotpotQA workflow
    hotpotqa_workflow = HotpotQAWorkflow("HotpotQAWorkflow", llm_config, "HotpotQA")
    
    # You can now use these workflows in your evaluation scripts
