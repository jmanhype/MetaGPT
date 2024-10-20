from typing import Literal, Dict, Any, Tuple
from examples.aflow.scripts.optimized.HotpotQA.workflows.template.operator import (
    historical_fact_extractor,
    alternate_scenario_generator,
    plausibility_checker,
    narrative_coherence_enhancer,
    historical_accuracy_verifier
)
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "Trading"]

class HotpotQAWorkflow:
    def __init__(
        self,
        name: str,
        llm_config: dict,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()

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
        # Step 1: Extract historical facts
        extracted_facts = await historical_fact_extractor(
            llm=self.llm,
            input_data=question_input,
            instruction="Extract relevant historical facts for the following question:"
        )

        # Step 2: Generate alternate scenarios
        generated_scenario = await alternate_scenario_generator(
            llm=self.llm,
            facts=extracted_facts['response'],
            instruction="Generate an alternate historical scenario based on the extracted facts:"
        )

        # Step 3: Check plausibility
        plausibility = await plausibility_checker(
            llm=self.llm,
            scenario=generated_scenario['response'],
            instruction="Evaluate the plausibility of the following alternate scenario:"
        )

        # Step 4: Enhance narrative coherence
        coherent_scenario = await narrative_coherence_enhancer(
            llm=self.llm,
            scenario=generated_scenario['response'],
            instruction="Enhance the narrative coherence of the following scenario:"
        )

        # Step 5: Verify historical accuracy
        accuracy = await historical_accuracy_verifier(
            llm=self.llm,
            scenario=coherent_scenario['response'],
            instruction="Verify the historical accuracy of the following scenario:"
        )

        evaluation_results = {
            "plausibility": plausibility.get("response", 0.0),
            "coherence": coherent_scenario.get("response", 0.0),
            "accuracy": accuracy.get("response", 0.0)
        }

        return evaluation_results, self.llm.cost_manager.total_cost
