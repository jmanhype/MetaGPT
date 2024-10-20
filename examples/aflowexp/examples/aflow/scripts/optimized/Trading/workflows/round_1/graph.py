from typing import Literal, Callable, Tuple
import json

from examples.aflow.scripts.optimized.Trading.workflows.template.operator import (
    generate_trading_strategy,
    evaluate_trading_strategy
)
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "Trading"]


class TradingWorkflow:
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

    async def execute_trading_workflow(
        self, strategy_input: str, historical_data: str
    ) -> Tuple[Dict[str, Any], float]:
        """
        Implementation of the trading workflow.

        Args:
            strategy_input (str): Input data or parameters for strategy generation.
            historical_data (str): Reference to historical trading data for evaluation.

        Returns:
            Tuple[Dict[str, Any], float]: Evaluated strategy response and total cost.
        """
        # Generate a new trading strategy
        generated_strategy = await generate_trading_strategy(
            llm=self.llm,
            input_data=strategy_input,
            instruction="Generate a new trading strategy based on the following parameters:"
        )

        # Evaluate the generated strategy
        evaluation_result = await evaluate_trading_strategy(
            llm=self.llm,
            strategy=generated_strategy['response'],
            historical_data=historical_data
        )

        return evaluation_result['response'], self.llm.cost_manager.total_cost
