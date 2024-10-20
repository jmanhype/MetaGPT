import asyncio
import json
import random
from typing import Dict, Any, Tuple, List
import pandas as pd
from examples.aflow.scripts.optimized.HotpotQA.workflows.round_1.graph import HotpotQAWorkflow
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

async def recursive_optimization(
    initial_params: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    llm_config: dict,
    max_iterations: int = 10
) -> Tuple[Dict[str, Any], float]:
    """
    Perform recursive optimization to improve alternate history scenarios.
    
    Args:
        initial_params (Dict[str, Any]): Initial parameters for strategy generation.
        dataset (List[Dict[str, Any]]): Loaded HotpotQA dataset.
        llm_config (dict): Configuration for the language model.
        max_iterations (int): Maximum number of optimization iterations.
    
    Returns:
        Tuple[Dict[str, Any], float]: Best strategy and its performance score.
    """
    workflow = HotpotQAWorkflow("HotpotQA_AlternateHistory", llm_config, "HotpotQA")
    best_strategy = None
    best_performance = float('-inf')

    for iteration in range(max_iterations):
        print(f"Starting iteration {iteration + 1}/{max_iterations}")
        current_score = 0
        strategies = []

        for item in dataset:
            original_question = item.get("question", "")
            evaluation, cost = await workflow.execute_alternate_history_workflow(original_question)
            strategies.append({
                "question": original_question,
                "alternate_scenario": evaluation
            })
            # Example: Aggregate scores (customize as needed)
            current_score += (evaluation["plausibility"] + evaluation["coherence"] + evaluation["accuracy"]) / 3

        average_score = current_score / len(dataset)
        print(f"Iteration {iteration + 1}: Average Score = {average_score}")

        if average_score > best_performance:
            best_performance = average_score
            best_strategy = strategies
            print(f"New best strategy found with score {best_performance}")

        # Optionally, implement logic to adjust parameters based on performance

    # Save the best strategy to a file
    with open("best_alternate_history_strategy.json", "w") as f:
        json.dump(best_strategy, f, indent=2)

    return best_strategy, best_performance
