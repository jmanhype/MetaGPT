# -*- coding: utf-8 -*-
# @Date    : 10/19/2024
# @Author  : Your Name
# @Desc    : Recursive Optimization for HotpotQA Alternate History Generation

import asyncio
import json
import random
from typing import Dict, Any, Tuple, List
import pandas as pd
import logging
from tqdm import tqdm
from scripts.optimized.HotPotQa.workflows.round_1.graph import HotpotQAWorkflow
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def adjust_parameters(current_params: Dict[str, Any], performance: float) -> Dict[str, Any]:
    """
    Adjust parameters based on current performance.
    This is a placeholder function - implement your own logic here.
    
    Args:
        current_params (Dict[str, Any]): Current parameters.
        performance (float): Current performance score.
    
    Returns:
        Dict[str, Any]: Adjusted parameters.
    """
    # Example adjustment: Increase 'creativity' if performance is low
    if performance < 0.5:
        current_params['creativity'] = min(1.0, current_params.get('creativity', 0.5) + 0.1)
    
    # Add more parameter adjustments based on your specific needs
    
    return current_params

async def recursive_optimization(
    initial_params: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    llm: Any,
    max_iterations: int = 10,
    convergence_threshold: float = 0.01
) -> Tuple[Dict[str, Any], float]:
    """
    Perform recursive optimization to improve alternate history scenarios.
    
    Args:
        initial_params (Dict[str, Any]): Initial parameters for strategy generation.
        dataset (List[Dict[str, Any]]): Loaded HotpotQA dataset.
        llm_config (dict): Configuration for the language model.
        max_iterations (int): Maximum number of optimization iterations.
        convergence_threshold (float): Threshold for early stopping if improvement is minimal.
    
    Returns:
        Tuple[Dict[str, Any], float]: Best strategy and its performance score.
    """
    workflow = HotpotQAWorkflow("HotpotQA_AlternateHistory", llm, "HotpotQA")
    best_strategy = None
    best_performance = float('-inf')
    current_params = initial_params.copy()
    
    performance_history = []
    
    try:
        for iteration in range(max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
            current_score = 0
            strategies = []
            
            for item in tqdm(dataset, desc=f"Processing items in iteration {iteration + 1}"):
                original_question = item.get("question", "")
                try:
                    evaluation, cost = await workflow.execute_alternate_history_workflow(original_question)
                    strategies.append({
                        "question": original_question,
                        "alternate_scenario": evaluation
                    })
                    # Aggregate scores (customize as needed)
                    current_score += (evaluation["plausibility"] + evaluation["coherence"] + evaluation["accuracy"]) / 3
                except Exception as e:
                    logger.error(f"Error processing question: {original_question}. Error: {str(e)}")
                    continue
            
            average_score = current_score / len(dataset)
            logger.info(f"Iteration {iteration + 1}: Average Score = {average_score}")
            performance_history.append(average_score)
            
            if average_score > best_performance:
                improvement = average_score - best_performance
                best_performance = average_score
                best_strategy = strategies
                logger.info(f"New best strategy found with score {best_performance}")
                
                # Save intermediate best result
                with open(f"best_alternate_history_strategy_iteration_{iteration + 1}.json", "w") as f:
                    json.dump(best_strategy, f, indent=2)
                
                if improvement < convergence_threshold:
                    logger.info(f"Improvement below threshold. Stopping optimization.")
                    break
            
            # Adjust parameters based on performance
            current_params = await adjust_parameters(current_params, average_score)
            logger.info(f"Adjusted parameters: {current_params}")
        
        # Save the final best strategy
        with open("best_alternate_history_strategy_final.json", "w") as f:
            json.dump(best_strategy, f, indent=2)
        
        # Save performance history
        pd.DataFrame({"iteration": range(1, len(performance_history) + 1), "score": performance_history}).to_csv("optimization_performance_history.csv", index=False)
        
        return best_strategy, best_performance
    
    except Exception as e:
        logger.error(f"An error occurred during optimization: {str(e)}")
        return None, float('-inf')

async def main():
    # Example usage
    initial_params = {
        "creativity": 0.5,
        "historical_accuracy": 0.7,
        # Add more parameters as needed
    }
    
    llm_config = {
        "model": "gpt-4-turbo",
        "api_key": "your-api-key-here"
    }
    
    # Load your dataset here
    dataset = [
        {"question": "What event led to the start of World War I?"},
        {"question": "Who was the first president of the United States?"},
        # Add more questions...
    ]
    
    best_strategy, best_performance = await recursive_optimization(initial_params, dataset, llm_config)
    
    if best_strategy:
        logger.info(f"Optimization completed. Best performance: {best_performance}")
    else:
        logger.error("Optimization failed.")

if __name__ == "__main__":
    asyncio.run(main())
