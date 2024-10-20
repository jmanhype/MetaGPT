# -*- coding: utf-8 -*-
# @Date    : 10/19/2024
# @Author  : Your Name
# @Desc    : Main script to run the optimization process for HotpotQA Alternate History Generator

import asyncio
import json
import logging
from typing import Dict, Any, List
import os
from datetime import datetime
import sys
from types import SimpleNamespace
from metagpt.configs.llm_config import LLMConfig

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from metagpt.provider.openai_api import OpenAILLM
from scripts.optimized.HotPotQa.recursive_optimization import recursive_optimization
from scripts.optimizer import Optimizer
from scripts.config_utils import load_config
from metagpt.provider.llm_provider_registry import LLMProviderRegistry
from metagpt.provider.llm_provider_registry import create_llm_instance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register the OpenAI provider
LLMProviderRegistry().register("openai", OpenAILLM)

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset from a JSONL file.
    
    Args:
        file_path (str): Path to the HotpotQA JSONL file.
    
    Returns:
        List[Dict[str, Any]]: Loaded dataset.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        logger.info(f"Successfully loaded {len(data)} questions from {file_path}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise
    return data

def save_results(results: Dict[str, Any], file_path: str):
    """
    Save results to a JSON file.
    
    Args:
        results (Dict[str, Any]): Results to save.
        file_path (str): Path to save the results.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {str(e)}")

async def main():
    try:
        # Load configuration
        config = load_config("/home/batmanosama/poc-kagnar/experiments/dslmodel-prefect/MetaGPT-MathAI/examples/aflow/config/hotpotqa_config.yaml")
        
        initial_params: Dict[str, Any] = config.get("initial_params", {
            "creativity": 0.7,
            "historical_accuracy": 0.8,
            "narrative_coherence": 0.6,
            # Add other parameters as needed
        })
        
        # Load HotpotQA dataset
        hotpotqa_data = load_data(config["data_path"])
        
        # Prepare llm_config using LLMConfig
        llm_config = LLMConfig(**config["llm_config"])
        
        # Create LLM instance
        llm = create_llm_instance(llm_config)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/alternate_history_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run optimization
        best_strategy, best_performance = await recursive_optimization(
            initial_params,
            hotpotqa_data,
            llm,
            max_iterations=config.get("max_iterations", 45),
            convergence_threshold=config.get("convergence_threshold", 0.01)
        )
        
        if best_strategy is not None:
            logger.info("Optimization completed successfully.")
            logger.info(f"Best Performance Score: {best_performance}")
            
            # Save the best strategy
            best_strategy_path = os.path.join(output_dir, "best_alternate_history_strategy.json")
            save_results({"strategy": best_strategy, "performance": best_performance}, best_strategy_path)
            
            # Optionally, run final evaluation using the Optimizer
            optimizer = Optimizer(config.get("eval_path", "path/to/eval"))
            final_score = await optimizer.graph_evaluate(
                dataset="HotpotQA",
                graph=best_strategy,  # You might need to adjust this based on how your Optimizer expects the input
                params={"llm_config": llm_config},
                path=output_dir
            )
            logger.info(f"Final Evaluation Score: {final_score}")
            
            # Save final results
            final_results = {
                "best_strategy": best_strategy,
                "best_performance": best_performance,
                "final_evaluation_score": final_score
            }
            save_results(final_results, os.path.join(output_dir, "final_results.json"))
        else:
            logger.error("Optimization failed to produce a valid strategy.")
    
    except Exception as e:
        logger.exception(f"An error occurred during the optimization process: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
