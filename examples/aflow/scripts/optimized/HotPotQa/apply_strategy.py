# -*- coding: utf-8 -*-
# @Date    : 10/19/2024
# @Author  : Your Name
# @Desc    : Script to apply and review optimized alternate history scenarios

import json
import argparse
import logging
from typing import List, Dict, Any
from examples.aflow.scripts.optimized.HotpotQA.workflows.round_1.graph import HotpotQAWorkflow
from metagpt.provider.llm_provider_registry import create_llm_instance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_strategy(strategy_path: str) -> List[Dict[str, Any]]:
    """
    Load the optimized alternate history strategy from a JSON file.
    
    Args:
        strategy_path (str): Path to the JSON file containing the alternate history scenarios.
    
    Returns:
        List[Dict[str, Any]]: Loaded strategy data.
    """
    try:
        with open(strategy_path, 'r') as f:
            strategies = json.load(f)
        logger.info(f"Successfully loaded strategy from {strategy_path}")
        return strategies
    except Exception as e:
        logger.error(f"Error loading strategy from {strategy_path}: {str(e)}")
        raise

def print_scenario(question: str, evaluation: Dict[str, Any]):
    """
    Print the details of an alternate history scenario.
    
    Args:
        question (str): The original HotpotQA question.
        evaluation (Dict[str, Any]): Evaluation results for the alternate scenario.
    """
    print(f"Original Question: {question}")
    print(f"Alternate Scenario Plausibility: {evaluation['plausibility']:.2f}")
    print(f"Alternate Scenario Coherence: {evaluation['coherence']:.2f}")
    print(f"Alternate Scenario Accuracy: {evaluation['accuracy']:.2f}")
    if 'scenario_text' in evaluation:
        print(f"Alternate Scenario:\n{evaluation['scenario_text']}")
    print("-" * 50)

async def generate_new_scenario(workflow: HotpotQAWorkflow, question: str) -> Dict[str, Any]:
    """
    Generate a new alternate history scenario using the optimized workflow.
    
    Args:
        workflow (HotpotQAWorkflow): The optimized HotpotQA workflow.
        question (str): The original HotpotQA question.
    
    Returns:
        Dict[str, Any]: The generated alternate scenario and its evaluation.
    """
    try:
        evaluation, _ = await workflow.execute_alternate_history_workflow(question)
        return evaluation
    except Exception as e:
        logger.error(f"Error generating new scenario for question '{question}': {str(e)}")
        return {}

async def apply_alternate_history_strategy(strategy_path: str, questions: List[str] = None, generate_new: bool = False):
    """
    Apply the optimized alternate history strategy to review or generate narratives.
    
    Args:
        strategy_path (str): Path to the JSON file containing the alternate history scenarios.
        questions (List[str], optional): List of specific questions to process. If None, process all in the strategy.
        generate_new (bool): If True, generate new scenarios using the optimized workflow.
    """
    strategies = load_strategy(strategy_path)
    
    if generate_new:
        # Initialize the workflow for generating new scenarios
        llm_config = {
            "model": "gpt-4-turbo",
            "api_key": "your-api-key-here"  # Replace with actual API key or load from config
        }
        workflow = HotpotQAWorkflow("HotpotQA_AlternateHistory", llm_config, "HotpotQA")
    
    processed_questions = set()
    
    for item in strategies:
        question = item["question"]
        if questions and question not in questions:
            continue
        
        processed_questions.add(question)
        
        if generate_new:
            evaluation = await generate_new_scenario(workflow, question)
            if evaluation:
                print_scenario(question, evaluation)
            else:
                print(f"Failed to generate new scenario for: {question}")
        else:
            evaluation = item["alternate_scenario"]
            print_scenario(question, evaluation)
    
    # Process any remaining questions not found in the strategy
    if questions:
        remaining_questions = set(questions) - processed_questions
        if remaining_questions and generate_new:
            for question in remaining_questions:
                evaluation = await generate_new_scenario(workflow, question)
                if evaluation:
                    print_scenario(question, evaluation)
                else:
                    print(f"Failed to generate new scenario for: {question}")
        elif remaining_questions:
            logger.warning(f"The following questions were not found in the strategy: {remaining_questions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply and review optimized alternate history scenarios")
    parser.add_argument("strategy_path", help="Path to the JSON file containing the alternate history scenarios")
    parser.add_argument("--questions", nargs="*", help="Specific questions to process")
    parser.add_argument("--generate", action="store_true", help="Generate new scenarios using the optimized workflow")
    args = parser.parse_args()

    import asyncio
    asyncio.run(apply_alternate_history_strategy(args.strategy_path, args.questions, args.generate))