import json

def apply_alternate_history_strategy(strategy_path: str, original_question: str):
    """
    Apply the optimized alternate history strategy to generate narratives.
    
    Args:
        strategy_path (str): Path to the JSON file containing the alternate history scenarios.
        original_question (str): The original HotpotQA question.
    
    Returns:
        None
    """
    with open(strategy_path, 'r') as f:
        strategies = json.load(f)

    for item in strategies:
        question = item["question"]
        evaluation = item["alternate_scenario"]
        print(f"Original Question: {question}")
        print(f"Alternate Scenario Plausibility: {evaluation['plausibility']}")
        print(f"Alternate Scenario Coherence: {evaluation['coherence']}")
        print(f"Alternate Scenario Accuracy: {evaluation['accuracy']}")
        print("-" * 50)

# Usage
if __name__ == "__main__":
    apply_alternate_history_strategy("best_alternate_history_strategy.json", "Sample HotpotQA Question")
