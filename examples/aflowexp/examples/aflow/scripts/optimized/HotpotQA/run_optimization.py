import asyncio
import json
from typing import Dict, Any
from examples.aflow.scripts.optimized.HotpotQA.recursive_optimization import recursive_optimization
from examples.aflow.scripts.optimizer import Optimizer  # Ensure correct import path

def main():
    initial_params: Dict[str, Any] = {
        # Define any initial parameters if necessary
        # For alternate history, this might include constraints or specific instructions
    }

    # Load HotpotQA dataset
    hotpotqa_data = load_data("path/to/hotpotqa_public_test.jsonl")  # Update the path accordingly

    llm_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1500,
        # Add other necessary configurations
    }

    best_strategy, best_performance = asyncio.run(
        recursive_optimization(initial_params, hotpotqa_data, llm_config, max_iterations=45)
    )

    print("Best Alternate History Strategy:")
    print(json.dumps(best_strategy, indent=2))
    print(f"Best Performance Score: {best_performance}")

    # Optionally, save the best strategy to a file
    with open("best_alternate_history_strategy.json", "w") as f:
        json.dump(best_strategy, f, indent=2)

if __name__ == "__main__":
    main()
