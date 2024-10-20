from typing import List
from examples.aflow.scripts.optimizer import Optimizer


def main():
    # Define LLM configurations for optimization and execution
    opt_llm_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1500,
        # Add other necessary configurations
    }
    exec_llm_config = {
        "model": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 1000,
        # Add other necessary configurations
    }

    # Define the operators to be used
    operators: List[str] = [
        "HistoricalFactExtractor",
        "AlternateScenarioGenerator",
        "PlausibilityChecker",
        "NarrativeCoherenceEnhancer",
        "HistoricalAccuracyVerifier"
    ]

    # Initialize the Optimizer
    optimizer = Optimizer(
        dataset="HotpotQA",
        question_type="qa",
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        operators=operators,
        sample=35,
        check_convergence=False,
        optimized_path="/home/writer/aflow/optimized/alternate_history_generators",
        initial_round=1,
        max_rounds=45
    )

    # Start the optimization process
    optimizer.optimize()


if __name__ == "__main__":
    main()
