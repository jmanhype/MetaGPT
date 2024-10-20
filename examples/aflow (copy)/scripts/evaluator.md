# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all (updated for HotpotQA Alternate History Generation)
# @Desc    : Evaluation for different datasets, including HotpotQA Alternate History Generation

import json
import pandas as pd
from typing import List, Dict, Any, Literal, Tuple, Optional
import asyncio

from examples.aflow.benchmark.gsm8k import optimize_gsm8k_evaluation
from examples.aflow.benchmark.math import optimize_math_evaluation
from examples.aflow.benchmark.humaneval import optimize_humaneval_evaluation
from examples.aflow.benchmark.hotpotqa import optimize_hotpotqa_evaluation
from examples.aflow.benchmark.mbpp import optimize_mbpp_evaluation
from examples.aflow.benchmark.drop import optimize_drop_evaluation

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "HotpotQAAlternateHistory"]

class Evaluator:
    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs = {
            "GSM8K": {"name": "GSM8K", "eval_func": optimize_gsm8k_evaluation},
            "MATH": {"name": "MATH", "eval_func": optimize_math_evaluation},
            "HumanEval": {"name": "HumanEval", "eval_func": optimize_humaneval_evaluation},
            "HotpotQA": {"name": "HotpotQA", "eval_func": optimize_hotpotqa_evaluation},
            "MBPP": {"name": "MBPP", "eval_func": optimize_mbpp_evaluation},
            "DROP": {"name": "DROP", "eval_func": optimize_drop_evaluation},
            "HotpotQAAlternateHistory": {"name": "HotpotQAAlternateHistory", "eval_func": self.optimize_hotpotqa_alternate_history_evaluation},
        }

    def graph_evaluate(self, dataset: DatasetType, graph, params: dict, path, is_test=False):
        if dataset in self.dataset_configs:
            return self._generic_eval(dataset, graph, params, path, is_test)
        else:
            return None

    async def _generic_eval(self, dataset: DatasetType, graph_class, params: dict, path: str, test: bool = False) -> Tuple[float, float, float]:
        async def load_graph():
            dataset_config = params["dataset"]
            llm_config = params["llm_config"]
            return graph_class(name=self.dataset_configs[dataset]["name"], llm_config=llm_config, dataset=dataset_config)

        data_path, va_list = self._get_data_path_and_va_list(dataset, test)
        graph = await load_graph()
        
        eval_func = self.dataset_configs[dataset]["eval_func"]
        avg_score, avg_cost, total_cost = await eval_func(graph, data_path, path, va_list)
        
        return avg_score, avg_cost, total_cost

    def _get_data_path_and_va_list(self, dataset: DatasetType, test: bool) -> Tuple[str, Optional[list]]:
        base_path = f"examples/aflow/data/{dataset.lower()}"
        if test:
            return f"{base_path}_test.jsonl", None
        else:
            return f"{base_path}_validate.jsonl", [1, 2, 3]  # Replace with actual filtered index list

    # HotpotQA Alternate History specific functions
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def evaluate_problem(self, original_question: str, alternate_scenario: str) -> Dict[str, Any]:
        # Implement evaluation logic, possibly using automated metrics or manual assessment
        evaluation = {
            "plausibility": 0.0,  # Placeholder value
            "coherence": 0.0,     # Placeholder value
            "accuracy": 0.0       # Placeholder value
        }
        # TODO: Add actual evaluation code
        return evaluation

    def evaluate_all_problems(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in data:
            original_question = item.get("question", "")
            alternate_scenario = item.get("alternate_scenario", "")
            evaluation = self.evaluate_problem(original_question, alternate_scenario)
            results.append({
                "question": original_question,
                "alternate_scenario": alternate_scenario,
                **evaluation
            })
        return results

    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

    def calculate_overall_score(self, results: List[Dict[str, Any]]) -> float:
        total_plausibility = sum(item["plausibility"] for item in results)
        total_coherence = sum(item["coherence"] for item in results)
        total_accuracy = sum(item["accuracy"] for item in results)
        count = len(results)
        overall_score = (total_plausibility + total_coherence + total_accuracy) / (count * 3)
        return overall_score

    async def optimize_hotpotqa_alternate_history_evaluation(self, workflow, data_path: str, path: str, va_list: Optional[List[int]]) -> Tuple[float, float, float]:
        data = self.load_data(data_path)
        if va_list:
            data = [data[i] for i in va_list]
        
        original_questions = [item["question"] for item in data]
        alternate_scenarios = await workflow.generate_scenarios(original_questions)
        
        evaluation_results = self.evaluate_all_problems(alternate_scenarios)
        self.save_results_to_csv(evaluation_results, f"{path}/results.csv")
        
        overall_score = self.calculate_overall_score(evaluation_results)
        
        # Placeholder values for avg_cost and total_cost
        avg_cost, total_cost = 0.0, 0.0
        # TODO: Implement cost calculation if needed
        
        return overall_score, avg_cost, total_cost

# Alias methods for backward compatibility
for dataset in ["gsm8k", "math", "humaneval", "mbpp", "hotpotqa", "drop"]:
    setattr(Evaluator, f"_{dataset}_eval", lambda self, *args, dataset=dataset.upper(), **kwargs: self._generic_eval(dataset, *args, **kwargs))

# Special method for HotpotQA Alternate History
setattr(Evaluator, "_hotpotqa_alternate_history_eval", lambda self, *args, **kwargs: self._generic_eval("HotpotQAAlternateHistory", *args, **kwargs))

# Example usage
if __name__ == "__main__":
    evaluator = Evaluator("/path/to/eval")
    
    # Example params (you would need to fill these with actual values)
    params = {
        "dataset": {"some_config": "value"},
        "llm_config": {"model": "gpt-4-turbo", "api_key": "your-api-key"}
    }
    
    # Evaluate HotpotQA Alternate History Generation
    asyncio.run(evaluator._hotpotqa_alternate_history_eval(SomeGraphClass, params, "/path/to/output", test=False))