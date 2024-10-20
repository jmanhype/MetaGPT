import asyncio
from typing import List, Tuple, Callable, Any
from tqdm.asyncio import tqdm_asyncio
import csv


async def evaluate_trading_strategy(
    strategy: Dict[str, Any],
    graph: Callable[[Dict[str, Any], str], Any],
    path: str,
    max_concurrent_tasks: int = 50
) -> Tuple[float, float, float]:
    """
    Evaluate a single trading strategy.

    Args:
        strategy (Dict[str, Any]): The trading strategy to evaluate.
        graph (Callable): The graph function to execute the strategy.
        path (str): Path to save evaluation results.
        max_concurrent_tasks (int): Maximum number of concurrent evaluation tasks.

    Returns:
        Tuple[float, float, float]: Average score, average cost, total cost.
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def sem_evaluate(strat: Dict[str, Any]) -> Tuple[float, float, float]:
        async with semaphore:
            return await graph(strat, path)

    result = await sem_evaluate(strategy)
    return result


async def evaluate_all_trading_strategies(
    data: List[Dict[str, Any]],
    graph: Callable[[Dict[str, Any], str], Any],
    path: str,
    max_concurrent_tasks: int = 50
) -> List[Tuple[float, float, float]]:
    """
    Evaluate all trading strategies concurrently.

    Args:
        data (List[Dict[str, Any]]): List of trading strategies.
        graph (Callable): The graph function to execute each strategy.
        path (str): Path to save evaluation results.
        max_concurrent_tasks (int): Maximum number of concurrent evaluation tasks.

    Returns:
        List[Tuple[float, float, float]]: List of evaluation results for each strategy.
    """
    tasks = [
        evaluate_trading_strategy(strategy, graph, path, max_concurrent_tasks)
        for strategy in data
    ]
    return await tqdm_asyncio.gather(
        *tasks,
        desc="Evaluating Trading Strategies",
        total=len(data)
    )


async def save_trading_evaluation_results(
    results: List[Tuple[float, float, float]],
    path: str
) -> Tuple[float, float, float]:
    """
    Save evaluation results to a CSV file and calculate averages.

    Args:
        results (List[Tuple[float, float, float]]): Evaluation results.
        path (str): Path to save the CSV file.

    Returns:
        Tuple[float, float, float]: Average score, average cost, total cost.
    """
    with open(f"{path}/trading_evaluation_results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Average Score", "Average Cost", "Total Cost"])
        writer.writerows(results)

    average_score = sum(r[0] for r in results) / len(results) if results else 0
    average_cost = sum(r[1] for r in results) / len(results) if results else 0
    total_cost = sum(r[2] for r in results) if results else 0

    return average_score, average_cost, total_cost
