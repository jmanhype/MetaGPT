from metagpt.actions.action_node import ActionNode
from metagpt.llm import LLM
from typing import Any, Dict


async def generate_trading_strategy(llm: LLM, input_data: str, instruction: str) -> Dict[str, Any]:
    """
    Generates a new trading strategy based on the input data and instruction.

    Args:
        llm (LLM): Language model instance.
        input_data (str): Current trading parameters or signal data.
        instruction (str): Instructions for generating the strategy.

    Returns:
        Dict[str, Any]: Generated strategy details.
    """
    prompt = instruction + input_data
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response


async def evaluate_trading_strategy(llm: LLM, strategy: str, historical_data: str) -> Dict[str, Any]:
    """
    Evaluates the performance of a trading strategy using historical data.

    Args:
        llm (LLM): Language model instance.
        strategy (str): The trading strategy to evaluate.
        historical_data (str): Path or reference to historical trading data.

    Returns:
        Dict[str, Any]: Evaluation results, such as return, risk metrics, etc.
    """
    prompt = (
        f"Evaluate the following trading strategy:\n\n{strategy}\n\n"
        f"Using historical data: {historical_data}"
    )
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response
