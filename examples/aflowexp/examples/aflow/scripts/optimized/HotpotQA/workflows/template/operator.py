from metagpt.actions.action_node import ActionNode
from metagpt.llm import LLM
from typing import Any, Dict

async def historical_fact_extractor(llm: LLM, input_data: str, instruction: str) -> Dict[str, Any]:
    """
    Extracts historical facts related to the input question.
    
    Args:
        llm (LLM): Language model instance.
        input_data (str): The original HotpotQA question.
        instruction (str): Instructions for extracting facts.
    
    Returns:
        Dict[str, Any]: Extracted historical facts.
    """
    prompt = instruction + input_data
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response

async def alternate_scenario_generator(llm: LLM, facts: str, instruction: str) -> Dict[str, Any]:
    """
    Generates alternate historical scenarios based on extracted facts.
    
    Args:
        llm (LLM): Language model instance.
        facts (str): Extracted historical facts.
        instruction (str): Instructions for generating scenarios.
    
    Returns:
        Dict[str, Any]: Generated alternate scenarios.
    """
    prompt = instruction + facts
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response

async def plausibility_checker(llm: LLM, scenario: str, instruction: str) -> Dict[str, Any]:
    """
    Checks the plausibility of the generated alternate scenarios.
    
    Args:
        llm (LLM): Language model instance.
        scenario (str): Generated alternate scenario.
        instruction (str): Instructions for plausibility checking.
    
    Returns:
        Dict[str, Any]: Plausibility assessment.
    """
    prompt = instruction + scenario
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response

async def narrative_coherence_enhancer(llm: LLM, scenario: str, instruction: str) -> Dict[str, Any]:
    """
    Enhances the narrative coherence of the generated scenarios.
    
    Args:
        llm (LLM): Language model instance.
        scenario (str): Generated alternate scenario.
        instruction (str): Instructions for enhancing coherence.
    
    Returns:
        Dict[str, Any]: Enhanced narrative.
    """
    prompt = instruction + scenario
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response

async def historical_accuracy_verifier(llm: LLM, scenario: str, instruction: str) -> Dict[str, Any]:
    """
    Verifies the historical accuracy of the generated scenarios.
    
    Args:
        llm (LLM): Language model instance.
        scenario (str): Generated alternate scenario.
        instruction (str): Instructions for accuracy verification.
    
    Returns:
        Dict[str, Any]: Accuracy verification results.
    """
    prompt = instruction + scenario
    node = await ActionNode.from_pydantic(GenerateOp).fill(
        context=prompt, llm=llm, mode="single_fill"
    )
    response = node.instruct_content.model_dump()
    return response
