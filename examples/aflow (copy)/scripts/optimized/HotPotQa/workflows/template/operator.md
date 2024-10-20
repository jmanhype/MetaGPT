# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of ags
import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Union, Any

from tenacity import retry, stop_after_attempt, wait_fixed

from .operator_an import *
from .op_prompt import *
from metagpt.actions.action_node import ActionNode, CustomJSONEncoder
from metagpt.llm import LLM
from metagpt.logs import logger
import re
import json


class Operator:
    def __init__(self, name, llm: LLM):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    # Enhanced _ensure_serializable method to handle PydanticUndefinedType
    def _ensure_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._ensure_serializable(obj.__dict__)
        elif obj.__class__.__name__ == 'PydanticUndefinedType':
            return None  # Convert PydanticUndefinedType to None
        else:
            return str(obj)


class Custom(Operator):
    def __init__(self, llm: LLM, name: str = "Custom"):
        super().__init__(name, llm)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        node = await ActionNode.from_pydantic(GenerateOp).fill(context=prompt, llm=self.llm, mode="single_fill")
        response = node.instruct_content.model_dump()
        return response
    
class AnswerGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "AnswerGenerate"):
        super().__init__(name, llm)

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        fill_kwargs = {"context": prompt, "llm": self.llm}
        node = await ActionNode.from_pydantic(AnswerGenerateOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()
        return response

class ScEnsemble(Operator):
    def __init__(self,llm: LLM , name: str = "ScEnsemble"):
        super().__init__(name, llm)

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        node = await ActionNode.from_pydantic(ScEnsembleOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}

# New operators for Alternate History Generation

class HistoricalFactExtractor(Operator):
    def __init__(self, llm: LLM, name: str = "HistoricalFactExtractor"):
        super().__init__(name, llm)

    async def __call__(self, input: str, instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(input=input)
        node = await ActionNode.from_pydantic(HistoricalFactExtractorOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = json.loads(json.dumps(response, cls=CustomJSONEncoder))
        if not isinstance(serialized_response, dict) or 'facts' not in serialized_response:
            return {"facts": {"default": [str(serialized_response)]}}
        return serialized_response

class AlternateScenarioGenerator(Operator):
    def __init__(self, llm: LLM, name: str = "AlternateScenarioGenerator"):
        super().__init__(name, llm)

    async def __call__(self, facts: Dict[str, List[str]], instruction: str) -> Dict[str, str]:
        prompt = instruction.format(facts=facts)
        node = await ActionNode.from_pydantic(AlternateScenarioGeneratorOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {"alternate_scenario": serialized_response.get("alternate_scenario", "")}

class PlausibilityChecker(Operator):
    def __init__(self, llm: LLM, name: str = "PlausibilityChecker"):
        super().__init__(name, llm)

    async def __call__(self, scenario: str, original_question: str, instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(scenario=scenario, original_question=original_question)
        node = await ActionNode.from_pydantic(PlausibilityCheckerOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "plausibility_score": float(serialized_response.get("plausibility_score", 0.0)),
            "reasoning": serialized_response.get("reasoning", "")
        }

class NarrativeCoherenceEnhancer(Operator):
    def __init__(self, llm: LLM, name: str = "NarrativeCoherenceEnhancer"):
        super().__init__(name, llm)

    async def __call__(self, scenario: str, original_question: str, instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(scenario=scenario, original_question=original_question)
        node = await ActionNode.from_pydantic(NarrativeCoherenceEnhancerOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "enhanced_scenario": serialized_response.get("enhanced_scenario", ""),
            "changes_made": serialized_response.get("changes_made", [])
        }

class HistoricalAccuracyVerifier(Operator):
    def __init__(self, llm: LLM, name: str = "HistoricalAccuracyVerifier"):
        super().__init__(name, llm)

    async def __call__(self, scenario: str, original_question: str, original_facts: Dict[str, List[str]], instruction: str) -> Dict[str, Any]:
        prompt = instruction.format(scenario=scenario, original_question=original_question, original_facts=original_facts)
        node = await ActionNode.from_pydantic(HistoricalAccuracyVerifierOp).fill(context=prompt, llm=self.llm)
        response = node.instruct_content.model_dump()
        serialized_response = self._ensure_serializable(response)
        return {
            "accuracy_score": float(serialized_response.get("accuracy_score", 0.0)),
            "inaccuracies": serialized_response.get("inaccuracies", []),
            "suggestions": serialized_response.get("suggestions", [])
        }

# Factory functions to create operator instances

def get_historical_fact_extractor(llm: LLM) -> HistoricalFactExtractor:
    return HistoricalFactExtractor(llm)

def get_alternate_scenario_generator(llm: LLM) -> AlternateScenarioGenerator:
    return AlternateScenarioGenerator(llm)

def get_plausibility_checker(llm: LLM) -> PlausibilityChecker:
    return PlausibilityChecker(llm)

def get_narrative_coherence_enhancer(llm: LLM) -> NarrativeCoherenceEnhancer:
    return NarrativeCoherenceEnhancer(llm)

def get_historical_accuracy_verifier(llm: LLM) -> HistoricalAccuracyVerifier:
    return HistoricalAccuracyVerifier(llm)

async def historical_fact_extractor(self, input: str, instruction: str) -> Dict[str, Any]:
    prompt = f"{instruction}\n\nInput: {input}"
    node = await ActionNode.from_pydantic(HistoricalFactExtractorOp).fill(context=prompt, llm=self.llm)
    response = node.content
    if isinstance(response, dict) and 'response' in response:
        return response
    else:
        return {"response": str(response)}

