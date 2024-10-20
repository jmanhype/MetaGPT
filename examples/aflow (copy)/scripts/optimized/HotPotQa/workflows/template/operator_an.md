# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi (updated for Alternate History Generation)
# @Desc    : action nodes for operator, including new nodes for Alternate History Generation

from pydantic import BaseModel, Field
from typing import List, Dict

class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")

# New action nodes for Alternate History Generation

class HistoricalFactExtractorOp(BaseModel):
    facts: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted historical facts categorized by topic")

class AlternateScenarioGeneratorOp(BaseModel):
    alternate_scenario: str = Field(default="", description="Generated alternate historical scenario")

class PlausibilityCheckerOp(BaseModel):
    plausibility_score: float = Field(..., description="Plausibility score of the alternate scenario (0-1)")
    reasoning: str = Field(default="", description="Reasoning behind the plausibility score")

class NarrativeCoherenceEnhancerOp(BaseModel):
    enhanced_scenario: str = Field(default="", description="Enhanced version of the alternate scenario with improved narrative coherence")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made to improve coherence")

class HistoricalAccuracyVerifierOp(BaseModel):
    accuracy_score: float = Field(..., description="Historical accuracy score of the alternate scenario (0-1)")
    inaccuracies: List[str] = Field(default_factory=list, description="List of identified historical inaccuracies")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improving historical accuracy")