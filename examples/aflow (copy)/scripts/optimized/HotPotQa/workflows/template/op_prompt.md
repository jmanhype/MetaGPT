# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi (updated for Alternate History Generation)
# @Desc    : Prompts for operators, including new prompts for Alternate History Generation

SC_ENSEMBLE_PROMPT = """
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

# New prompts for Alternate History Generation

HISTORICAL_FACT_EXTRACTION_PROMPT = """
Analyze the following input and extract key historical facts. Categorize these facts by relevant topics (e.g., people, events, dates, locations).

Input: {input}

In your response:
1. List each extracted fact under its relevant category.
2. Ensure that each fact is concise, accurate, and directly related to the historical context provided in the input.
3. If a fact could belong to multiple categories, choose the most relevant one.

Provide your response in a structured format, with each category as a key and a list of facts as its value.
"""

ALTERNATE_SCENARIO_GENERATION_PROMPT = """
Based on the following historical facts, generate a plausible alternate historical scenario. Your scenario should diverge from actual history in a significant but believable way.

Historical Facts:
{facts}

In your response:
1. Identify a key point of divergence from the actual historical events.
2. Describe how this change would alter subsequent events.
3. Ensure that your alternate scenario maintains internal consistency and plausibility given the historical context.
4. Provide a brief (2-3 sentences) summary of the alternate scenario.
5. Then, elaborate on the scenario in 2-3 paragraphs, detailing the most significant changes and their immediate consequences.

Your alternate scenario should be creative yet grounded in the provided historical facts.
"""

PLAUSIBILITY_CHECK_PROMPT = """
Evaluate the plausibility of the following alternate historical scenario. Consider factors such as historical context, human behavior patterns, and logical consistency.

Alternate Scenario:
{scenario}

In your response:
1. Provide a plausibility score between 0 (completely implausible) and 1 (highly plausible).
2. In the "reasoning" field, explain your rationale for the given score. Consider:
   - How well the scenario aligns with known historical patterns and human behavior.
   - The logical consistency of the alternate events.
   - Any potential issues or inconsistencies in the scenario.

Be critical in your analysis, pointing out both strengths and weaknesses in the scenario's plausibility.
"""

NARRATIVE_COHERENCE_ENHANCEMENT_PROMPT = """
Enhance the narrative coherence of the following alternate historical scenario. Focus on improving the flow, clarity, and internal consistency of the narrative.

Original Scenario:
{scenario}

In your response:
1. Provide an enhanced version of the scenario in the "enhanced_scenario" field.
2. In the "changes_made" field, list the key changes you made to improve coherence.

When enhancing the scenario:
- Ensure a clear and logical progression of events.
- Maintain consistency in character motivations and actions.
- Clarify cause-and-effect relationships between events.
- Improve the overall narrative structure and flow.
- Preserve the core alternate history concept while enhancing its presentation.
"""

HISTORICAL_ACCURACY_VERIFICATION_PROMPT = """
Verify the historical accuracy of the following alternate scenario against the original historical facts. Identify any inaccuracies or inconsistencies while acknowledging the intentional divergence point of the alternate history.

Alternate Scenario:
{scenario}

Original Historical Facts:
{original_facts}

In your response:
1. Provide an accuracy score between 0 (completely inaccurate) and 1 (highly accurate, considering the intentional divergence).
2. In the "inaccuracies" field, list any historical inaccuracies or inconsistencies found in the alternate scenario.
3. In the "suggestions" field, provide recommendations for improving historical accuracy while maintaining the alternate history concept.

Consider:
- Accuracy of historical details prior to the point of divergence.
- Plausibility of changes after the divergence point, given the historical context.
- Consistency with known historical figures, technologies, and social conditions of the time.
"""