import { Workflow, DatasetType, Any, Tuple } from './types';
import { prompt_custom } from './prompt_custom';
import { get_historical_fact_extractor, get_alternate_scenario_generator, get_plausibility_checker, get_narrative_coherence_enhancer, get_historical_accuracy_verifier } from './operator';

class HotpotQAWorkflow(Workflow) {
    constructor(name: string, llm_config: dict, dataset: DatasetType) {
        super(name, llm_config, dataset);
        this.historical_fact_extractor = get_historical_fact_extractor(this.llm);
        this.alternate_scenario_generator = get_alternate_scenario_generator(this.llm);
        this.plausibility_checker = get_plausibility_checker(this.llm);
        this.narrative_coherence_enhancer = get_narrative_coherence_enhancer(this.llm);
        this.historical_accuracy_verifier = get_historical_accuracy_verifier(this.llm);
    }

    async execute_alternate_history_workflow(question_input: string): Promise<Tuple<Dict<string, Any>, float>> {
        // Step 1: Extract historical facts
        const extracted_facts = await this.historical_fact_extractor(
            input=question_input,
            instruction=prompt_custom.HISTORICAL_FACT_EXTRACTION_PROMPT
        );

        // Step 2: Generate alternate scenarios
        const generated_scenario = await this.alternate_scenario_generator(
            facts=extracted_facts['response'],
            instruction=prompt_custom.ALTERNATE_SCENARIO_GENERATION_PROMPT
        );

        // Step 3: Check plausibility
        const plausibility = await this.plausibility_checker(
            scenario=generated_scenario,
            original_question=question_input,
            instruction=prompt_custom.PLAUSIBILITY_CHECK_PROMPT
        );

        // Step 4: Enhance narrative coherence
        const coherent_scenario = await this.narrative_coherence_enhancer(
            scenario=generated_scenario,
            original_question=question_input,
            instruction=prompt_custom.NARRATIVE_COHERENCE_ENHANCEMENT_PROMPT
        );

        // Step 5: Verify historical accuracy
        const accuracy = await this.historical_accuracy_verifier(
            scenario=coherent_scenario,
            original_question=question_input,
            original_facts=extracted_facts['response'],
            instruction=prompt_custom.HISTORICAL_ACCURACY_VERIFICATION_PROMPT
        );

        const evaluation_results = {
            "plausibility": plausibility.get("plausibility_score", 0.0),
            "coherence": coherent_scenario,
            "accuracy": accuracy.get("accuracy_score", 0.0),
            "inaccuracies": accuracy.get("inaccuracies", [])
        };

        return evaluation_results, this.llm.cost_manager.total_cost;
    }
}
