# LLM_Code_Summarization_Assessment_data

A script that includes the raw data used for evaluation, as well as the output of the model based on the data and the evaluation results of the reference text. The evaluation script input must include at least two fields: reference text and generated text

Due to the uncertainty of the output of large language models, even if parameters such as temperature are set to 0, the output results may vary slightly each time. The output results are for reference only. 

The metric_out.py script is used to compute automated evaluation metrics, including sacrebleu_bleu_4, meteor, rouge_l, llm_as_judge, and Semantic Similarity. The latter two metrics correspond to the functions set_llm_as_judge and generate_embedding, respectively. 

The ANOVA folder contains significance test results and corresponding scripts. The subfolder model-qwen holds the significance results and post‑hoc pairwise comparisons for different prompt templates under the Qwen model. Jsoncpp contains the test results for the incomplete repository used as a negative example. prompt8 contains the significance results and post‑hoc pairwise comparisons for different large language models under Template 8. Files prefixed with repo- contain significance tests and post‑hoc pairwise comparisons for different repositories at the corresponding granularity. The ANOVA folder also includes the evaluation scripts. 

The human-assess folder contains the human evaluation results. The script questionaire-analyse.py is used to test inter‑rater agreement and the correlation between human and automated metrics. The two .xlsx files contain the corresponding results.(Waiting for adding...)

