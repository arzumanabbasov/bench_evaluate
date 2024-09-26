import yaml
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from base import *
from qa_quality import *
from multiple_choice import *
from context_qa import *
import login

# Load the YAML file Metadata
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access metadata and dataset files
metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']

def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        ending = ending + '.xlsx'
        if filename.endswith(ending):
            return benchmark_type
    raise ValueError(f"Filename {filename} does not match any known benchmark type")

def handle_qa(question, actual_answer, model, tokenizer):
    predicted_answer = get_answer_from_local_huggingface_context(model, question, None, tokenizer)
    score = (
        0.25 * int(float(get_evaluation_score(question, predicted_answer, actual_answer))) + 
        calculate_bleu_score(actual_answer, predicted_answer) + 
        calculate_rouge_score(actual_answer, predicted_answer) + 
        calculate_levenshtein_score(actual_answer, predicted_answer)
    )
    return score

def handle_multiple_choice(question, options, correct_option, model, tokenizer):
    predicted_option = get_model_answer_multiple_options(question, options=options, model=model, dstype='mc', tokenizer=tokenizer)
    score = compare_answers(actual_answer=correct_option, predicted_answer=predicted_option)
    return score

def handle_context_qa(question, context, actual_answer, model, tokenizer):
    predicted_answer = get_answer_from_local_huggingface_context(model, question, context, tokenizer)
    score = (
        0.25 * int(float(get_evaluation_score_context(question, actual_answer, predicted_answer))) + 
        calculate_bleu_score(actual_answer, predicted_answer) + 
        calculate_rouge_score(actual_answer, predicted_answer) + 
        calculate_levenshtein_score(actual_answer, predicted_answer)
    )
    return score

def handle_topic_classification(question, topic_options, correct_topic, model, tokenizer):
    predicted_topic = get_model_answer_multiple_options(question=question, model=model, tokenizer=tokenizer, options=topic_options, dstype='tc')
    print(predicted_topic)
    score = compare_answers(actual_answer=correct_topic, predicted_answer=predicted_topic)
    return score

def run_benchmark(model_name, tokenizer, benchmark_type, df, results):
    scores = []
    if benchmark_type == "QA":
        for index, row in df.iterrows():
            question = row['Sual']
            actual_answer = row['Cavab']
            score = handle_qa(question, actual_answer, model_name, tokenizer)
            scores.append(score)
    
    elif benchmark_type == "Reshad":
        for index, row in df.iterrows():
            question = row['text']
            options = row['options']
            correct_option = row['answer']
            score = handle_topic_classification(question, options, correct_option, model_name, tokenizer)
            scores.append(score)

    elif benchmark_type == "ContextQA":
        for index, row in df.iterrows():
            question = row['question']
            context = row['context']
            actual_answer = row['answer']
            score = handle_context_qa(question, context, actual_answer, model_name, tokenizer)
            scores.append(score)

    elif benchmark_type == "Arzuman":
        for index, row in df.iterrows():
            question = row['text']
            topic_options = row['options']
            correct_topic = row['answer']
            score = handle_multiple_choice(question, topic_options, correct_topic, model_name, tokenizer)
            scores.append(score)

    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    if scores:
        average_score = sum(scores) / len(scores)
        results.loc[model_name, benchmark_type] = average_score

results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

for file in dataset_files:
    benchmark_type = get_benchmark_from_filename(file, metadata)
    print(f"Running {benchmark_type} benchmark for file: {file}")
    
    df = pd.read_excel(file)
    df = df[:2] 
    print(df)

    for model_name in metadata['supported_models_hf']:
        print(f"Running {benchmark_type} for model {model_name}")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        run_benchmark(model, tokenizer, benchmark_type, df, results)

print("\nAverage Scores:\n", results)

results.to_excel(results_file)
