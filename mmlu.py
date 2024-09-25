import os
import pandas as pd
from typing import List, Tuple
from base import *


def get_model_answer_multiple_options(question, options, model) -> str:
    """
    Sends a query to the model and retrieves the response.

    Args:
        question (str): The question to be categorized.
        options (str): The options for categorization.
        existing_answer (str): The existing answer.

    Returns:
        str: The model's response.
    """

    messages = [
        {
            "role": "user",
            "content": f"{question} Options: {options}"
        }
    ]

    
    answer = ''
    try:
        stream = client_ollama.chat(
            model=model,  
            messages=messages,
            stream=True
        )
    except Exception as e:
        print(f"Error during streaming: {e}")
        return "Error"

    try:
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                answer += chunk['message']['content']
            else:
                print(f"Unexpected chunk format: {chunk}")
    except Exception as e:
        print(f"Error processing stream: {e}")
        return "Error"
    
    return answer


def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    """
    Compare the actual answer with the predicted answer.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    return 1 if actual_answer.lower() == predicted_answer.lower() else 0


def evaluate_llm_on_datasets(model: str,) -> None:
    """
    Evaluate the LLM on multiple datasets by comparing predicted answers to actual answers.
    
    Args:
        model (str): The name of the LLM model to be used.
    """
    results = []

    for dirname, _, filenames in os.walk('./datasets/input_datasets/mmlu/'):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirname, filename))

            if 'translated_choices' in df.columns and 'question' in df.columns and 'actual_answer' in df.columns:
                for _, row in df.iterrows():
                    question = row['question']
                    options = row['translated_choices']
                    actual_answer = row['actual_answer']  

                    predicted_answer = get_model_answer_multiple_options(question, options, model)

                    result = compare_answers(actual_answer, predicted_answer)
                    results.append({
                        'question': question,
                        'options': options,
                        'actual_answer': actual_answer,
                        'predicted_answer': predicted_answer,
                        'result': result
                    })

    total_questions = len(results)
    correct_answers = sum(res['result'] for res in results)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0

    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")



if __name__ == "__main__":
    model_name = '' 
    evaluate_llm_on_datasets(model_name)
