import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from base import *

def get_model_answer_multiple_options(question, options, model, tokenizer, dstype) -> str:
    """
    Sends a query to the model and retrieves the response.

    Args:
        question (str): The question to be categorized.
        options (str): The options for categorization.
        model (str): The model to be used.
        dstype (str): The type of task ('tc', 'mc').

    Returns:
        str: The model's response.
    """


    if dstype == 'tc':
        prompt = (
            "You are given a statement along with multiple options that represent different topics. "
            "Choose the option that best categorizes the statement based on its topic. "
            "Select the single option (e.g., A, B, C, etc.) that most accurately describes the topic of the statement.\n"
            f"Statement: {question}\nOptions: {options}\n"
        )
    elif dstype == 'mc':
        prompt = (
            "You are an AI tasked with selecting the most accurate answer in Azerbaijani based on a given question. "
            "You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. "
            "Choose the single letter (A, B, C, etc.) that best answers the question. "
            f"Question: {question}\nOptions: {options}\n"
        )
    else:
        raise Exception("Invalid dstype")

    # Tokenize the input prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract scores
    start_score = torch.max(outputs.start_logits).item()
    end_score = torch.max(outputs.end_logits).item()

    # Determine the predicted option based on scores
    best_score = start_score + end_score

    # Select the best option based on the score (simplified logic)
    predicted_answer_index = torch.argmax(outputs.start_logits).item()
    predicted_option = chr(65 + predicted_answer_index)  # Assuming 'A', 'B', etc.

    return predicted_option


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
