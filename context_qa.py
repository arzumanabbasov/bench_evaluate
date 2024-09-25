from base import *
import logging
import torch

def create_combined_prompt_context(context: str, question: str) -> str:
    """
    Create the prompt for the LLM to generate an answer based on the given context and question.
    """
    return f"""
        You are an answer generator AI in Azerbaijani. Your task is to generate answers based on the provided context and the given question.

        **Example:**

        **Context:** Azərbaycan Respublikası Cənubi Qafqazda yerləşən bir ölkədir. İqtisadi və mədəni mərkəzi Bakı şəhəridir.
        
        **Question in Azerbaijani:** Azərbaycan Respublikasının paytaxtı haradır?

        **Generated Answer in Azerbaijani:** Bakı şəhəri.

        **Your Task:**

        **Context in Azerbaijani:** {context}

        **Question in Azerbaijani:** {question}

        Provide a clear and accurate answer in Azerbaijani based on the context, and include your answer in 1-2 sentences.
    """

def get_answer_from_local_huggingface_context(model, question: str, context: str, tokenizer) -> str:
    """
    Send a prompt to the Hugging Face model and retrieve the answer based on the provided context and question.
    
    Args:
        model: The Hugging Face model for question answering.
        question (str): The question to be answered.
        context (str): The context in which the question is to be answered.
        tokenizer: The tokenizer associated with the model.
    
    Returns:
        str: The generated answer based on the context and question.
    """

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1 

    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer.strip() if answer else "Error"

def get_evaluation_score_context(question: str, actual_answer: str, predicted_answer: str) -> str:
    """
    Generate an evaluation score between 0 and 100 by comparing the actual and predicted answers.
    """

    prompt = f"""
            Evaluate the following answers and provide a score from 0 to 100 based on how well the predicted
            answer matches the actual answer based on the asked question. Provide the score only, without any additional text.

            0-10: No answer or completely incorrect
            11-30: Significant errors or missing key information
            31-50: Some errors or incomplete information, but recognizable effort
            51-70: Mostly accurate with minor errors or omissions
            71-90: Very close to the actual answer with only minor discrepancies
            91-100: Accurate or nearly perfect match

            **Example:**

            **Question that asked in Azerbaijani:** Makroiqtisadiyyat nədir və mikroiqtisadiyyatdan necə fərqlənir?  
            **Actual Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın böyük miqyasda təhlili ilə məşğul olur, mikroiqtisadiyyat isə kiçik miqyasda, yəni fərdi bazarlarda və şirkətlərdə baş verən prosesləri öyrənir.  
            **Predicted Answer in Azerbaijani:** Makroiqtisadiyyat iqtisadiyyatın ümumi aspektlərini öyrənir, mikroiqtisadiyyat isə fərdi bazarları təhlil edir.  
            **Score (0 to 100):** 65

            **Your Task:**

            **Question that asked in Azerbaijani:** {question}

            **Actual Answer in Azerbaijani:** {actual_answer}

            **Predicted Answer in Azerbaijani:** {predicted_answer}

            **Score (0 to 100):**
            """


    payload = {
        "model": MODEL_LLAMA_3_1_405B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }

    for attempt in range(NUM_RETRIES):
        try:
            completion = client_openai.chat.completions.create(**payload)
            if completion.choices:
                content = completion.choices[0].message.content
                if content:
                    return content.strip()
                logging.error("Content in response is None.")
            else:
                logging.error(f"Unexpected response format: {completion}")
        except Exception as e:
            logging.error(f"Request to OpenAI failed: {e}")
    return "Error"

