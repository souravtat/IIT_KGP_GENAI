"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    def generate_answer(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Generate the first answer without any previous context.
    first_prompt = (
        "Answer the question accurately and concisely.\n"
        f"Question: {questions[0]}\n"
        "Answer:"
    )
    answer_1 = generate_answer(first_prompt)


    # Supply question 1 and its generated answer as context for question 2.
    second_prompt = (
        "Use the previous question and answer as context for the follow-up "
        "question. Answer accurately and concisely.\n\n"
        f"Previous question: {questions[0]}\n"
        f"Previous answer: {answer_1}\n\n"
        f"Follow-up question: {questions[1]}\n"
        "Answer:"
    )
    answer_2 = generate_answer(second_prompt)
    
    # For the final question, score only the two permitted answers. Scoring
    # complete candidate sequences keeps the result deterministic and also
    # works if a candidate is represented by more than one tokenizer token.
    final_prompt = (
        "Use the context below to answer the question. The answer must be "
        "either Yes or No.\n\n"
        f"Context question: {questions[1]}\n"
        f"Context answer: {answer_2}\n\n"
        f"Question: {questions[2]}\n"
        "Answer (Yes or No):"
    )
    final_inputs = tokenizer(final_prompt, return_tensors="pt")

    scores = {}
    for candidate in ("Yes", "No"):
        labels = tokenizer(candidate, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(**final_inputs, labels=labels).logits

        token_log_probs = torch.log_softmax(logits, dim=-1)
        candidate_log_probs = token_log_probs.gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        scores[candidate] = candidate_log_probs.mean().item()

    final_output = max(scores, key=scores.get).upper()
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())

    """ End to call """
