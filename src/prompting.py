"""
This program is build with Flan-T5-XL LLM to be able to determine output of a MCQ question with four options. 

> It accepts five parameters provided as a command line input. 
> The first input represents the question and the next four input are the options. 
> The output should be the option number: A/B/C/D 
> Output should be upper-case
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Blue" "Green" "Red" "Yellow"
Terminal Output: A

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Blue" "Red" "Yellow"
Terminal Output: B

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Red" "Blue" "Yellow"
Terminal Output: C

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Red" "Yellow" "Blue"
Terminal Output: D

You are expected to create some examples of your own to test the correctness of your approach.
Remember that smaller models require much smarter prompt engineering to achive desired output.

Note: You have to set HF_TOKEN environment variable from HuggingFace in your system.
      Never share your token in this code.

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

def llm_function(model,tokenizer,q,a,b,c,d):
    '''
    The steps are given for your reference:

    1. Properly formulate the prompt as per the question. The output must always be upper-case. You may post-process to get the desired output.
    2. Tokenize the prompt
    3. Pass the tokenized prompt to the model get output in terms of logits since the output is deterministic (Hint: Advanced Prompt Engineering Hands-On "Handling discrette outputs effectively using logits")  
    4. Extract the correct option from the model.
    5. Clean output and return.
    6. Output is case-sensative: A,B,C or D

    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
          The output should be in upper-case and should not contain any warning or additional text beyond the actual answer.
          Your output will be evaluated against the correct answer and only "Exact Matches" will be marked Correct.
    export HF_TOKEN=
    python prompting.py "What color is the sky on a clear, sunny day?" "Blue" "Green" "Red" "Yellow"
    '''
    
    # Create a proper prompt for the MCQ question
    prompt_text = (
        f"Question: {q}\n"
        f"Option A: {a}\n"
        f"Option B: {b}\n"
        f"Option C: {c}\n"
        f"Option D: {d}\n"
        "Which option is correct? Answer with just the letter A, B, C or D:"
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").input_ids

    # Score each candidate letter directly via logits (deterministic, avoids
    # free-form generation drifting away from a single clean letter).
    candidates = ["A", "B", "C", "D"]
    decoder_start_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_id]])

    with torch.no_grad():
        logits = model(input_ids=inputs, decoder_input_ids=decoder_input_ids).logits[0, -1]

    best_choice, best_score = "", float("-inf")
    for letter in candidates:
        token_id = tokenizer.encode(letter, add_special_tokens=False)[0]
        score = logits[token_id].item()
        if score > best_score:
            best_score, best_choice = score, letter

    final_output = best_choice
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':
    question = sys.argv[1].strip()
    option_a = sys.argv[2].strip()
    option_b = sys.argv[3].strip()
    option_c = sys.argv[4].strip()
    option_d = sys.argv[5].strip()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,question,option_a,option_b,option_c,option_d)
    print(out.strip())

    """ End to call """