"""
This program is build with Flan-T5-XL LLM to be able to answer a question in YES/NO using the provided context as in-context learning. 

> The program accepts two parameters provided as a command line input. 
> The two inputs represent the context and the question.
> The question output is deterministic i.e. its either YES or NO. You are required to use logits to extract the output.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.
> Note that the assignment and evaluation test cases are carefully sampled from the model itself, eliminating any chance of hallucination.

Syntax: python template.py <CONTEXT> <QUESTION>

The following example is given for your reference:

 Terminal Input: python assignment.py 'Albert has been working on his project all week. He finished the final report today and submitted it to his manager before the deadline.' 'Did Albert submit his project report on time?'
Terminal Output: YES

 Terminal Input: python assignment.py 'Albert has been working on his project all week. He finished the final report today and submitted it to his manager after the deadline.' 'Did Albert submit his project report on time?'
Terminal Output: NO

 Terminal Input: 'John started watering his plants every morning this week.' 'Did John water his plants yesterday morning?'
Terminal Output: YES

 Terminal Input: 'John started watering his plants every morning this week.' 'Did John water his plants last month?'
Terminal Output: NO

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

#####
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model,tokenizer,context,question): 

    # Generate a deterministic output either 'YES' or 'NO' answering the question using the provided context

    '''
    1. Engineer the prompt using the query and the context.
    2. Tokenize the prompt.
    3. Generate output for the prompt. (Suggestion: Extract the logits to determine the output directly.)
    4. Format the output to be exactly YES or NO. 
    
    Remember that there should be no additional output including any warning messages in the terminal.
    '''

    prompt = (
        f"Read the context and answer the question with yes or no.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Options: yes, no\n\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Score only the first decoder step; the answer is decided by comparing the "yes" and "no" logits
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    logits = outputs.logits[0, -1, :]

    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("no", add_special_tokens=False).input_ids[0]

    final_output = "YES" if logits[yes_id] > logits[no_id] else "NO"

    return final_output


"""
ALERT: * * * No changes are allowed below this comment  * * *
"""
if __name__ == '__main__':

    context = sys.argv[1].strip().lower()
    question = sys.argv[2].strip().lower()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,context,question)
    print(out.strip())

    """ End to call """