"""
This program performs Aspect-Based Sentiment Analysis (ABSA) on a single customer review
using a pre-trained transformer sentiment classifier.

A customer review usually carries more than one opinion ("the pasta was delicious, but the
service was rude"). A single document-level label therefore destroys information. Your task is
to report the sentiment of ONE requested aspect of the review.

> The program accepts two parameters provided as a command line input.
> The first input is the customer review. The second input is the aspect to be analysed.
> The aspect will always be one of the keys of ASPECT_LEXICON given below.
> The output must be exactly one of: POSITIVE / NEGATIVE / NEUTRAL / NOT_MENTIONED
> Output should be in upper-case.
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from
  the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string>

The following examples are given for your reference:

 Terminal Input: python template.py "The pasta was absolutely delicious, but the service was rude and careless." "food"
Terminal Output: POSITIVE

 Terminal Input: python template.py "The pasta was absolutely delicious, but the service was rude and careless." "service"
Terminal Output: NEGATIVE

 Terminal Input: python template.py "The pasta was absolutely delicious, but the service was rude and careless." "price"
Terminal Output: NOT_MENTIONED

You are expected to create some examples of your own to test the correctness of your approach.

Note: The solution must run on CPU only. The model is small and the review is short, so a
      correct solution takes only a few seconds of computation after the model has loaded.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import re
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
ALERT: * * * No changes are allowed to the three constants below  * * *
"""
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# The classifier returns three logits. This is the fixed meaning of each index.
ID2SENTIMENT = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# An aspect is considered to be mentioned in a clause if any one of its keywords occurs in
# that clause as a WHOLE WORD (case-insensitive).
ASPECT_LEXICON = {
    "food":      ["food", "foods", "dish", "dishes", "meal", "meals", "pasta", "pizza",
                  "burger", "curry", "starter", "starters", "dessert", "desserts",
                  "taste", "tastes", "flavour", "flavours", "flavor", "flavors",
                  "portion", "portions"],
    "service":   ["service", "staff", "waiter", "waiters", "waitress", "waitresses",
                  "server", "servers", "manager", "host", "hostess"],
    "wait_time": ["wait", "waits", "waited", "waiting", "queue", "queues", "delay",
                  "delays", "delayed", "slow", "quick", "prompt"],
    "price":     ["price", "prices", "priced", "cost", "costs", "bill", "value",
                  "expensive", "cheap", "overpriced", "affordable"],
    "ambience":  ["ambience", "ambiance", "atmosphere", "decor", "music", "noisy",
                  "loud", "seating", "vibe", "clean", "dirty"],
}

"""
* * * Changes allowed from here  * * * 
"""

def absa_function(model, tokenizer, review, aspect):
    '''
    The steps are given for your reference:

    1. CLAUSE SEGMENTATION.
       Split the review into clauses. A clause boundary occurs at any of the following:
         (a) immediately after a sentence-final ".", "!" or "?"
         (b) at a ";"
         (c) at any of these contrast/coordination markers, matched as whole words and
             case-insensitively: but, however, although, though, whereas, yet
             (the marker word itself, and any comma immediately before or after it, is
              removed and does NOT belong to either clause)
       After splitting, strip leading/trailing whitespace and leading/trailing commas from
       every clause, and discard any clause that is empty.

    2. ASPECT MATCHING.
       Keep only those clauses in which at least one keyword of ASPECT_LEXICON[aspect] occurs
       as a whole word, case-insensitively. A useful pattern is r"\\b" + keyword + r"\\b" with
       the re.IGNORECASE flag.
       If NO clause matches, the function must return "NOT_MENTIONED".

    3. PER-CLAUSE CLASSIFICATION.
       For each matching clause, in the order in which it appears in the review:
         - tokenize the clause on its own with truncation=True and max_length=128
         - run the model under torch.no_grad() to obtain the logits
         - convert the logits into a probability distribution using softmax
       Process one clause at a time. Do not batch the clauses together.

    4. AGGREGATION.
       Average the probability distributions of all matching clauses element-wise, i.e. take
       the mean over clauses for each of the three positions. Then take the index of the
       largest value of this averaged distribution and map it through ID2SENTIMENT.
       Note: averaging the probabilities is NOT the same as taking a majority vote over the
       per-clause labels, and the two can disagree. Only the averaging rule is accepted.

    5. Return the resulting label. Output is case-sensitive and must be exactly one of:
       POSITIVE, NEGATIVE, NEUTRAL, NOT_MENTIONED

    Note: The model and tokenizer are already initialized. Do not modify that section.
          The output should be in upper-case and should not contain any warning or additional
          text beyond the actual answer.
          Your output will be evaluated against the correct answer and only "Exact Matches"
          will be marked Correct.
    '''

    # 1. CLAUSE SEGMENTATION
    markers = ["but", "however", "although", "though", "whereas", "yet"]
    marker_pattern = r"\s*,?\s*\b(?:" + "|".join(markers) + r")\b\s*,?\s*"

    text = re.sub(marker_pattern, "\n", review, flags=re.IGNORECASE)
    text = re.sub(r"[.!?]", "\n", text)
    text = text.replace(";", "\n")

    raw_clauses = text.split("\n")
    clauses = []
    for c in raw_clauses:
        c = c.strip()
        c = c.strip(",")
        c = c.strip()
        if c:
            clauses.append(c)

    # 2. ASPECT MATCHING
    keywords = ASPECT_LEXICON[aspect]
    matching_clauses = []
    for c in clauses:
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, c, flags=re.IGNORECASE):
                matching_clauses.append(c)
                break

    if not matching_clauses:
        return "NOT_MENTIONED"

    # 3. PER-CLAUSE CLASSIFICATION
    prob_list = []
    for clause in matching_clauses:
        inputs = tokenizer(clause, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        prob_list.append(probs)

    # 4. AGGREGATION
    stacked = torch.stack(prob_list, dim=0)
    mean_probs = torch.mean(stacked, dim=0)
    label_idx = int(torch.argmax(mean_probs).item())
    final_output = ID2SENTIMENT[label_idx]

    # 5. Return
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':

    review = sys.argv[1].strip()
    aspect = sys.argv[2].strip().lower()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = absa_function(model, tokenizer, review, aspect)
    print(out.strip())

    """ End to call """
