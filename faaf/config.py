
ACCEPTED_LM_NAMES = [
    "gpt-3.5-turbo-0125", 
    "gpt-4-0125-preview",
    "mistral-large-latest",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229"
    ]

FACT_CONFIRMATION_LITERAL = "True"
FACT_REJECTION_LITERAL = "False"
NOT_CLEAR_LITERAL = "not clear from the given passage"

HUMAN_ANNOTATED_FACTS_COL_NAMES = [
    'human_evaluated_facts_answer',
    'human_evaluated_facts_ungrounded_answer',
    'human_evaluated_facts_poor_answer'
    ]
FACT_VERIFICATION_METHODS = [
    'prompt',
    'faaf'
]