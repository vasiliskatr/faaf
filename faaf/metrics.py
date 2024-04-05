from pandas import DataFrame

def safe_divide(numerator, denominator):
        """Safely divide two numbers, avoiding division by zero."""
        if denominator == 0:
            return 0
        else:
            return numerator / denominator

def data_prep_for_scoring(df:DataFrame, human_annotated_facts_col: str, pred_facts_col: str, is_valid_col: None|str = None) -> DataFrame:
    '''
    '''
    assert human_annotated_facts_col in df.columns, f"{human_annotated_facts_col} not found it the DataFrame!"
    assert pred_facts_col in df.columns, f"{pred_facts_col} not found it the DataFrame!"

    # Only relevant for FaaF:
    # If an is_valid_col is given, filter out the invalid rows and consider only the valid ones.
    # the rows where is_valid_col=True contain mock LM responses, resulting 
    # from failure of the LM to format the response appropriately for parsing into function object.
    if is_valid_col:
        print(f'{(~df[is_valid_col]).sum()} rows have invalid FaaF responses. Filtering them out..')
        df_failed_pydantic = df[~df[is_valid_col]]
        failed_facts = {key: value for d in df_failed_pydantic[pred_facts_col].tolist() for key, value in d.items()}
        df = df[df[is_valid_col]].copy()
        print(f'Number of rows with valid FaaF responses: {len(df)}')
    else:
        failed_facts = {}

    # Flatten the lists of dictionaries into single dictionaries
    flattened_gt_facts = {key: value for d in df[human_annotated_facts_col].tolist() for key, value in d.items()}
    flattened_pred_facts = {key: value for d in df[pred_facts_col].tolist() for key, value in d.items()}

    # confirm the number of gt facts is same as the number of pred facts.
    assert len(flattened_gt_facts) == len(flattened_pred_facts)

    # confirm all  verification values are one of:
    accepted_values = ['False', 'No_response', 'True']
    assert all(item in accepted_values for item in set(flattened_pred_facts.values())), "There are values beyon 'False', 'No_response', 'True' in the set."
    assert all(item in accepted_values for item in set(flattened_gt_facts.values())), "There are values beyon 'False', 'No_response', 'True' in the set."

    # Only relevant for prompting verification
    # Filter fact keys where flattened_pred_facts value is not 'no_response' 
    facts_with_valid_answer = {key: value for key, value in flattened_pred_facts.items() if value != 'No_response'}
    no_answer_facts = {key: flattened_pred_facts[key] for key in flattened_pred_facts.keys() if key not in facts_with_valid_answer}
    print(f'{len(flattened_pred_facts) - len(facts_with_valid_answer)} facts with "No_answer" value found - ignoring from ER calculation')

    # Create new dictionaries excluding the 'no_response' rows
    flattened_gt_facts = {key: flattened_gt_facts[key] for key in facts_with_valid_answer}
    flattened_pred_facts = {key: flattened_pred_facts[key] for key in facts_with_valid_answer}

    # confirm all filtered verification values are one of:
    accepted_values_tf = ['False', 'True']
    assert all(item in accepted_values_tf for item in set(flattened_pred_facts.values())), "There are values beyon 'False', 'True' in pred facts."
    assert all(item in accepted_values_tf for item in set(flattened_gt_facts.values())), "There are values beyon 'False', 'True' in human annotated facts."
    assert len(flattened_gt_facts) == len(flattened_pred_facts)

    # ensuere that all the human annotated facts can be found in the pred facts
    assert set(flattened_gt_facts.keys()) == set(flattened_pred_facts.keys())

    return flattened_gt_facts, flattened_pred_facts, {**failed_facts, **no_answer_facts}

def error_rate_(human_annotated_facts_dict: dict, pred_facts_dict:dict) -> float:
    '''
    Calculates the error rate between human-annotated facts and predicted facts 
    as the ratio of incorrect predictions to the total number of predictions.
    Ignores No-response facts from prompt approach and not valid FaaF responses!
    '''
    # assert the same and only the same facts exist in pred and human anotated dicts.
    assert set(human_annotated_facts_dict.keys()) == set(pred_facts_dict.keys())
    
    errors = 0  # Initialize the count of errors
    # Iterate through the predictions and compare with ground truth
    for key, pred_value in pred_facts_dict.items():

        if pred_value != human_annotated_facts_dict[key]:
            errors += 1
    
    # Calculate error rate
    error_rate = safe_divide(errors, len(pred_facts_dict))
    
    return error_rate

def f1_micro_(human_annotated_facts_dict: dict, pred_facts_dict:dict) -> float :
    '''
    Calculates the F1 micro score for binary classification tasks, specifically focusing on the 'False' class.
    The function computes Precision, Recall, and the F1 micro score based on the 'False' predictions and ground truth.

   Ignores No-response facts from prompt approach and not valid FaaF responses!

   The formula for F1 micro score is F1MICRO = (2 · P · R) / (P + R), where:
    - P is Precision = (P_ ∩ G) / P_, the ratio of correctly predicted False facts to all predicted False facts.
    - R is Recall = (P_ ∩ G) / G, the ratio of correctly predicted False facts to all ground-truth False facts.
    P_ is the set of predicted False and G is the set of ground truth False.
    '''
    # assert the same and only the same facts exist in pred and human anotated dicts.
    assert set(human_annotated_facts_dict.keys()) == set(pred_facts_dict.keys())

    # Extract 'False' facts from ground truth and predictions
    gt_false = {key: value for key, value in human_annotated_facts_dict.items() if value == 'False'}
    pred_false = {key: value for key, value in pred_facts_dict.items() if value == 'False'}

    # Compute intersection of 'False' predictions with ground truth 'False' facts
    intersection_false_values = {k: v for k, v in gt_false.items() if k in pred_false and v == 'False' and pred_false[k] == 'False'}
    #print(intersection_false_values)
    # Calculate Precision and Recall
    P = safe_divide(len(intersection_false_values), len(pred_false))
    R = safe_divide(len(intersection_false_values), len(gt_false))

    print("Precision:", P)
    print("Recall:", R)

    # Calculate F1 micro score
    F1_micro = safe_divide((2 * P * R), (P + R))
    return F1_micro



