from openai import OpenAI
from pandas import DataFrame
import concurrent.futures
from typing import Tuple, List, Dict

def gt_answer2facts(client: OpenAI, gt_question:str, gt_answer:str):
    
    """
    Extracts and formats facts from a given passage that answer a specified question, using OpenAI's API.

    Parameters:
    - client (OpenAI): An authenticated instance of the OpenAI client.
    - gt_question (str): The question to be answered by the facts.
    - gt_answer (str): The passage containing information relevant to the question.

    Returns:
    - dict: A dictionary containing the original question (`query`) and the extracted facts (`gt_facts`) as a formatted string.

    The function formats a prompt for the OpenAI API, instructing it to generate a list of succinct, independent facts from the passage that directly answer the question. Each fact is presented in a clear, pronoun-free syntax and is prefixed with a "-" for clarity.
    """
    prompt = f'''
    Convert the given passage into a list of short facts which specifically answer the given question.
    Make sure that the facts can be found in the given passage.
    The facts should be coherent and succint sentences with clear and simple syntax.
    Do not use pronouns as the subject or object in the syntax of each fact.
    The facts should be independent to each other.
    Do not create facts from the passage which are not answering the given question.
    Add a "-" before each fact.

    Passage: "{gt_answer.replace('Answer: ', '')}"

    Question: "{gt_question.replace('Question: ', '')}"
    '''
    print(prompt)

    llm_output = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model = "gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=60)
    
    response =  llm_output.choices[0].message.content

    return {
        "query":gt_question,
        "gt_facts":response
        }


def make_gt_facts_col(test_data: DataFrame, client: OpenAI, num_threads: int = 5) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Parallelizes the extraction of ground truth (GT) facts from a dataset using the `gt_answer2facts` function.
    
    This function applies the `gt_answer2facts` to each row in `test_data` in parallel, using a specified number of threads,
    to extract coherent and succinct facts from the answers that specifically address the questions.
    
    Parameters:
    - test_data (pd.DataFrame): A DataFrame containing at least two columns: 'question' and 'answer'.
    - client: An instance of the OpenAI client, configured and authenticated to use the OpenAI API.
    - num_threads (int, optional): The number of threads to use for parallel processing. Defaults to 5.
    
    Returns:
    - Tuple containing:
        - Updated `test_data` DataFrame with a new column 'gt_facts' containing the extracted facts.
        - A list of dictionaries with the original query and the corresponding extracted GT facts.
    
    Each dictionary in the results list contains:
    - 'query': The original question.
    - 'gt_facts': The extracted facts as a string, each fact prefixed with "-".
    
    Utilizes `ThreadPoolExecutor` for parallel processing to enhance efficiency.
    """
    def apply_parallel(item, client):
        return gt_answer2facts(client=client, gt_question=item["question"], gt_answer=item["answer"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(apply_parallel, item=item, client=client) for _, item in test_data.iterrows()]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Map the results back to the test_data DataFrame
    gt_facts_list = []
    for _, item in test_data.iterrows():
        gt_facts_list.append([res['gt_facts'] for res in results if res["query"] == item["question"]][0])

    test_data["gt_facts"] = gt_facts_list

    return test_data, results





