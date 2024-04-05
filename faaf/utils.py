import time
import functools
import pandas as pd
import random

from faaf.config import ACCEPTED_LM_NAMES, HUMAN_ANNOTATED_FACTS_COL_NAMES
from openai import OpenAI
from mistralai.client import MistralClient
from anthropic import Anthropic

from dotenv import load_dotenv
import os

from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

def llm_init(llm_name:str) -> MistralClient | OpenAI | Anthropic:
    '''
    Initializes and returns a client object for interacting with a specified large language model (LLM).

    Based on the name of the LLM provided, this function looks up the relevant API key from the environment variables
    and initializes the appropriate client. The function supports clients for Mistral, OpenAI, and Anthropic models.
    '''
    assert llm_name in ACCEPTED_LM_NAMES, f"llm_name must be one of {ACCEPTED_LM_NAMES}"

    if llm_name == "mistral-large-latest":
        llm_api_key = os.getenv("MISTRAL_API_KEY")
        client = MistralClient(api_key=llm_api_key)

    elif llm_name in ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]:
        llm_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=llm_api_key)

    elif llm_name in ["claude-3-opus-20240229","claude-3-sonnet-20240229"]:
        llm_api_key = os.getenv("CLAUDE_API")
        client = Anthropic(api_key=llm_api_key)

    return client


def load_test_data_hugging_face() -> pd.DataFrame:
    '''
    Load and prepare the facts-enriched WikiEval test data.
    '''
    dataset = load_dataset('Vaskatr/WikiEvalFacts')
    df = dataset["train"].to_pandas()
    df = deserialise_human_annotated_facts(df=df)

    # normalisation of the humman annotated fact keys is necessary because
    # we rely on matching human annotated and pred facts in the metrics calculation.
    return normalise_fact_keys(
        df=df,
        cols=HUMAN_ANNOTATED_FACTS_COL_NAMES
        )

def load_test_data(fpath: str) -> pd.DataFrame:
    '''
    Load and prepare the WikiEval test data.
    '''
    df = pd.read_csv(fpath)

    df = deserialise_human_annotated_facts(df=df)

    # normalisation of the humman annotated fact keys is necessary because
    # we rely on matching human annotated and pred facts in the metrics calculation.
    return normalise_fact_keys(
        df=df,
        cols=HUMAN_ANNOTATED_FACTS_COL_NAMES
        )
    

def deserialise_human_annotated_facts(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts the human anotated facts entries from str to dictionaries.
    '''
    assert all(column in df.columns for column in HUMAN_ANNOTATED_FACTS_COL_NAMES), "Not all columns form human annotation are in the DataFrame"
    for col in HUMAN_ANNOTATED_FACTS_COL_NAMES:
        df[col]= df[col].apply(lambda x: eval(x))

    return df


def time_function(func):
    """
    Decorator to measure the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__!r} executed in {(end_time - start_time):.4f} seconds")
        return result

    return wrapper

def retry(max_retries=4):
    ''' Retry decorator for LLM calls - currently not used'''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
        return wrapper
    return decorator

def normalise_fact_keys(df:pd.DataFrame, cols:list[str]):
    '''
    This will be applied to the human_annotated fact strings of 
    the dataset upon importing.
    It mirrors the text formating changes made in make_facts_list(). 
    We will need to match the human_annotated and pred fact string statements
    when calculating the metrics. 
    '''
    for col in cols:
        df[col] = df[col].apply(lambda d: {k.replace('-', ' ')
                                           .replace('\n', '')
                                           .replace('.', '')
                                           .replace("'","")
                                           .replace('"','')
                                           .strip(): v for k, v in d.items()})
    
    return df

def make_facts_list(facts_str:str):
    '''
    Given a string of generated facts seperated by "-", it retruns a list of fact statements. 
    '''
    facts_str = facts_str.replace("'", '').replace('"', '')
    facts_list=[item for item in facts_str.split('- ') if item]
    facts_list = [fact.replace('-', ' ')
                  .replace('\n', '')
                  .replace('.', '')
                  .strip()
                  for fact in facts_list]
    
    return facts_list

def generate_mock_value(field_schema):
    """
    Generate a mock value based on the provided schema.
    This is used in FaaF cases where the LLM response fails to parsed into the pydantic class.
    This can happen for several reasons and depends on the LLM. 
    """
    field_type = field_schema.get('type', None)
    if 'enum' in field_schema:
        return random.choice(field_schema['enum'])
    elif field_type == 'string':
        return "Mock String"
    elif field_type == 'integer':
        return random.randint(1, 100)  # Example range
    elif field_type == 'number':
        return random.uniform(1.0, 100.0)  # Example range
    elif field_type == 'boolean':
        return random.choice([True, False])
    elif 'anyOf' in field_schema:
        # Example handling for 'anyOf' - could be extended for more complex scenarios
        return generate_mock_value(random.choice(field_schema['anyOf']))
    else:
        return None

def create_mock_response(schema):
    """
    Create a mock response based on the provided JSON schema of a Pydantic model.
    """
    mock_response = {}
    for property_name, property_schema in schema.get('properties', {}).items():
        mock_response[property_name] = generate_mock_value(property_schema)
    return mock_response
