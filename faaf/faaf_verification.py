from faaf.utils import (
    make_facts_list,
    create_mock_response
    )
from faaf.config import (
    FACT_CONFIRMATION_LITERAL,
    FACT_REJECTION_LITERAL,
    NOT_CLEAR_LITERAL
    )
from pydantic import BaseModel, create_model, Field
from typing import Literal, Optional
import json
from openai import OpenAI
from mistralai.client import MistralClient
from anthropic import Anthropic
from faaf.claude_utils import pydantic2claude_xml, construct_tool_use_system_prompt, llm2dict_claude

def dynamic_to_dict(self):
    """
    Convert a model instance to a dictionary.
    Equivalent to model.dump() but in an explicitelly adjustable format.
    """
    return {field_properties.title: getattr(self, field_name) for field_name, field_properties in self.model_fields.items()}

def make_dynamic_field_evaluator(self):
    """Factory function to create a dynamic evaluator method."""
    def dynamic_field_evaluator(self):
        """Dynamically evaluates field values based on external conditions."""
        result = {}
        for field_name, field_properties in self.model_fields.items():
            value = getattr(self, field_name)
            if 'text_evidence' not in field_name:
                result[field_properties.title] = value
        return result
    
    return dynamic_field_evaluator

def create_dynamic_model_with_methods(gt_facts, tfn:bool, citation:bool):
    '''
    Dynamically create a pydantic model with the pre-defined methods and input fact statements.
    '''
    assert isinstance(gt_facts, str)
    
    # Convert the string of gt_facts to a list with some cleaning
    gt_facts_list = make_facts_list(facts_str=gt_facts)

    if tfn:
        print("* Using TFN")
        type_fact = '''Literal[FACT_CONFIRMATION_LITERAL, FACT_REJECTION_LITERAL, NOT_CLEAR_LITERAL]'''
    else:
        print("* Using TF")
        type_fact = '''Literal[FACT_CONFIRMATION_LITERAL, FACT_REJECTION_LITERAL]'''

    if citation:
        print("* Using citation")
        type_citation = '''Optional[str]'''
        fields = {}
        for index, fact in enumerate(gt_facts_list):
            fields[f'fact_{index}_text_evidence'] = (
                type_citation, 
                Field(
                ..., 
                title=f'{fact}_evidence',
                description=f"Copy the exact text from the passage that directly supports the claim in triple backtics: ```{fact}```"
                )
            )
            fields[f'fact_{index}'] = (
                type_fact, 
                Field(
                    ..., 
                    title=fact, 
                    description=f"It is clear from the passage that {fact}. Respond by using one of the accepted Enum types",
                )
            )

        DynamicModel = create_model('FactChecker', **fields)
    else:
        DynamicModel = create_model('FactChecker', **{f'fact_{index}': (type_fact, Field(...,title=fact, description=f"It is clear from the passage that {fact}. Respond by using one of the accepted Enum types")) for index, fact in enumerate(gt_facts_list)})
    
    # Dynamically add methods to the model
    setattr(DynamicModel, 'to_dict', dynamic_to_dict)
    evaluator_method = make_dynamic_field_evaluator(DynamicModel)
    setattr(DynamicModel, 'evaluate_fields', evaluator_method)

    return DynamicModel


def _convert_schema(schema: dict) -> dict:
    props = {k: {"title": k, **v} for k, v in schema["properties"].items()}
    return {
        "type": "object",
        "properties": props,
        "required": schema.get("required", []),
    }

def _get_extraction_function_oai(entity_schema: dict) -> dict:
    return {"type": "function",
            "function": {
                        "name": "fact_checker",
                        "description": "Confirms or rejects facts from the passage.",
                        "parameters": {
                                    "type": "object",
                                    "properties": {
                                                  "info": {"type": "array", "items": _convert_schema(entity_schema)}
                                                  },
                                    "required": ["info"]
                                      },
                        }
           }

def _get_extraction_function_mistral(entity_schema: dict) -> dict:
    return {"type": "function",
            "function": {
                        "name": "fact_checker",
                        "description": "Confirms or rejects facts from the passage.",
                        "parameters": 
                             _convert_schema(entity_schema)}          
           }

def pydantic2function_object(model: BaseModel, llm_name:str):
    if llm_name == "mistral-large-latest":
        return [_get_extraction_function_mistral(model.model_json_schema())]
    elif llm_name in ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]:
        return [_get_extraction_function_oai(model.model_json_schema())]


def llm2dict_oai(response:str):
    try:
        res = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        # sometimes the LLM returns the function fields without the underscores.
        # we need to make sure that if that happens we replace the whitespaces with underscores.
        # otherwise the pydantic object instantiation will fail
        if isinstance(res["info"], dict):
            #print('ITS DICT!')
            return {'_'.join(key.split()): value for key, value in res["info"].items()}
        elif isinstance(res["info"], list):
            #print('ITS LIST!')
            return {'_'.join(key.split()): value for key, value in res["info"][0].items()}
    
    except Exception as e:
        # Handle any exception
        print(f"An error occurred: {e}")
        return
    
def llm2dict_mistral(response:str):
    try:

        res = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        # sometimes the LLM returns the function fields without the underscores.
        # we need to make sure that if that happens we replace the whitespaces with underscores.
        # otherwise the pydantic object instantiation will fail
        if isinstance(res, dict):
            #print('ITS DICT!')
            #print(res)
            print({'_'.join(key.split()): value for key, value in res.items()})
            
            return {'_'.join(key.split()): value for key, value in res.items()}
        elif isinstance(res, list):
            #print('ITS LIST!')
            #print(res)
            return {'_'.join(key.split()): value for key, value in res[0].items()}
    
    except Exception as e:
        # Handle any exception
        print(f"An error occurred: {e}")
        return

def llm_confirm_facts(pred_answer: str, pydantic_model: BaseModel, llm_name: str, client: MistralClient | OpenAI | Anthropic):
    '''
    Confirm facts encapsulated in the input pydantic model and the given passage using an LLM.
    Returns:
        tuple: A tuple containing the following elements:
            - O (BaseModel): An instance of the Pydantic model containing the confirmed facts.
            - token_count_dict (dict): A dictionary containing counts of completion and prompt tokens used during inference.
            - valid_response (bool): A flag indicating whether the response is valid.
    Notes:
        - The function interacts with different language models based on the provided `llm_name`.
        - It constructs appropriate tools and system prompts for different model types.
        - In case of an error during response validation, a mock response is created based on the provided Pydantic model.

    '''
    token_count_dict={}
    valid_response=True
    extraction_message = f"""Consider the given passage and assign the correct values in the 'fact_checker' function.
    
    Passage:\n
    {pred_answer}
    """  
    messages = [{"role": "user", "content": extraction_message}]

    #print(messages)
    #print(f'* Using: {llm_name}')
    #print(f' pred_answer ------> {pred_answer}\n')
    if llm_name == "mistral-large-latest":

        tool = pydantic2function_object(model=pydantic_model,
                                         llm_name=llm_name)

        completion = client.chat(
            model=llm_name,
            messages=messages,
            tools=tool
            )
        response_dict = llm2dict_mistral(completion)
        token_count_dict["completion_tokens"] = completion.usage.completion_tokens
        token_count_dict["prompt_tokens"] = completion.usage.prompt_tokens
    
    elif llm_name in ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]:

        tool = pydantic2function_object(model=pydantic_model,
                                         llm_name=llm_name)
        
        completion = client.chat.completions.create(
            model=llm_name,
            messages=messages,
            tools=tool,
            logprobs = True,
            top_logprobs = 4,
            tool_choice={"type": "function",
                        "function": {"name": "fact_checker"}},
            temperature=0
            )
        response_dict = llm2dict_oai(completion)
        token_count_dict["completion_tokens"] = completion.usage.completion_tokens
        token_count_dict["prompt_tokens"] = completion.usage.prompt_tokens

    elif llm_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        # anthropic models require tool introduction via system prompt
        tool = pydantic2claude_xml(model=pydantic_model)
        system_prompt = construct_tool_use_system_prompt([tool])

        completion = client.messages.create(
            max_tokens=4096,
            messages=messages,
            system=system_prompt,
            stop_sequences=["\n\nHuman:", "\n\nAssistant", "</function_calls>"],
            model=llm_name
            )
        
        response_dict = llm2dict_claude(completion=completion,
                                        pydantic_model=pydantic_model)
        token_count_dict["completion_tokens"] = completion.usage.output_tokens
        token_count_dict["prompt_tokens"] = completion.usage.input_tokens

    #print(f' completion ------> {completion}\n')
    #print(f' response dict ------> {response_dict}')

    try:
        O = pydantic_model(*[],**response_dict)
    except Exception as e:
        print("An error occurred:", e)
        print('\n=======MOCK-RESPONSE=======\n')
        valid_response = False
        schema= pydantic_model.model_json_schema()
        mock_response_dict = create_mock_response(schema)
        O = pydantic_model(*[],**mock_response_dict)

    return O, token_count_dict, valid_response

def not_clear2false(d: dict):
    '''
    If tfn mode, convert the NOT_CLEAR_LITERAL response to False
    '''
    assert isinstance(d, dict)
    
    tf_dict={}
    for k, v in d.items():
        if v == NOT_CLEAR_LITERAL:
            tf_dict[k]='False'
        else:
            tf_dict[k]=v
            
    return tf_dict

def fact_check_faaf_(gt_facts: str, tfn: bool, citation: bool, unique_id: str, context: str, llm_name: str, client: MistralClient | OpenAI | Anthropic):
    '''
    Performs a fact-checking operation using a given context and ground truth facts. This function dynamically creates
    a Pydantic model based on the ground truth facts, then leverages a Large Language Model (LLM) to confirm the accuracy
    of these facts within the provided context. The function can optionally transform ambiguous fact checks into a False
    value if the tfn (true-false-not clear) mode is enabled.

    Returns:
    - A dictionary containing the unique_id, evaluated facts, names of the evaluated facts, number of tokens used in the
      completion and the prompt, and a flag indicating if the response is considered valid.
    '''
    _dynamic_pydantic_model = create_dynamic_model_with_methods(
        gt_facts = gt_facts,
        tfn=tfn,
        citation=citation
        )
    #print(_dynamic_pydantic_model.model_json_schema())
    (
        llm_response_model,
        token_count_dict,
        valid_response
        ) = llm_confirm_facts(
         pred_answer=context,
         pydantic_model=_dynamic_pydantic_model,
         llm_name=llm_name,
         client=client
         )
    
    # if tfn mode, convert the NOT_CLEAR_LITERAL response to False
    if tfn:
        evaluated_fatcs = not_clear2false(llm_response_model.evaluate_fields())
    else:
        evaluated_fatcs = llm_response_model.evaluate_fields()

    #print(f'formated response ----------> ')
    #for key,value in llm_response_model.to_dict().items():
        #print(f'{key} -> {value}')
        #pass
    print('-')
    return {
        "unique_id": unique_id,
        "evaluated_facts": evaluated_fatcs,
        "evaluated_facts_names": llm_response_model.to_dict(),
        "completion_tokens": token_count_dict["completion_tokens"],
        "prompt_tokens": token_count_dict["prompt_tokens"],
        "valid_response": valid_response
        }

