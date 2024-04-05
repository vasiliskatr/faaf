
from faaf.utils import make_facts_list
from openai import OpenAI
from mistralai.client import MistralClient
from anthropic import Anthropic


# # # # # # # # # # # # # FACT-CHECK with PROMPT # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#@retry(max_retries=9)
def fact_check_prompt(context:str, unique_id:str, gt_facts:str, llm_name:str, client: MistralClient | OpenAI | Anthropic):
    '''
    Fact-check at QA level with prompt + context. Each fact is verified individually.
    '''
    # create list of facts statements
    gt_facts_list = make_facts_list(facts_str=gt_facts)

    # dict to hold the full fact verification response from the LM
    prompt_response_dict = {}
    # dict to hold the extracted True/False from the LM response by exact match
    verified_facts_dict = {}
    completion_tokens=0
    prompt_tokens=0


    for fact in gt_facts_list:
        usr_prompt = f'''
        passage:"{context}"
        
        Considering the given passage, the claim: {fact} is True or False?
        '''
        #print(usr_prompt)

        if llm_name == "mistral-large-latest":

            llm_output = client.chat(
                model=llm_name,
                messages=[{"role": "user",
                           "content": usr_prompt}]
                )
            print('-')
            response = llm_output.choices[0].message.content
            verification_result = exact_match_true_false(
                lm_text_response=response
                )
            completion_tokens += llm_output.usage.completion_tokens
            prompt_tokens += llm_output.usage.prompt_tokens

        elif llm_name in ["gpt-3.5-turbo-0125","gpt-4-0125-preview"]:

            llm_output = client.chat.completions.create(
                model=llm_name,
                messages=[{"role": "user",
                           "content": usr_prompt}],
                temperature=0,
                timeout=60
                )
            print('-')
            response = llm_output.choices[0].message.content
            verification_result = exact_match_true_false(
                lm_text_response=response
                )
            completion_tokens += llm_output.usage.completion_tokens
            prompt_tokens += llm_output.usage.prompt_tokens

        elif llm_name in ["claude-3-opus-20240229","claude-3-sonnet-20240229"]:

            llm_output = client.messages.create(
                model=llm_name,
                max_tokens=4096,
                messages=[{"role": "user", 
                           "content": usr_prompt}]
                           )
            print('-')
            #print(llm_output)
            response = llm_output.content[0].text
            verification_result = exact_match_true_false(
                lm_text_response=response
                )
            completion_tokens += llm_output.usage.output_tokens
            prompt_tokens += llm_output.usage.input_tokens

        #print(llm_output)
        #res[fact] = response
        prompt_response_dict[fact] = response
        verified_facts_dict[fact] = verification_result
    # convert raw LM response to True/False accordning to FACTSCORE
    #res = prompt_response2tf(res)

    return {"unique_id":unique_id,
            f"prompt_evaluated_facts_full_response": prompt_response_dict,
            f"prompt_evaluated_facts": verified_facts_dict,
            f"completion_tokens": completion_tokens,
            f"prompt_tokens": prompt_tokens
            }


def exact_match_true_false(lm_text_response: str) -> str:
    '''
    Determines whether the given response string contains the term 'true', 'false', or neither,
    regardless of case sensitivity.
    '''
    assert isinstance(lm_text_response, str) and lm_text_response != "", "lm_text_response must be a non-empty string"
    
    if 'true' in lm_text_response.lower():
        return 'True'
    elif 'false' in lm_text_response.lower():
        return 'False'
    else:
        return 'No_response'
