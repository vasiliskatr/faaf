import re
from anthropic.types.message import Message
from pydantic._internal._model_construction import ModelMetaclass

def pydantic2list_of_parameters(model: ModelMetaclass):
    parameters=[]
    for param_name, param_properties in model.model_json_schema()['properties'].items():
        d = {"name": param_name}
        if 'anyOf' in param_properties.keys():
            types = {"anyOf": [{'type': 'string'},
                               {'type': 'null'}]}
        elif 'enum' in param_properties.keys():
            types = {'enum': [value for value in param_properties['enum']]}
        d.update(types)
        d["description"] = param_properties["description"]

        parameters.append(d)
    return parameters

def construct_format_tool_for_claude_prompt(parameters:list, name="fact_checker", description="Confirms or rejects facts from the passage."):

    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_claude_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )
    return constructed_prompt

def construct_format_parameters_claude_prompt(parameters:list):
    
    def handle_nested_list_of_dicts(l):
        return "\n".join(f"<{key}>{value}</{key}>" for item in l for key, value in item.items())

    def handle_nested_list_of_str(l):
        return "\n".join(f"<enum>{value}</enum>" for value in l)
    xml=[]
    for parameter in parameters:
        
        xml_chunk="".join(
            "<parameter>\n" + 
            "\n".join(f"<{key}>{value}</{key}>" if isinstance(value, str) 
                      else handle_nested_list_of_dicts(value) if all(isinstance(item, dict) for item in value) 
                      else handle_nested_list_of_str(value) for key, value in parameter.items()
            ) +
            "\n</parameter>")
        xml.append(xml_chunk)
    
    return "\n".join(chunk for chunk in xml)

def pydantic2claude_xml(model):

    parameters = pydantic2list_of_parameters(model)
    function_prompt = construct_format_tool_for_claude_prompt(parameters=parameters)
    return function_prompt


def construct_tool_use_system_prompt(tools):

    assert isinstance(tools, list), "tools must be a list of strings"
    
    tool_use_system_prompt = (
        "In this environment you have access to a set of tools you can use to answer the user's question.\n"
        "\n"
        "You may call them like this:\n"
        "<function_calls>\n"
        "<invoke>\n"
        "<tool_name>$TOOL_NAME</tool_name>\n"
        "<parameters>\n"
        "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
        "...\n"
        "</parameters>\n"
        "</invoke>\n"
        "</function_calls>\n"
        "\n"
        "Here are the tools available:\n"
        "<tools>\n"
        + '\n'.join([tool for tool in tools]) +
        "\n</tools>"
    )
    return tool_use_system_prompt


def extract_between_tags(tag: str, string: str, strip: bool = True) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list
    
def safe_flatten_list(l):
    if l:
        return l[0]
    else:
        return None
    
def llm2dict_claude(completion: Message, pydantic_model: ModelMetaclass):
    '''
    '''
    assert isinstance(completion, Message), "completion must be a anthropic.types.message.Message object"
    
    fact_keys = list(pydantic_model.__fields__.keys())

    response_text = completion.content[0].text

    response_dict={}
    for key in fact_keys:
        response_dict[key]=safe_flatten_list(extract_between_tags(tag=key,
                                                             string=response_text,
                                                             strip=True))
    return response_dict