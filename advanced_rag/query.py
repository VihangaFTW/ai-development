from util import load_and_get_key
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam



api_key = load_and_get_key()
client = OpenAI(api_key=api_key)


def llm_response_generator(query: str, model: str = 'gpt-4.1-nano') -> str:
    """
    Generate an augmented query using OpenAI to provide example answers for financial research.
    
    Args:
        query (str): The original query to augment
        model (str): The OpenAI model to use for generation
        
    Returns:
        str: The generated augmented response
        
    Raises:
        ValueError: If the response content is None or empty
    """
    system_prompt = """You are a helpful expert financial research assistant. 
    Provide an example answer to the given question, that might be found in a document like an annual report."""
    
    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    response = client.chat.completions.create(
        model=model, 
        messages=messages
    )
    
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI API returned None content")
    
    return content
