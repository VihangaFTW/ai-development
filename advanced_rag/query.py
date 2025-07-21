from util import load_and_get_key
from openai import OpenAI

api_key = load_and_get_key()
client = OpenAI(api_key=api_key)


