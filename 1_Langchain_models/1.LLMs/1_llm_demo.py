from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = OpenAI(model="gpt-4o-mini", temperature=0)

result = llm.invoke("what is the capital of india?")

print(result)