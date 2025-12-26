# we dont have money to use anthropic api right now
# we will only keep this file for reference

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

chat_model = ChatAnthropic(model="claude-2", temperature=0)

result = chat_model.invoke("what is the capital of india?")

print(result.content)
