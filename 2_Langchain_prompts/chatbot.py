from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=200,
    temperature=0.7
)
chat_model = ChatHuggingFace(llm=llm)

chat_history = [
  SystemMessage(content="You are a helpful assistant.")
]

while True :
  user_input = input("User: ")
  chat_history.append(HumanMessage(content=user_input))
  if user_input == "exit":
    break
  result = chat_model.invoke(chat_history)
  chat_history.append(AIMessage(content=result.content))
  print("Bot:", result.content)

print(chat_history)
# response = chat_model.invoke("What is the capital of india?")
# print(response.content)