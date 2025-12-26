from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

st.header('Research tool')

user_input = st.text_input("Enter your research question:")


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=200,
    temperature=0.7
)

chat_model = ChatHuggingFace(llm=llm)

if st.button("Search"):
  result = chat_model.invoke(user_input)
  st.text(result.content)

# response = chat_model.invoke("What is quantum computing?")
# print(response.content)
