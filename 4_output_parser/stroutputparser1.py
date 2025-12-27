from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="HuggingFaceH4/zephyr-7b-beta",
  task="text-generation",
  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
  max_new_tokens=200,
  temperature=0.7
)

model = ChatHuggingFace(llm=llm, temperature=0)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template ="Generate a detailed report on {topic} ",
    input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template ="Write 5 line summary on the following text: \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Climate Change"})

print(result)