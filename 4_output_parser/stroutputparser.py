from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

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

#Invoke 1st prompt
prompt1 = template1.invoke({ 'topic':'black holes'})
result = model.invoke(prompt1)

#Invoke 2nd prompt
prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)

print(result1.content)