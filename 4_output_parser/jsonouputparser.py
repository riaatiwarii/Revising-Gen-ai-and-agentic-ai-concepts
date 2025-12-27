from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="HuggingFaceH4/zephyr-7b-beta",
  task="text-generation",
  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
  max_new_tokens=200,
  temperature=0.7
)

model = ChatHuggingFace(llm=llm, temperature=0)

parser = JsonOutputParser()

template = PromptTemplate(
  template =" Give me a name, age and city of a fictional person \n {fomat_instructions}",
  input_variables = [],
  partial_variables={"fomat_instructions": parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
print(type(final_result))