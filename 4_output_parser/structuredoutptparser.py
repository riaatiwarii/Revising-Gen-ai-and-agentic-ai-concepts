# it is no longer working because of the new huggingface endpoint changes

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="HuggingFaceH4/zephyr-7b-beta",
  task="text-generation",
  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
  max_new_tokens=200,
  temperature=0.7
)

model = ChatHuggingFace(llm=llm, temperature=0)

schmea = [
  ResponseSchema(name="fact1", description="fact1 about the topic"),
  ResponseSchema(name="fact2", description="fact2 about the topic"),
  ResponseSchema(name="fact3", description="fact3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schmea)

template = PromptTemplate(
  template =' give 3 facts abot {topic} in the below format:\n{format_instructions}',
  input_variables=["topic"],
  partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({'topic':"black hole"})

response = model.invoke(prompt)

final_result = parser.parse(response.content)

print(final_result)