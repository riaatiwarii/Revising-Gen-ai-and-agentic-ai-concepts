from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="HuggingFaceH4/zephyr-7b-beta",
  task="text-generation",
  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
  max_new_tokens=200,
  temperature=0.7
)

model = ChatHuggingFace(llm=llm, temperature=0)

class person(BaseModel):
  name : str = Field(description="The name of the person")
  age : int = Field(gt= 18, description="age greater than 18")
  city : str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
  template = """
You MUST follow ALL rules below:

- Return ONLY ONE valid JSON object
- Do NOT return schema
- Do NOT explain anything
- Do NOT add text before or after JSON
- Use fictional but realistic values

{format_instructions}

Generate a fictional {place} person.
""",
  input_variables = ["place"],
  partial_variables = {"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format(place="Canadian")

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({"place": "Canadian"})

print(final_result)