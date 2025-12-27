from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="HuggingFaceH4/zephyr-7b-beta",
  task="text-generation",
  huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
  max_new_tokens=60,
  temperature=0.7
)

model = ChatHuggingFace(llm=llm, temperature=0)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative","Neutral"] = Field(description = "Give me sentiment of the feedback")
    
parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
  template = """
  You are a JSON generator.

  Return ONLY valid JSON. No explanations.

  {format_instructions}

  Review: {feedback}
  """,
  input_variables=["feedback"],
  partial_variables = {'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'write an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'write an appropriate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt4 = PromptTemplate(
    template = 'write an appropriate response to this neutral feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser),
    (lambda x:x.sentiment == 'Neutral', prompt4 | model | parser),
    RunnableLambda (lambda x: "could not find sentiment" )
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':"this is a terrible phone"})

print(result)

chain.get_graph().print_ascii()