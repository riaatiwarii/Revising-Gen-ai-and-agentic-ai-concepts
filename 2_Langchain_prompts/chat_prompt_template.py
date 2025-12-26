from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


chat_template = ChatPromptTemplate([
  ('system', "You are a helpful {domain} expert "),
  ('human', " Explain in simple terms, what is {topic}")
  # SystemMessage(content="You are a helpful {domain} expert "),
  # HumanMessage(content=" Explain in simple terms, what is {topic}")

])

prompt = chat_template.invoke({
  'domain': 'cricket',
  'topic': 'best matches of all time'
})

print(prompt)