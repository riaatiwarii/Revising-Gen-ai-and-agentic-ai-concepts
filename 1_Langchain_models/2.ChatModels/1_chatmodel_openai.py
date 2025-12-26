from  langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

result = chat_model.invoke("what is the capital of india?")
print(result.content)

#temperature - parameter that controls the randomness of the model's output.
# A temperature of 0 makes the output more deterministic and focused, while higher values (like 0.7 or 1) introduce more randomness and creativity.
# Factual answer(maths,code,facts) = 0.0-0.3
# balanced(general QA, explainations) = 0.5-0.7
# creative(content generation, storytelling) = 0.9-1.2
# maximum randomness(wild ideas, brainstorming) = 1.5+