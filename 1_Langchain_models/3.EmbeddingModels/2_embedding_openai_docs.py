from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents =[
    "Quantum computing is a type of computation that leverages the principles of quantum mechanics to process information.",
    "kolkata is the capital of west bengal",
    "paris is the capital of france"
]

result = embedding.embed_documents(documents)

print(str(result))