from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents =[
  "virat kohli is a great cricketer.",
  "sachin tendulkar is a legendary batsman.",
  "mumbai is the financial capital of india.",
  "dhoni is a successful cricket captain."
]

query = "tell me about virat kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# print("Similarity Scores:", similarities)

index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score:", score)