from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts ='Delhi is the capital of India.'

vector_embedding = embeddings.embed_query(texts)


print(str(vector_embedding))