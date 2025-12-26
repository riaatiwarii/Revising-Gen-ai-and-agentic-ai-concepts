from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="chat-completions",
    temperature=0,
    max_new_tokens=50,
)

result = llm.invoke("What is the capital of India?")
print(result)


# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=-1  # CPU
# )

# prompt = "<|user|>What is the capital of India?<|assistant|>"

# result = pipe(prompt, max_new_tokens=50, do_sample=True)
# print(result)

