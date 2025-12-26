from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatOpenAI()

#schema

# class ReviewSummary(TypedDict):
#   summary: Annotated[str, "A brief summary of the review"]
#   sentiment: Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]

class ReviewSummary(TypedDict):
  key_themes: Annotated[list[str], "A list of key themes mentioned in the review"]
  summary : Annotated[str, "A brief summary of the review"]
  sentiment: Annotated[str, "The overall sentiment of the review, e.g., positive, negative, neutral"]
  pros : Annotated[Optional[list[str]], "A list of pros mentioned in the review, if any"]
  cons : Annotated[Optional[list[str]], "A list of cons mentioned in the review, if any"]

structured_model = model.with_structured_output(ReviewSummary)

# result =  structured_model.invoke("""The hardware is great, but the software feels bloated. There are too
#                        many pre-installed applications that I can't remove. Also, the UI outdated compared to other
#                        brands. Hoping for a software update to improve performance and user experience.""")


result = structured_model.invoke('''I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh ''')

print(result)  # type: ReviewSummary