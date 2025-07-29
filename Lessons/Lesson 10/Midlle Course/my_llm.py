from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
# Create the LLM Chain - Option A: map_reduce Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)