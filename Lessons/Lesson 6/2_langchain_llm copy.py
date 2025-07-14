from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from gradio_chats.gradio_chat import start_gradio
import gradio as gr

#Set your OpenAI key in the .env file
load_dotenv()

# Create an OpenAI chat model instance
chat_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    )

# Create a prompt with a simple system rule and a dynamic user prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates rhyming words."),
    ("user", "{input}"),
])

# Chain the prompt, model, and string output parser
rhyme_chain = prompt | chat_llm | StrOutputParser()

# Run the chain with a sample input
result = rhyme_chain.invoke("Hello World!")

print(result)

start_gradio(rhyme_chain, "Hello World!")
