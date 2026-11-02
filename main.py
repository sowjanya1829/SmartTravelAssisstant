import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

#load values from the .env file
load_dotenv()

#create the groq language model
llm=ChatGroq(
    model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),
    temperature=0
)

#Define the instructions and user question

messages=[
    SystemMessage(content="You are a helpful AI travel planning assistant."),
    HumanMessage(content="Plan a 2-day budget-friendly trip to goa.")
]

#send the request to Groq
response=llm.invoke(messages)

#display the model's response
print(response.content)