import os
import pandas as pd
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_experimental.tools import PythonREPLTool
import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph ,START ,END
from  langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langgraph.graph.message import add_messages
import io
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() # Load environment variables from .env file

try:
    df = pd.read_csv('first_half.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please ensure it is in the same directory.")
    exit()

# Ensure API Key is available
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY.")

# Initialize the LLM with your settings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # Use 'flash' for speed, or 'pro' for better reasoning on code
    temperature=0,
    google_api_key=api_key,
    allow_dangerous_code=True
)






