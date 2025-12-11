import os
import pandas as pd
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage # For clarity if you use chat history
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.document_loaders import CSVLoader,JSONLoader,UnstructuredCSVLoader,UnstructuredExcelLoader


load_dotenv() # Load environment variables from .env file


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






