import os
import pandas as pd
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_pandas_dataframe_agent     # ← FIXED
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
#AIzaSyB0pmTvTBZ_HHVmG9ujEB9PthfVWeulwX4
# --- 1. Configuration ---
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ API Key not found.")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
    allow_dangerous_code=True
)

# --- 2. Load CSV ---
file_path = "train.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ File '{file_path}' not found")

# --- 3. Create Pandas Agent ---
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True   # allows Python execution
)

# --- 4. Run the Analysis ---
analysis_prompt = """
what is average of the age?.
"""

print("\n--- Running LangChain Agent ---")
response = agent.invoke({"input": analysis_prompt})

print("\n--- Final Answer ---")
print(response["output"].content)
