import google.generativeai as genai
from langchain_experimental.tools import PythonREPLTool
import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph ,START ,END
from  langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langgraph.graph.message import add_messages
import io
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. DATA LOADING ---
# Ensure 'train.csv' is in the same directory as this script
try:
    df = pd.read_csv('first_half.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please ensure it is in the same directory.")
    exit()

# --------------------------------------
# INSERT YOUR GEMINI API KEY HERE
# --------------------------------------
genai.configure(api_key="AIzaSyB0pmTvTBZ_HHVmG9ujEB9PthfVWeulwX4")

# --- 2. DATA STATS FUNCTION ---
def dfstats(data):
    """Generates a comprehensive summary of the DataFrame as a single string."""
    output = []

    output.append("=== DATA INFO ===")
    buffer = io.StringIO()
    data.info(buf=buffer) 
    output.append(buffer.getvalue())

    output.append("\n=== DATA SHAPE ===")
    output.append(str(data.shape))

    output.append("\n=== DATA DESCRIPTION (Numeric) ===")
    output.append(str(data.describe()))

    output.append("\n=== MISSING VALUES (Count) ===")
    output.append(str(data.isnull().sum()))

    output.append("\n===== CATEGORICAL SUMMARY =====")
    # Using 'data' for consistency, though your original code used 'df'
    output.append(str(data.describe(include=['object']))) 

    output.append("\n===== MISSING VALUES (%) =====")
    output.append(str((data.isnull().sum() / len(data)) * 100))

    output.append("\n===== UNIQUE VALUES (Count) =====")
    output.append(str(data.nunique()))

    output.append("\n===== DUPLICATED ROWS (Count) =====")
    output.append(str(data.duplicated().sum()))

    output.append("\n===== COLUMN TYPES =====")
    output.append(str(data.dtypes))

    return "\n".join(output)


datastats = dfstats(df) 
print("Data statistics successfully calculated.")
# --- 3. MODEL INITIALIZATION AND REPL FIX ---

model = genai.GenerativeModel("gemini-2.5-flash")

# 🚀 FIX 1: Pass 'df' and 'pd' into the REPL's execution scope
# This prevents the "NameError: name 'df' is not defined" error.
python_repl = PythonREPLTool(
    globals={'df': df, 'pd': pd} 
)

# Memory only stores (role="user" or "model", text)
history = []


def build_messages():
    msgs = []
    for role, text in history:
        msgs.append({"role": role, "parts": [text]})
    return msgs


# --- 4. CONVERSATION LOOP ---
while True:
    user = input("\nYou: ")

    # Save user message
    history.append(("user", user))

    # STEP 1 — Ask Gemini for Python code
    # 🚀 FIX 2A: Stronger instruction to always use DATA_STATS for summaries
    code_prompt = f"""
You can control and use pythonRepl when needed.
The full dataset statistics are provided below. 
**CRITICAL INSTRUCTION:** If the user's request can be answered *entirely* by summarizing the information in DATA_STATS (e.g., 'What is the shape?', 'How many rows?', 'Missing values?'), you **MUST** respond with **ONLY** the text: **NO_CODE**. 
For any other request (e.g., filtering, calculating mean), generate Python code.

DATA_STATS:
{datastats}

Conversation so far:
{history}

User request: {user}

If Python is needed, respond ONLY with Python code.
If Python is NOT needed, respond EXACTLY: NO_CODE
"""

    ai = model.generate_content(code_prompt).text.strip()

    # Save model output (must use role="model")
    history.append(("model", ai))

    if ai == "NO_CODE":
        # 🚀 FIX 2B: Explicitly inject DATA_STATS into the chat history for 
        # the 'NO_CODE' response to ensure the model has context for summarization.
        
        # 1. Temporarily add the DATA_STATS as context for the next call
        # This acts as a highly relevant system instruction for this specific turn.
        temp_history = build_messages() + [
            {"role": "model", "parts": [f"DATA_STATS for context: {datastats}"]}
        ]
        
        # 2. Add the user's request
        temp_history.append({"role": "user", "parts": [user]})

        # 3. Get the reply using the full context
        reply = model.generate_content(temp_history).text
        
        print("Gemini:", reply)
        history.append(("model", reply))
        continue
    
    print("\n[Gemini generated code]:\n", ai)

    # STEP 2 — Execute Python
    try:
        # Code execution works now because df is in python_repl's globals
        result = python_repl.run(ai) 
    except Exception as e:
        result = f"Error: {e}"

    print("\n[Python REPL Result]:", result)

    # Save tool output as "model" (because Gemini sees it as internal context)
    history.append(("model", f"Python executed:\n{ai}\nResult:\n{result}"))

    # STEP 3 — Ask Gemini to explain result
    explanation_prompt = f"""
Explain this Python execution to the user.

Code:
{ai}

Result:
{result}
"""

    explanation = model.generate_content(
        build_messages() + [
            {"role": "user", "parts": [explanation_prompt]}
        ]
    ).text

    print("\nGemini:", explanation)

    history.append(("model", explanation))