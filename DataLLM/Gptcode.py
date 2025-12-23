# chat_data_agent_tool_based_fixed.py
from __future__ import annotations
import os, io, uuid
import pandas as pd
from typing import List, Optional, Protocol, Literal
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Load dataset ---
df = pd.read_csv("train.csv")

# --- Load API key ---
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing Gemini API key")

# --- Agent State ---
class UndoEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    mask_bytes: bytes

class DataStore(Protocol):
    def to_parquet(self, data_id: str) -> bytes: ...
    def from_parquet(self, blob: bytes) -> str: ...

class AgentState(BaseModel):
    raw_id: str = ""
    work_id: str = ""
    history: List[UndoEntry] = Field(default_factory=list)
    next_node: Literal["upload","eda","chat","repl_execute","undo","export"] = "upload"
    user_message: str = ""
    error: Optional[str] = None

    def push_undo(self, store: DataStore, desc: str):
        self.history.append(UndoEntry(description=desc, mask_bytes=store.to_parquet(self.work_id)))

    def undo(self, store: DataStore):
        if not self.history:
            self.user_message = "Nothing to undo."
            return
        entry = self.history.pop()
        self.work_id = store.from_parquet(entry.mask_bytes)
        self.user_message = f"Undone: {entry.description}"

    @field_validator("raw_id")
    @classmethod
    def _immutable_raw(cls, v, info):
        if info.data.get("raw_id") and v != info.data["raw_id"]:
            raise ValueError("raw_id is immutable")
        return v

# --- In-memory store ---
class MemStore:
    def __init__(self):
        self._db: dict[str, bytes] = {}
    def write_df(self, df: pd.DataFrame) -> str:
        key = str(uuid.uuid4())
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._db[key] = buf.getvalue()
        return key
    def get_df(self, data_id: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self._db[data_id]))
    def to_parquet(self, data_id: str) -> bytes:
        return self._db[data_id]
    def from_parquet(self, blob: bytes) -> str:
        key = str(uuid.uuid4())
        self._db[key] = blob
        return key

store = MemStore()

# --- Python REPL tool ---
@tool
def python_repl(code: str, data_id: str) -> str:
    """Execute Python code that modifies `df` in-place."""
    df = store.get_df(data_id)
    local_vars = {"df": df, "pd": pd}
    allowed_builtins = {"len": len, "sum": sum, "min": min, "max": max, "print": print}
    exec(code, {"__builtins__": allowed_builtins}, local_vars)
    new_df = local_vars["df"]
    return store.write_df(new_df)

# --- LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=API_KEY,
).bind_tools([python_repl])

# --- Data stats ---
def dfstats(data):
    output = []
    buf = io.StringIO()
    output.append("=== DATA INFO ===")
    data.info(buf=buf)
    output.append(buf.getvalue())
    output.append("\n=== DATA SHAPE ===")
    output.append(str(data.shape))
    output.append("\n=== NUMERIC DESCRIPTION ===")
    output.append(str(data.describe()))
    output.append("\n=== MISSING VALUES ===")
    output.append(str(data.isnull().sum()))
    output.append("\n=== CATEGORICAL DESCRIPTION ===")
    output.append(str(data.describe(include=['object'])))
    return "\n".join(output)

datastats = dfstats(df)
print("Data statistics successfully calculated.")

# --- Nodes ---
def upload_node(state: AgentState) -> AgentState:
    df_local = pd.read_csv("train.csv")
    state.raw_id = store.write_df(df_local)
    state.work_id = state.raw_id
    state.user_message = "Dataset uploaded. What would you like to do?"
    return state

def eda_node(state: AgentState) -> AgentState:
    df_local = store.get_df(state.work_id)
    state.user_message = f"Shape: {df_local.shape}\nMissing values:\n{df_local.isna().sum()}\nYou can now chat with me."
    return state

def chat_node(state: AgentState) -> AgentState:
    msg = input("\nUser: ").strip()
    state.user_message = msg
    text = msg.lower()
    if "undo" in text:
        state.next_node = "undo"
    elif "export" in text or "save" in text:
        state.next_node = "export"
    else:
        state.next_node = "repl_execute"
    return state

def repl_execute_node(state: AgentState) -> AgentState:
    prompt = f"""
You are a friendly data assistant. 
Answer user requests naturally. 
You have precomputed DATA_STATS of the dataset:

{datastats}

User request:
{state.user_message}

- If the answer can be given using DATA_STATS, answer directly.
- If Python code is needed, generate code using python_repl.
- Always provide friendly chat responses, never "Done. What next?"
"""
    response = llm.invoke([HumanMessage(content=prompt)])

    if response.tool_calls:
        call = response.tool_calls[0]
        state.push_undo(store, f"User request: {state.user_message}")
        new_id = python_repl.invoke(call["args"])
        state.work_id = new_id
        state.user_message = f"Python code executed successfully. Ask me anything next!"
    else:
        state.user_message = response.content

    state.next_node = "chat"
    print("\nAssistant:", state.user_message)
    return state

def undo_node(state: AgentState) -> AgentState:
    state.undo(store)
    state.next_node = "chat"
    print("\nAssistant:", state.user_message)
    return state

def export_node(state: AgentState) -> AgentState:
    df_local = store.get_df(state.work_id)
    df_local.to_csv("cleaned.csv", index=False)
    state.user_message = "Saved cleaned.csv. Anything else?"
    state.next_node = "chat"
    print("\nAssistant:", state.user_message)
    return state

# --- Build graph ---
workflow = StateGraph(AgentState)
workflow.add_node("upload", upload_node)
workflow.add_node("eda", eda_node)
workflow.add_node("chat", chat_node)
workflow.add_node("repl_execute", repl_execute_node)
workflow.add_node("undo", undo_node)
workflow.add_node("export", export_node)

workflow.add_edge(START, "upload")
workflow.add_edge("upload", "eda")
workflow.add_edge("eda", "chat")
workflow.add_conditional_edges("chat", lambda s: s.next_node, {"repl_execute": "repl_execute", "undo": "undo", "export": "export"})
workflow.add_edge("repl_execute", "chat")
workflow.add_edge("undo", "chat")
workflow.add_edge("export", "chat")

graph = workflow.compile()

# --- Main ---
if __name__ == "__main__":
    print("\n=== CHAT DATA AGENT (REAL TOOL MODE) ===")
    print("Examples:\n- handle missing values\n- encode categorical columns\n- undo last step\n- export\n")
    graph.invoke(AgentState())
