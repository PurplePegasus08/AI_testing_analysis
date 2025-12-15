import os
import io
import json
import uuid
import pandas as pd
from dotenv import load_dotenv

# --- LangChain / LangGraph ---
from langchain_experimental.tools import PythonREPLTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Your models ---
from modelState import AgentState, UndoEntry, DataStore

# ---------------- ENV ----------------

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found")

# ---------------- LLM ----------------

python_repl = PythonREPLTool()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
    allow_dangerous_code=True,
)

llm.bind_tools([python_repl], tool_choice="auto")

# ---------------- MEMSTORE ----------------

class Memstore(DataStore):
    def __init__(self):
        self._db: dict[str, bytes] = {}

    def write_df(self, df: pd.DataFrame) -> str:
        key = str(uuid.uuid4())
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._db[key] = buf.getvalue()
        return key

    def to_parquet(self, data_id: str) -> bytes:
        return self._db[data_id]

    def from_parquet(self, blob: bytes) -> str:
        key = str(uuid.uuid4())
        self._db[key] = blob
        return key

    def get_df(self, data_id: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self._db[data_id]))

store = Memstore()

# ---------------- NODES ----------------

def load_node(state: AgentState) -> AgentState:
    df = pd.read_csv("train.csv")
    state.raw_id = store.write_df(df)
    state.work_id = state.raw_id
    state.user_message = "File uploaded"
    state.next_node = "eda"
    return state

def eda_node(state: AgentState) -> AgentState:
    df = store.get_df(state.work_id)
    state.user_message = f"Shape {df.shape} Missing: {df.isna().sum().to_dict()}"
    state.next_node = "llm_suggest"
    return state

def llm_suggest_node(state: AgentState) -> AgentState:
    df_head = store.get_df(state.work_id).head().to_csv(index=False)

    prompt = f"""
You see this dataframe head:
{df_head}
Return ONE line of pandas code that fills missing Age with the median.
Output ONLY valid Python.
"""

    ai_msg = llm.invoke(prompt)
    state.generated_code = ai_msg.content.strip().strip("```")
    state.user_message = f"Generated code: {state.generated_code}"
    state.next_node = "execute"
    return state

def execute_node(state: AgentState) -> AgentState:
    df_old = store.get_df(state.work_id)

    state.history.append(
        UndoEntry(
            op="exec",
            params={"code": state.generated_code},
            description="Execute pandas code",
            mask_bytes=store.to_parquet(state.work_id),
        )
    )

    loc = {"df": df_old, "pd": pd}

    try:
        exec(state.generated_code, loc)
        new_key = store.write_df(loc["df"])
        state.work_id = new_key
        state.user_message = "Execution successful"
    except Exception as e:
        state.user_message = f"Execution error: {e}"

    state.next_node = "human_review"
    return state

def undo_node(state: AgentState) -> AgentState:
    if state.history:
        entry = state.history.pop()
        state.work_id = store.from_parquet(entry.mask_bytes)
        state.user_message = "Undo successful"
    else:
        state.user_message = "Nothing to undo"

    state.next_node = "human_review"
    return state

def export_node(state: AgentState) -> AgentState:
    store.get_df(state.work_id).to_csv("cleaned.csv", index=False)
    state.user_message = "Saved cleaned.csv"
    state.next_node = "END"
    return state

# ---------------- GRAPH ----------------

workflow = StateGraph(AgentState)

workflow.add_node("load", load_node)
workflow.add_node("eda", eda_node)
workflow.add_node("llm_suggest", llm_suggest_node)
workflow.add_node("execute", execute_node)
workflow.add_node("undo", undo_node)
workflow.add_node("export", export_node)
workflow.add_node("human_review", lambda s: s)

workflow.add_edge(START, "load")
workflow.add_edge("load", "eda")
workflow.add_edge("eda", "llm_suggest")
workflow.add_edge("llm_suggest", "execute")
workflow.add_edge("execute", "human_review")
workflow.add_edge("undo", "human_review")
workflow.add_edge("export", END)

workflow.add_conditional_edges(
    "human_review",
    lambda s: s.next_node,
    {
        "execute": "execute",
        "undo": "undo",
        "export": "export",
        "END": END,
    },
)

graph = workflow.compile(checkpointer=InMemorySaver())

# ---------------- MAIN ---------------- 

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "demo"}}
    final = graph.invoke(AgentState(), thread)
    print(final.user_message)
