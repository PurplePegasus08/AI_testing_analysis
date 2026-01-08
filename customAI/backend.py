# agent_core.py  –  no web dependencies at all
import os, io, uuid, json, re
import pandas as pd
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing Gemini API key")

try:
    from langchain_experimental.tools.python.tool import PythonREPLTool
    from langchain_experimental.utilities import PythonREPL
    _HAS_LANGCHAIN_PY_REPL = True
except Exception:
    PythonREPLTool = None
    PythonREPL = None
    _HAS_LANGCHAIN_PY_REPL = False

# ---------- memory ----------
class MemStore:
    def __init__(self): self._db: dict[str, bytes] = {}
    def write_df(self, df: pd.DataFrame) -> str:
        key = str(uuid.uuid4())
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._db[key] = buf.getvalue()
        return key
    def get_df(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self._db[key]))
store = MemStore()

# ---------- state ----------
class UndoEntry(BaseModel):
    description: str
    snapshot: bytes

class AgentState(BaseModel):
    raw_id: str = ""
    work_id: str = ""
    history: List[UndoEntry] = Field(default_factory=list)
    next_node: Literal["upload", "eda", "human_input", "execute", "undo", "export", "END"] = "upload"
    user_message: str = ""
    error: Optional[str] = None
    export_filename: str = "cleaned.csv"
    retry_count: int = 0
    MAX_RETRIES: int = 3

    def push_undo(self, desc: str):
        if self.work_id and self.work_id in store._db:
            self.history.append(UndoEntry(description=desc, snapshot=store._db[self.work_id]))
    def undo(self) -> str:
        if not self.history: return "Nothing to undo."
        entry = self.history.pop()
        key = str(uuid.uuid4())
        store._db[key] = entry.snapshot
        self.work_id = key
        return f"Undone: {entry.description}"

# ---------- LLM ----------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1, google_api_key=API_KEY)

def compute_stats(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    missing = df.isna().sum()
    return f"""
Shape: {df.shape}
Columns & Types:
{buf.getvalue()}
Missing:
{missing[missing>0] if missing.sum()>0 else "None"}
Numeric:
{df.describe() if len(df.select_dtypes(include=['number']).columns)>0 else "No numeric"}
Categorical:
{df.describe(include=['object']) if len(df.select_dtypes(include=['object']).columns)>0 else "No categorical"}
""".strip()

def get_stats(work_id: str) -> str:
    if not work_id: return "No data."
    try: return compute_stats(store.get_df(work_id))
    except Exception as e: return f"Stats error: {e}"

# ---------- code exec ----------
SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "range": range, "enumerate": enumerate, "zip": zip}
def exec_code(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    try:
        loc = {"df": df.copy(), "pd": pd}
        exec(code, {"__builtins__": SAFE_BUILTINS}, loc)
        if "df" not in loc: return df, "Missing df variable"
        return loc["df"], None
    except Exception as e:
        return df, f"{type(e).__name__}: {e}"

def run_code_with_langchain_repl(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str], str]:
    if not _HAS_LANGCHAIN_PY_REPL:
        new_df, err = exec_code(code, df)
        return new_df, err, ""
    try:
        repl = PythonREPL(locals={"df": df.copy(), "pd": pd})
        tool = PythonREPLTool(python_repl=repl)
        output = tool.run(code) or ""
        new_df = getattr(repl, "locals", {}).get("df")
        if not isinstance(new_df, pd.DataFrame):
            return df, "Missing df variable", output
        return new_df, None, output
    except Exception as e:
        return df, f"{type(e).__name__}: {e}", ""

# ---------- prompt ----------
def build_prompt(state: AgentState) -> str:
    err = f"\nError: {state.error}\nFix it." if state.error else ""
    retry = f"\nRetry {state.retry_count+1}/{state.MAX_RETRIES}" if state.retry_count else ""
    return f"""You are a data assistant. Output ONLY JSON.
Current data:
{get_stats(state.work_id)}
User: "{state.user_message}"
{err}{retry}
If you choose action "code", output Python code that uses pandas and the variable df (a DataFrame). Keep the final DataFrame assigned to df. You may print intermediate results.
JSON: {{"action": "answer"|"code"|"clarify", "content": "..."}}
""".strip()

# ---------- nodes ----------
def upload_node(state: AgentState, file_content: bytes) -> AgentState:
    df = pd.read_csv(io.StringIO(file_content.decode("utf-8")))
    state.raw_id = store.write_df(df)
    state.work_id = state.raw_id
    state.user_message = f"Loaded {len(df)} rows × {len(df.columns)} cols"
    state.next_node = "eda"
    return state

def eda_node(state: AgentState) -> AgentState:
    state.user_message = get_stats(state.work_id)
    state.next_node = "human_input"
    return state

def human_input_node(state: AgentState) -> AgentState:
    txt = (state.user_message or "").strip()
    low = txt.lower()
    if low in {"undo", "/undo"}:
        state.next_node = "undo"
        return state
    if low.startswith("export"):
        parts = txt.split(maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            state.export_filename = parts[1].strip()
        state.next_node = "export"
        return state
    state.next_node = "execute"
    return state

def execute_node(state: AgentState) -> AgentState:
    prompt = build_prompt(state)
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        m = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.S) or re.search(r'(\{.*?\})', raw, re.S)
        if not m: raise ValueError("No JSON")
        res = json.loads(m.group(1))
        action, content = res.get("action"), res.get("content", "")
        if action == "answer":
            state.user_message, state.error, state.retry_count = content, None, 0
            state.next_node = "human_input"
        elif action == "clarify":
            state.user_message, state.error, state.retry_count = content, None, 0
            state.next_node = "human_input"
        elif action == "code":
            state.push_undo(f"Code: {content[:60]}...")
            df = store.get_df(state.work_id)
            new_df, err, output = run_code_with_langchain_repl(content, df)
            if err:
                state.error, state.retry_count = err, state.retry_count + 1
                state.user_message = f"Error: {err}"
                state.next_node = "execute" if state.retry_count < state.MAX_RETRIES else "human_input"
            else:
                state.work_id = store.write_df(new_df)
                msg = "✅ Success"
                if output.strip():
                    msg = msg + "\n\n" + output.strip()
                state.user_message, state.error, state.retry_count = msg, None, 0
                state.next_node = "eda"
        else: raise ValueError(f"Unknown action {action}")
    except Exception as e:
        state.error, state.retry_count = str(e), state.retry_count + 1
        state.user_message = f"Error: {state.error}"
        state.next_node = "execute" if state.retry_count < state.MAX_RETRIES else "human_input"
    return state

def undo_node(state: AgentState) -> AgentState:
    state.user_message = state.undo()
    state.next_node = "eda"
    return state

def export_node(state: AgentState) -> AgentState:
    try:
        df = store.get_df(state.work_id)
        df.to_csv(state.export_filename, index=False)
        state.user_message = f"Saved {state.export_filename}"
    except Exception as e:
        state.user_message = f"Export failed: {e}"
    state.next_node = "human_input"
    return state

# ---------- graph builder ----------
def build_graph():
    from langgraph.graph import StateGraph, START, END
    w = StateGraph(AgentState)
    w.add_node("upload", lambda s, fc: upload_node(s, fc))
    w.add_node("eda", eda_node)
    w.add_node("execute", execute_node)
    w.add_node("undo", undo_node)
    w.add_node("export", export_node)
    w.add_node("human_input", human_input_node)
    w.add_edge(START, "upload")
    w.add_edge("upload", "eda")
    w.add_edge("eda", "human_input")
    w.add_edge("undo", "eda")
    w.add_edge("export", "human_input")
    w.add_conditional_edges("human_input", lambda s: s.next_node,
        {"execute": "execute", "undo": "undo", "export": "export", "END": END})
    w.add_conditional_edges("execute", lambda s: s.next_node,
        {"human_input": "human_input", "execute": "execute", "eda": "eda"})
    return w.compile()
