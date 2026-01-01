# model.py – Agentic Data Analysis with Proper LangGraph
import os, io, uuid, json, re
import pandas as pd
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ========================= CONFIGURATION =========================
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
SOURCE_CSV = "train.csv"

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing Gemini API key")

# ========================= DATA STORE =========================
class MemStore:
    def __init__(self):
        self._db: dict[str, bytes] = {}
    
    def write_df(self, df: pd.DataFrame) -> str:
        key = str(uuid.uuid4())
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._db[key] = buf.getvalue()
        return key
    
    def get_df(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self._db[key]))

store = MemStore()

# ========================= STATE =========================
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

    @field_validator("raw_id")
    @classmethod
    def raw_id_immutable(cls, v, info):
        if info.data.get("raw_id") and v != info.data["raw_id"]:
            raise ValueError("raw_id is immutable")
        return v

    def push_undo(self, store: MemStore, desc: str):
        if self.work_id and self.work_id in store._db:
            self.history.append(UndoEntry(description=desc, snapshot=store._db[self.work_id]))

    def undo(self, store: MemStore):
        if not self.history:
            return "Nothing to undo."
        entry = self.history.pop()
        key = str(uuid.uuid4())
        store._db[key] = entry.snapshot
        self.work_id = key
        return f"Undone: {entry.description}"

# ========================= LLM & STATS =========================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1, google_api_key=API_KEY)

def compute_stats(df: pd.DataFrame) -> str:
    """Generate current dataframe statistics"""
    buf = io.StringIO()
    df.info(buf=buf)
    missing = df.isna().sum()
    return f"""
📊 DATAFRAME STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shape: {df.shape} | Memory: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB

Columns & Types:
{buf.getvalue()}

Missing Values:
{missing[missing>0] if missing.sum()>0 else "None"}

Numeric Summary:
{df.describe() if len(df.select_dtypes(include=['number']).columns)>0 else "No numeric columns"}

Categorical Summary:
{df.describe(include=['object']) if len(df.select_dtypes(include=['object']). columns)>0 else "No categorical columns"}
"""

def get_current_stats(work_id: str) -> str:
    if not work_id:
        return "No data loaded."
    try:
        return compute_stats(store.get_df(work_id))
    except Exception as e:
        return f"Stats error: {e}"

# ========================= CODE EXECUTION =========================
SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max,
    "range": range, "enumerate": enumerate, "zip": zip,
}

def execute_with_retry(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """Execute code with error handling"""
    try:
        local_vars = {"df": df.copy(), "pd": pd}
        exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
        if "df" not in local_vars:
            return df, "Code completed but 'df' variable is missing"
        return local_vars["df"], None
    except Exception as e:
        return df, f"{type(e).__name__}: {e}"

# ========================= PROMPT BUILDER =========================
def build_prompt(state: AgentState) -> str:
    error_ctx = f"\n⚠️ PREVIOUS ATTEMPT FAILED:\nError: {state.error}\nPlease fix the code." if state.error else ""
    retry_ctx = f"\n🔄 Retry {state.retry_count+1}/{state.MAX_RETRIES}" if state.retry_count > 0 else ""

    return f"""You are a precise data analysis assistant. Output ONLY JSON.

CURRENT DATA:
{get_current_stats(state.work_id)}

USER REQUEST: "{state.user_message}"
{error_ctx}{retry_ctx}

OUTPUT JSON FORMAT:
{{"action": "answer"|"code"|"clarify", "content": "..."}}

RULES:
- "answer": plain-text explanation when no code needed
- "code": valid pandas code that modifies df (must include `df = ...`)
- "clarify": ask user for missing details
- Handle missing values appropriately based on column dtype
- Do NOT wrap content in markdown

EXAMPLES:
1. {{"action": "answer", "content": "Age has 177 missing values (19.9%). Median is 28.0."}}
2. {{"action": "code", "content": "df['Age'] = df['Age'].fillna(df['Age'].median())"}}
3. {{"action": "clarify", "content": "Which imputation method? mean, median, or mode?"}}

RESPOND WITH ONLY JSON:
"""

# ========================= NODES =========================
def upload_node(state: AgentState) -> AgentState:
    df = pd.read_csv(SOURCE_CSV)
    state.raw_id = store.write_df(df)
    state.work_id = state.raw_id
    state.user_message = f"✅ Loaded '{SOURCE_CSV}' | Shape: {df.shape}"
    state.next_node = "eda"
    return state

def eda_node(state: AgentState) -> AgentState:
    state.user_message = get_current_stats(state.work_id)
    state.next_node = "human_input"
    return state

def human_input_node(state: AgentState) -> AgentState:
    print(f"\n{state.user_message}")
    msg = input("\n👤 User: ").strip()
    
    if not msg:
        state.user_message = "Please enter a valid command."
        return state
    
    state.user_message = msg
    state.error = None
    state.retry_count = 0
    
    # Parse filename
    if match := re.search(r'\b(\w+\.csv)\b', msg, re.I):
        state.export_filename = match.group(1)
    
    cmd = msg.lower()
    state.next_node = (
        "undo" if "undo" in cmd else
        "export" if "export" in cmd or "save" in cmd else
        "END" if cmd in ["exit", "quit"] else
        "execute"
    )
    return state

def execute_node(state: AgentState) -> AgentState:
    """Agentic execution with automatic retry on failure"""
    prompt = build_prompt(state)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        # Extract JSON
        content = response.content.strip()
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.S) or re.search(r'(\{.*?\})', content, re.S)
        if not json_match:
            raise ValueError("No valid JSON found in LLM response")
        
        result = json.loads(json_match.group(1))
        action, content = result.get("action"), result.get("content", "")
        
        if action == "answer":
            state.user_message = f"🤖 {content}"
            state.error, state.retry_count = None, 0
            state.next_node = "human_input"
            
        elif action == "clarify":
            state.user_message = f"❓ {content}"
            state.error, state.retry_count = None, 0
            state.next_node = "human_input"
            
        elif action == "code":
            state.push_undo(store, f"Code: {content[:60]}...")
            df = store.get_df(state.work_id)
            
            modified_df, exec_error = execute_with_retry(content, df)
            
            if exec_error:
                state.error = exec_error
                state.retry_count += 1
                
                if state.retry_count >= state.MAX_RETRIES:
                    state.user_message = f"❌ Failed after {state.MAX_RETRIES} attempts. Last error: {exec_error}"
                    state.error, state.retry_count = None, 0
                    state.next_node = "human_input"
                else:
                    state.next_node = "execute"  # Auto-retry
            else:
                state.work_id = store.write_df(modified_df)
                state.user_message = f"✅ Success: {content}"
                state.error, state.retry_count = None, 0
                state.next_node = "eda"  # Show updated stats
                
        else:
            raise ValueError(f"Unknown action: {action}")
            
    except Exception as e:
        state.error = f"LLM parsing error: {e}"
        state.retry_count += 1
        state.next_node = "execute" if state.retry_count < state.MAX_RETRIES else "human_input"
    
    print(f"\n{state.user_message}")
    if state.error:
        print(f"⚠️ Error: {state.error}")
    
    return state

def undo_node(state: AgentState) -> AgentState:
    msg = state.undo(store)
    state.user_message = f"⏪ {msg}"
    state.next_node = "eda"
    return state

def export_node(state: AgentState) -> AgentState:
    try:
        df = store.get_df(state.work_id)
        df.to_csv(state.export_filename, index=False)
        state.user_message = f"💾 Saved '{state.export_filename}' | Shape: {df.shape}"
    except Exception as e:
        state.user_message = f"❌ Export failed: {e}"
    
    state.next_node = "human_input"
    return state

# ========================= GRAPH =========================
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("upload", upload_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("human_input", human_input_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("undo", undo_node)
    workflow.add_node("export", export_node)
    
    workflow.add_edge(START, "upload")
    workflow.add_edge("upload", "eda")
    workflow.add_edge("eda", "human_input")
    workflow.add_edge("undo", "eda")
    workflow.add_edge("export", "human_input")
    
    workflow.add_conditional_edges(
        "human_input",
        lambda s: s.next_node,
        {"execute": "execute", "undo": "undo", "export": "export", "END": END}
    )
    
    workflow.add_conditional_edges(
        "execute",
        lambda s: s.next_node,
        {"human_input": "human_input", "execute": "execute", "eda": "eda"}
    )
    
    return workflow.compile()

# ========================= MAIN =========================
def main():
    print("\n" + "="*60)
    print("🤖 AI DATA ANALYSIS AGENT")
    print("="*60)
    print("Commands: stats | undo | save as <file.csv> | exit")
    print("Examples: 'impute Age with median', 'show missing values'")
    print("="*60)
    
    graph = build_graph()
    try:
        for event in graph.stream(AgentState(), {"recursion_limit": 50}):
            if "__end__" in event:
                break
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
