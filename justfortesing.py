from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tool_node,tools_condition
from pydantic import BaseModel
from typing import List, Union
from dotenv import load_dotenv
from langchain_core.tools import tool
import os


# Load API key
load_dotenv()
# repl tool


api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ API Key not found.")
# You can create the tool to pass to an agent


# Python REPL Tool
repl = PythonREPLTool(
    python_opts={"packages": ["pandas","numpy", "matplotlib", "seaborn", "plotly", "statsmodels"]}
)
tools = [repl]



# LLM with tool calls
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
    allow_dangerous_code=True
).bind_tools(tools)

# ------------------------------
# STATE SCHEMA
# ------------------------------
class GraphState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = []

# ------------------------------
# AGENT NODE
# ------------------------------
def agent_node(state: GraphState) -> GraphState:
    """LLM decides whether to call tools."""
    response = llm.invoke(state.messages)

    return GraphState(messages=state.messages + [response])

# ------------------------------
# TOOL NODE
# ------------------------------
def tool_node(state: GraphState) -> GraphState:
    last_msg = state.messages[-1]

    outputs = []
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for call in last_msg.tool_calls:
            result = repl.invoke(call)
            outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    return GraphState(messages=state.messages + outputs)

# ------------------------------
# CONDITION FOR TOOL-CALLING
# ------------------------------
def tools_condition(state: GraphState) -> bool:
    last_msg = state.messages[-1]
    return hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0

# ------------------------------
# BUILD LANGGRAPH
# ------------------------------
graph = StateGraph(state_schema=GraphState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", tools_condition, "tools")
graph.add_edge("tools", "agent")

compiled_graph = graph.compile()

# ------------------------------
# RUN GRAPH
# ------------------------------
initial_state = GraphState(messages=[HumanMessage(content="Hello! Make a Python list without using any tool.")])

final_state = compiled_graph.invoke(initial_state)

print("FINAL STATE:", final_state)
print(AIMessage)
