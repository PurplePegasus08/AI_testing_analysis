# updated_modelstate.py
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Literal, Protocol
from pydantic import BaseModel, Field, field_validator


# ---------- Undo snapshot for dataframe changes ----------
class UndoEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    op: str  # operation name, e.g., "impute", "filter", "drop"
    params: Dict[str, Any]
    description: str
    mask_bytes: bytes  # parquet snapshot of dataframe before this op


# ---------- Main agent state ----------
class AgentState(BaseModel):
    # --- identifiers ---
    raw_id: str = ""       # immutable original dataframe
    work_id: str = ""      # current working version

    # --- undo history ---
    history: List[UndoEntry] = Field(default_factory=list)

    # --- LLM suggestions & code ---
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    generated_code: Optional[str] = None

    # --- routing ---
    next_node: Literal[
        "upload", "eda", "llm_suggest", "human_review",
        "execute", "undo", "export", "END"
    ] = "upload"

    # --- UI / chat state ---
    user_message: str = ""
    error: Optional[str] = None
    active_suggestion_id: Optional[str] = None

    # ---------- helpers ----------
    def push_undo(self, store: "DataStore", op: str, params: dict, desc: str) -> None:
        """Snapshot BEFORE applying a change to df."""
        self.history.append(
            UndoEntry(
                op=op,
                params=params,
                description=desc,
                mask_bytes=store.to_parquet(self.work_id)
            )
        )

    def undo(self, store: "DataStore") -> None:
        """Revert last change."""
        if not self.history:
            self.user_message = "Nothing to undo"
            return
        entry = self.history.pop()
        self.work_id = store.from_parquet(entry.mask_bytes)
        self.user_message = f"Undone: {entry.description}"

    def reset_to_raw(self, store: "DataStore") -> None:
        """Reset to original upload."""
        self.work_id = self.raw_id
        self.history.clear()
        self.user_message = "Reset to original data"

    # ---------- immutability guard ----------
    @field_validator("raw_id")
    @classmethod
    def _raw_once(cls, v: str, info):
        if info.data.get("raw_id") and v != info.data["raw_id"]:
            raise ValueError("raw_id is immutable once set")
        return v


# ---------- minimal external store protocol ----------
class DataStore(Protocol):
    def to_parquet(self, data_id: str) -> bytes: ...
    def from_parquet(self, blob: bytes) -> str: ...
