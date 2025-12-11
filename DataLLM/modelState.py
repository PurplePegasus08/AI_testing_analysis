# state.py
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Literal, Protocol
from pydantic import BaseModel, Field, validator

# ---------- light-weight state that LangGraph can serialise ----------
class UndoEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    op: Literal["impute", "clip", "encode", "drop", "convert"]
    params: Dict[str, Any]
    description: str
    # parquet bytes of working_df *before* the op
    mask_bytes: bytes


class AgentState(BaseModel):
    # 1. identifiers only ─ dataframes live in an external store (see below)
    raw_id: str = ""          # immutable original
    work_id: str = ""         # current version

    # 2. validated undo stack
    history: List[UndoEntry] = Field(default_factory=list)

    # 3. LLM proposals
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)

    # 4. router
    next_node: Literal[
        "upload", "eda", "llm_suggest", "human_review",
        "execute", "undo", "export", "END"
    ] = "upload"

    # 5. UI
    user_message: str = ""
    error: Optional[str] = None
    active_suggestion_id: Optional[str] = None

    # ---------- helpers ----------
    def push_undo(self, store: DataStore, op: str, params: dict, desc: str) -> None:
        """Create undo snapshot BEFORE you mutate."""
        self.history.append(UndoEntry(
            op=op, params=params, description=desc,
            mask_bytes=store.to_parquet(self.work_id)
        ))

    def undo(self, store: DataStore) -> None:
        """Restore previous version."""
        if not self.history:
            self.user_message = "Nothing to undo"
            return
        entry = self.history.pop()
        self.work_id = store.from_parquet(entry.mask_bytes)
        self.user_message = f"Undone: {entry.description}"

    def reset_to_raw(self, store: DataStore) -> None:
        """Jump back to original upload."""
        self.work_id = self.raw_id
        self.history.clear()
        self.user_message = "Reset to original data"

    # ---------- immutability guard ----------
    @validator("raw_id", allow_reuse=True)
    def _raw_once(cls, v: str, values: dict[str, Any]) -> str:
        if values.get("raw_id") and v != values["raw_id"]:
            raise ValueError("raw_id is immutable once set")
        return v


# ---------- minimal external store protocol ----------
class DataStore(Protocol):
    def to_parquet(self, data_id: str) -> bytes: ...
    def from_parquet(self, blob: bytes) -> str:   # returns new data_id
        ...
