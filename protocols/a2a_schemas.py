from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime, timezone

class AgentCapability(BaseModel):
    skill_name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

class AgentCard(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    version: str = "0.1.0"
    # We can add connection info here later if agents become separate services
    # e.g., endpoint: Optional[str] = None

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Artifact(BaseModel):
    artifact_id: str
    task_id: str
    creator_agent_id: str
    content_type: str # e.g., "text/plain", "application/json", "image/png"
    data: Any
    description: Optional[str] = None
    created_at: str # ISO 8601 timestamp

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: Optional[str] = None
    assigned_to_agent_id: Optional[str] = None # Agent currently assigned to perform the task
    initiator_agent_id: str # Agent that initiated the task, to report back to
    description: str
    status: TaskStatus = TaskStatus.PENDING
    input_artifacts: List[Artifact] = Field(default_factory=list)
    output_artifacts: List[Artifact] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status_updated_at: Optional[str] = None # Timestamp of the last status update specifically
    error_message: Optional[str] = None
    # priority: int = 0 # Future use
    # dependencies: List[str] = Field(default_factory=list) # Future use for task chaining
    # Could include priority, deadlines, etc.

# Example of a message structure (could be expanded)
class AgentMessage(BaseModel):
    message_id: str
    sender_agent_id: str
    receiver_agent_id: str
    timestamp: str # ISO 8601 timestamp
    message_type: Literal["task_assignment", "task_status_update", "artifact_delivery", "query_capability", "error"]
    payload: Dict[str, Any] # This could be a Task, Artifact, status, etc. 