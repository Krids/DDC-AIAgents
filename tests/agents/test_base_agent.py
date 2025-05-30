import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock # For mocking message_handler

from agents.base_agent import BaseAgent
from protocols.a2a_schemas import AgentCard, AgentCapability, Task, Artifact, TaskStatus, AgentMessage

@pytest.fixture
def base_agent():
    """Provides a BaseAgent instance for testing."""
    return BaseAgent(agent_id="test_base_001", name="Test Base Agent", description="A base agent for testing.")

def test_base_agent_initialization(base_agent: BaseAgent):
    assert base_agent.agent_id == "test_base_001"
    assert isinstance(base_agent.card, AgentCard)
    assert base_agent.card.name == "Test Base Agent"
    assert base_agent.card.description == "A base agent for testing."
    assert base_agent.card.version == "0.1.0" # Default version
    assert len(base_agent.card.capabilities) == 0
    assert base_agent.message_handler is None

def test_register_capability(base_agent: BaseAgent):
    base_agent.register_capability(
        skill_name="test_skill", 
        description="A test skill",
        input_schema={"type": "string"},
        output_schema={"type": "number"}
    )
    assert len(base_agent.card.capabilities) == 1
    capability = base_agent.card.capabilities[0]
    assert isinstance(capability, AgentCapability)
    assert capability.skill_name == "test_skill"
    assert capability.description == "A test skill"
    assert capability.input_schema == {"type": "string"}
    assert capability.output_schema == {"type": "number"}

def test_get_agent_card(base_agent: BaseAgent):
    card = base_agent.get_agent_card()
    assert card == base_agent.card

def test_create_task(base_agent: BaseAgent):
    initiator_id = "initiator_007"
    task_desc = "Perform a test task"
    task = base_agent.create_task(description=task_desc, initiator_agent_id=initiator_id)
    
    assert isinstance(task, Task)
    assert task.description == task_desc
    assert task.initiator_agent_id == initiator_id
    assert task.assigned_to_agent_id == base_agent.agent_id # Default assignment
    assert task.status == TaskStatus.PENDING
    assert len(task.input_artifacts) == 0
    assert len(task.output_artifacts) == 0
    assert task.task_id is not None
    assert task.created_at is not None
    assert task.updated_at == task.created_at

def test_create_task_with_specific_assignment_and_parent(base_agent: BaseAgent):
    assigned_id = "other_agent_001"
    parent_id = "parent_task_123"
    task = base_agent.create_task(
        description="Sub-task", 
        initiator_agent_id=base_agent.agent_id, 
        assigned_to_agent_id=assigned_id,
        parent_task_id=parent_id
    )
    assert task.assigned_to_agent_id == assigned_id
    assert task.parent_task_id == parent_id

def test_update_task_status(base_agent: BaseAgent):
    task = base_agent.create_task(description="Test status update", initiator_agent_id="system")
    original_updated_at = task.updated_at
    
    updated_task = base_agent.update_task_status(task, TaskStatus.IN_PROGRESS)
    assert updated_task.status == TaskStatus.IN_PROGRESS
    assert updated_task.updated_at > original_updated_at

def test_add_output_artifact_to_task(base_agent: BaseAgent):
    task = base_agent.create_task(description="Test artifacts", initiator_agent_id="system")
    artifact_data = "This is artifact data"
    artifact = base_agent.create_artifact(task.task_id, "text/plain", artifact_data, "Test output artifact")
    
    updated_task = base_agent.add_output_artifact_to_task(task, artifact)
    assert len(updated_task.output_artifacts) == 1
    assert updated_task.output_artifacts[0] == artifact

def test_create_artifact(base_agent: BaseAgent):
    task_id_for_artifact = str(uuid.uuid4())
    artifact_data = {"key": "value"}
    artifact_desc = "A JSON artifact"
    artifact = base_agent.create_artifact(task_id_for_artifact, "application/json", artifact_data, artifact_desc)
    
    assert isinstance(artifact, Artifact)
    assert artifact.task_id == task_id_for_artifact
    assert artifact.creator_agent_id == base_agent.agent_id
    assert artifact.content_type == "application/json"
    assert artifact.data == artifact_data
    assert artifact.description == artifact_desc
    assert artifact.artifact_id is not None
    assert artifact.created_at is not None

def test_set_message_handler(base_agent: BaseAgent):
    mock_handler = MagicMock()
    base_agent.set_message_handler(mock_handler)
    assert base_agent.message_handler == mock_handler

@pytest.mark.asyncio
async def test_send_message_success(base_agent: BaseAgent):
    mock_handler = AsyncMock()
    base_agent.set_message_handler(mock_handler)
    
    receiver_id = "receiver_001"
    message_type = "task_status_update"
    payload = {"data": "sample"}
    
    await base_agent.send_message(receiver_id, message_type, payload)
    
    mock_handler.assert_called_once()
    sent_message = mock_handler.call_args[0][0]
    assert isinstance(sent_message, AgentMessage)
    assert sent_message.receiver_agent_id == receiver_id
    assert sent_message.message_type == message_type
    assert sent_message.payload == payload
    assert sent_message.sender_agent_id == base_agent.agent_id

@pytest.mark.asyncio
async def test_send_message_no_handler(base_agent: BaseAgent, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    base_agent.message_handler = None # Ensure no handler
    await base_agent.send_message("receiver_001", "test_message", {})
    assert f"Agent {base_agent.agent_id} has no message handler configured" in caplog.text

@pytest.mark.asyncio
async def test_handle_incoming_task_assignment_message(base_agent: BaseAgent):
    # Mock process_task as it's usually overridden and we're testing handle_incoming_message here
    base_agent.process_task = AsyncMock() 
    
    task_payload = {
        "task_id": str(uuid.uuid4()),
        "initiator_agent_id": "system_test",
        "assigned_to_agent_id": base_agent.agent_id,
        "description": "A task from a message",
        "status": TaskStatus.PENDING.value,
        "input_artifacts": [],
        "output_artifacts": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    message = AgentMessage(
        message_id=str(uuid.uuid4()),
        sender_agent_id="orchestrator_test",
        receiver_agent_id=base_agent.agent_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message_type="task_assignment",
        payload=task_payload
    )
    
    await base_agent.handle_incoming_message(message)
    
    base_agent.process_task.assert_awaited_once()
    called_task = base_agent.process_task.call_args[0][0]
    assert isinstance(called_task, Task)
    assert called_task.task_id == task_payload["task_id"]
    assert called_task.description == task_payload["description"]

@pytest.mark.asyncio
async def test_handle_incoming_task_assignment_invalid_payload(base_agent: BaseAgent, caplog):
    import logging
    caplog.set_level(logging.ERROR)
    base_agent.process_task = AsyncMock()
    
    invalid_payload = {"bad_data": "no_task_id"}
    message = AgentMessage(
        message_id=str(uuid.uuid4()),
        sender_agent_id="test_sender",
        receiver_agent_id=base_agent.agent_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message_type="task_assignment",
        payload=invalid_payload
    )
    await base_agent.handle_incoming_message(message)
    assert "Validation error for task payload" in caplog.text
    base_agent.process_task.assert_not_called()

@pytest.mark.asyncio
async def test_handle_incoming_non_task_message(base_agent: BaseAgent, caplog):
    import logging
    caplog.set_level(logging.DEBUG) # BaseAgent logs other messages at DEBUG
    base_agent.process_task = AsyncMock() # Mock to ensure it's not called for non-task messages
    message = AgentMessage(
        message_id=str(uuid.uuid4()),
        sender_agent_id="test_sender",
        receiver_agent_id=base_agent.agent_id,
        timestamp=datetime.now(timezone.utc).isoformat(), # Ensure correct timestamp format
        message_type="query_capability", # Changed to a valid Literal type
        payload={"info": "some info"} # Ensure payload is not None
    )
    await base_agent.handle_incoming_message(message)
    assert f"Agent {base_agent.agent_id} received message ID {message.message_id} (query_capability)" in caplog.text
    base_agent.process_task.assert_not_called()

@pytest.mark.asyncio
async def test_base_process_task_flow(base_agent: BaseAgent):
    """Test the default process_task behavior in BaseAgent, including status updates and message sending."""
    initiator_id = "test_initiator_agent"
    task = base_agent.create_task(description="Base process test", initiator_agent_id=initiator_id)
    
    # Mock the message handler to check if status update is sent
    mock_message_handler = AsyncMock() # Handler itself is called async by send_message if it were real
    base_agent.set_message_handler(mock_message_handler)
    
    await base_agent.process_task(task)
    
    assert task.status == TaskStatus.COMPLETED
    # Check if send_message was called to notify initiator
    mock_message_handler.assert_called_once()
    sent_message = mock_message_handler.call_args[0][0]
    assert isinstance(sent_message, AgentMessage)
    assert sent_message.message_type == "task_status_update"
    assert sent_message.receiver_agent_id == initiator_id
    assert sent_message.payload["task_id"] == task.task_id
    assert sent_message.payload["status"] == TaskStatus.COMPLETED.value

@pytest.mark.asyncio
async def test_base_process_task_self_initiated(base_agent: BaseAgent):
    """Test that process_task doesn't send a message if task is self-initiated."""
    task = base_agent.create_task(description="Self-initiated", initiator_agent_id=base_agent.agent_id)
    mock_message_handler = AsyncMock()
    base_agent.set_message_handler(mock_message_handler)

    await base_agent.process_task(task)
    mock_message_handler.assert_not_called() 