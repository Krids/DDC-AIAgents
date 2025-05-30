import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
import logging # Import logging
import asyncio # Added asyncio
from pydantic import ValidationError
from abc import ABC, abstractmethod

from protocols.a2a_schemas import AgentCard, AgentCapability, Task, Artifact, TaskStatus, AgentMessage

logger = logging.getLogger(f"agentsAI.{__name__}") # Child logger

class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str, version: str = "0.1.0", **kwargs):
        self.agent_id = agent_id
        self.card = AgentCard(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=[],
            version=version
        )
        self.message_handler: Optional[Callable[[AgentMessage], Any]] = None
        self._last_status_update_sent_at = {}
        self._current_tasks = {}
        logger.info(f"Agent {self.agent_id} ({self.card.name}) initialized.")

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_capability(self, skill_name: str, description: str,
                            input_schema: Optional[Dict[str, Any]] = None,
                            output_schema: Optional[Dict[str, Any]] = None):
        capability = AgentCapability(
            skill_name=skill_name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema
        )
        self.card.capabilities.append(capability)
        logger.debug(f"Agent {self.agent_id} registered capability: {skill_name}")

    def get_agent_card(self) -> AgentCard:
        return self.card

    def create_task(self, description: str, initiator_agent_id: str,
                    assigned_to_agent_id: Optional[str] = None,
                    parent_task_id: Optional[str] = None,
                    input_artifacts: Optional[List[Artifact]] = None) -> Task:
        now = self._get_timestamp()
        task_id = str(uuid.uuid4())
        logger.debug(f"Agent {self.agent_id} creating task {task_id}: {description}")
        return Task(
            task_id=task_id,
            parent_task_id=parent_task_id,
            assigned_to_agent_id=assigned_to_agent_id or self.agent_id,
            initiator_agent_id=initiator_agent_id,
            description=description,
            status=TaskStatus.PENDING,
            input_artifacts=input_artifacts or [],
            output_artifacts=[],
            created_at=now,
            updated_at=now
        )

    def update_task_status(self, task: Task, status: TaskStatus) -> Task:
        task.status = status
        task.updated_at = self._get_timestamp()
        task.status_updated_at = task.updated_at
        logger.info(f"Task {task.task_id} status updated to {status} by agent {self.agent_id}")
        return task

    def add_output_artifact_to_task(self, task: Task, artifact: Artifact) -> Task:
        task.output_artifacts.append(artifact)
        task.updated_at = self._get_timestamp()
        logger.debug(f"Agent {self.agent_id} added artifact {artifact.artifact_id} to task {task.task_id}")
        return task

    def create_artifact(self, task_id: str, content_type: str, data: Any, description: Optional[str] = None) -> Artifact:
        artifact_id = str(uuid.uuid4())
        logger.debug(f"Agent {self.agent_id} creating artifact {artifact_id} for task {task_id} with content type {content_type}")
        return Artifact(
            artifact_id=artifact_id,
            task_id=task_id,
            creator_agent_id=self.agent_id,
            content_type=content_type,
            data=data,
            description=description,
            created_at=self._get_timestamp()
        )

    async def send_message(self, receiver_agent_id: str, message_type: str, payload: Dict[str, Any]):
        if not self.message_handler:
            logger.warning(f"Agent {self.agent_id} has no message handler configured to send {message_type} message to {receiver_agent_id}.")
            return

        message_id = str(uuid.uuid4())
        message = AgentMessage(
            message_id=message_id,
            sender_agent_id=self.agent_id,
            receiver_agent_id=receiver_agent_id,
            timestamp=self._get_timestamp(),
            message_type=message_type,
            payload=payload
        )
        try:
            # Shorten payload for logging if it's too long
            log_payload = str(payload)
            if len(log_payload) > 200:
                log_payload = log_payload[:200] + "... (truncated)"
            logger.debug(f"Agent {self.agent_id} sending message ID {message_id} ({message_type}) to {receiver_agent_id}. Payload: {log_payload}")
            await self.message_handler(message)
        except Exception as e:
            logger.error(f"Error sending message ID {message_id} from {self.agent_id} to {receiver_agent_id}: {e}", exc_info=True)

    def set_message_handler(self, handler: Callable[[AgentMessage], Any]):
        self.message_handler = handler
        logger.info(f"Message handler set for agent {self.agent_id}.")

    async def _send_status_update(self, task: Task):
        """Sends a task status update message to the task initiator if a message handler is set."""
        if task.status_updated_at == self._last_status_update_sent_at.get(task.task_id):
            # Avoid sending duplicate status updates if state hasn't changed since last send
            return

        logger.info(f"Task {task.task_id} status updated to {task.status} by agent {self.agent_id}")
        if self.message_handler and task.initiator_agent_id and task.initiator_agent_id != self.agent_id:
            message_payload = {
                "task_id": task.task_id,
                "status": task.status.value,
                "output_artifacts": [artifact.model_dump() for artifact in task.output_artifacts],
                "message": f"Task {task.task_id} is now {task.status.value}"
            }
            if task.status == TaskStatus.FAILED and task.error_message:
                message_payload["error_message"] = task.error_message

            status_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_agent_id=self.agent_id,
                receiver_agent_id=task.initiator_agent_id,
                message_type="task_status_update",
                payload=message_payload,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            try:
                # IMPORTANT: Await the handler if it's an async function (like AsyncMock)
                if asyncio.iscoroutinefunction(self.message_handler) or \
                   (hasattr(self.message_handler, '_is_coroutine') and self.message_handler._is_coroutine()): # Check for AsyncMock
                    await self.message_handler(status_message)
                else:
                    self.message_handler(status_message) # For sync handlers
                self._last_status_update_sent_at[task.task_id] = task.status_updated_at
                logger.debug(f"Sent task_status_update for {task.task_id} to {task.initiator_agent_id}")
            except Exception as e:
                logger.error(f"Agent {self.agent_id} failed to send status update for task {task.task_id} to {task.initiator_agent_id}: {e}")
        elif not self.message_handler:
            logger.warning(f"Agent {self.agent_id} has no message handler. Cannot send status for {task.task_id}.")
        elif task.initiator_agent_id == self.agent_id:
            logger.info(f"Task {task.task_id} was initiated by self. No status message sent.")

    async def handle_incoming_message(self, message: AgentMessage):
        log_payload = str(message.payload)
        if len(log_payload) > 200:
            log_payload = log_payload[:200] + "... (truncated)"
        logger.debug(f"Agent {self.agent_id} received message ID {message.message_id} ({message.message_type}) from {message.sender_agent_id}. Payload: {log_payload}")
        
        if message.message_type == "task_assignment":
            try:
                task_data = message.payload
                if isinstance(task_data, dict):
                    task = Task(**task_data)
                    logger.info(f"Agent {self.agent_id} received task assignment: '{task.description}' (ID: {task.task_id})")
                    await self.process_task(task)
                else:
                    logger.error(f"Task payload for {self.agent_id} (message ID {message.message_id}) is not a dictionary.")
            except ValidationError as ve:
                 logger.error(f"Validation error for task payload for agent {self.agent_id} (message ID {message.message_id}): {ve}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing task assignment for {self.agent_id} (message ID {message.message_id}): {e}", exc_info=True)
        else:
            logger.debug(f"Agent {self.agent_id} passing unhandled message type '{message.message_type}' to subclass or ignoring.")

    async def process_task(self, task: Task):
        logger.info(f"Agent {self.agent_id} starting processing of task {task.task_id}: '{task.description}'")
        self.update_task_status(task, TaskStatus.IN_PROGRESS)
        
        # Placeholder for actual work by subclasses
        # Simulate work by delaying slightly
        await asyncio.sleep(0.1) 
        logger.info(f"Agent {self.agent_id} (base class) completed simulated processing for task {task.task_id}.")
        
        self.update_task_status(task, TaskStatus.COMPLETED)
        
        if task.initiator_agent_id and self.message_handler and task.initiator_agent_id != self.agent_id:
             logger.info(f"Agent {self.agent_id} sending completion status for task {task.task_id} to initiator {task.initiator_agent_id}.")
             await self._send_status_update(task)
        elif task.initiator_agent_id == self.agent_id:
            logger.debug(f"Task {task.task_id} was initiated by self ({self.agent_id}), no status update message sent to self via message handler.")

    def __repr__(self):
        return f"<{self.__class__.__name__}(agent_id='{self.agent_id}', name='{self.card.name}')>"
