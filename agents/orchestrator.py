import asyncio
import uuid
from typing import List, Dict, Any, Optional
import logging # Import logging

from agents.base_agent import BaseAgent, Task, Artifact, TaskStatus, AgentMessage
from protocols.a2a_schemas import Task # AgentCard removed, Task might be imported directly if used for type hints

logger = logging.getLogger(f"agentsAI.{__name__}") # Child logger

class OrchestratorAgent(BaseAgent):
    def __init__(self, agent_id: str = "orchestrator_agent_001", name: str = "Orchestrator Agent",
                 description: str = "Manages and coordinates other agents to complete complex tasks.", **kwargs):
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.task_callbacks: Dict[str, asyncio.Future] = {}
        # No need to log init here, base class does it.

        self.register_capability(
            skill_name="manage_blog_post_creation",
            description="Orchestrates the creation of a blog post from topic to final draft.",
            input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"blog_post_artifact_id": {"type": "string"}}}
        )

    def register_agent(self, agent: BaseAgent):
        if agent.agent_id in self.registered_agents:
            logger.warning(f"Agent {agent.agent_id} ({agent.card.name}) is already registered.")
            return
        self.registered_agents[agent.agent_id] = agent
        agent.set_message_handler(self.route_message)
        logger.info(f"Agent {agent.card.name} (ID: {agent.agent_id}) registered with orchestrator.")
        logger.debug(f"  Capabilities for {agent.agent_id}: {[cap.skill_name for cap in agent.get_agent_card().capabilities]}")

    def discover_agents_with_capability(self, skill_name: str) -> List[BaseAgent]:
        logger.debug(f"Discovering agents with capability: {skill_name}")
        matching_agents = []
        for agent in self.registered_agents.values():
            for capability in agent.get_agent_card().capabilities:
                if capability.skill_name == skill_name:
                    matching_agents.append(agent)
                    break
        logger.debug(f"Found {len(matching_agents)} agent(s) with capability {skill_name}: {[a.agent_id for a in matching_agents]}")
        return matching_agents

    async def route_message(self, message: AgentMessage):
        log_payload = str(message.payload)
        if len(log_payload) > 200:
            log_payload = log_payload[:200] + "... (truncated)"
        logger.debug(f"Orchestrator routing message ID {message.message_id} from {message.sender_agent_id} to {message.receiver_agent_id}. Type: {message.message_type}. Payload: {log_payload}")
        if message.receiver_agent_id == self.agent_id:
            await self.handle_incoming_message(message)
        elif message.receiver_agent_id in self.registered_agents:
            recipient_agent = self.registered_agents[message.receiver_agent_id]
            await recipient_agent.handle_incoming_message(message)
        else:
            logger.error(f"Receiver agent {message.receiver_agent_id} not found for message ID {message.message_id}. Known agents: {list(self.registered_agents.keys())}")

    async def handle_incoming_message(self, message: AgentMessage):
        # Base class handle_incoming_message already logs the receipt.
        # We call it first to ensure that logging and basic parsing happen.
        await super().handle_incoming_message(message)

        if message.message_type == "task_status_update":
            task_data = message.payload
            if isinstance(task_data, dict):
                try:
                    updated_task = Task(**task_data)
                    logger.info(f"Orchestrator received task status update for {updated_task.task_id}: {updated_task.status} from {message.sender_agent_id}")
                    if updated_task.task_id in self.task_callbacks:
                        future = self.task_callbacks[updated_task.task_id]
                        if not future.done():
                            if updated_task.status == TaskStatus.COMPLETED:
                                future.set_result(updated_task)
                            elif updated_task.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
                                future.set_exception(RuntimeError(f"Task {updated_task.task_id} (delegated to {message.sender_agent_id}) reported status: {updated_task.status}. Full task: {updated_task}"))
                        # Clean up callback if task is terminal
                        if updated_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                            del self.task_callbacks[updated_task.task_id]
                            logger.debug(f"Removed callback for terminal task {updated_task.task_id}")
                    else:
                        logger.warning(f"Received status update for untracked/already completed task {updated_task.task_id} from {message.sender_agent_id}")
                except Exception as e:
                    logger.error(f"Error processing task_status_update payload in orchestrator: {e}. Payload: {task_data}", exc_info=True)
            else:
                logger.error(f"Orchestrator received non-dict task_status_update payload: {task_data} from {message.sender_agent_id}")

    async def assign_task_and_wait(self, agent: BaseAgent, task_description: str, input_artifacts: Optional[List[Artifact]] = None) -> Task:
        task = self.create_task(
            description=task_description,
            initiator_agent_id=self.agent_id,
            assigned_to_agent_id=agent.agent_id,
            input_artifacts=input_artifacts or []
        )
        logger.info(f"Orchestrator assigning task '{task.description}' (ID: {task.task_id}) to agent {agent.card.name} (ID: {agent.agent_id})")

        future = asyncio.Future()
        self.task_callbacks[task.task_id] = future

        assignment_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_agent_id=self.agent_id,
            receiver_agent_id=agent.agent_id,
            timestamp=self._get_timestamp(),
            message_type="task_assignment",
            payload=task.model_dump()
        )
        await self.route_message(assignment_message)

        try:
            completed_task = await asyncio.wait_for(future, timeout=300.0) # Adding a timeout (e.g., 5 minutes)
            logger.info(f"Orchestrator: Task {completed_task.task_id} (assigned to {agent.card.name}) completed with status {completed_task.status}")
            return completed_task
        except asyncio.TimeoutError:
            logger.error(f"Orchestrator: Timeout waiting for task {task.task_id} (assigned to {agent.card.name}) to complete.")
            # Clean up callback
            if task.task_id in self.task_callbacks:
                del self.task_callbacks[task.task_id]
            task.status = TaskStatus.FAILED # Mark task as failed due to timeout
            task.updated_at = self._get_timestamp()
            # We might want to send a cancellation to the agent if that's supported
            return task # Return the task marked as failed
        except Exception as e:
            logger.error(f"Orchestrator: Error while waiting for task {task.task_id} (assigned to {agent.card.name}): {e}", exc_info=True)
            if task.task_id in self.task_callbacks:
                del self.task_callbacks[task.task_id]
            task.status = TaskStatus.FAILED
            task.updated_at = self._get_timestamp()
            return task

    async def execute_blog_post_workflow(self, topic: str) -> Optional[Artifact]:
        logger.info(f"--- Orchestrator starting Blog Post Creation Workflow for topic: '{topic}' ---")
        final_blog_post_artifact = None
        try:
            research_agents = self.discover_agents_with_capability("research_topic_web")
            if not research_agents:
                logger.error("Orchestrator: No ContentResearchAgent with 'research_topic_web' capability found.")
                return None
            research_agent = research_agents[0]

            initial_research_artifact = self.create_artifact(
                task_id="initial_topic_artifact_for_" + str(uuid.uuid4()),
                content_type="text/plain",
                data=topic,
                description="Initial blog post topic"
            )
            logger.debug(f"Created initial topic artifact: {initial_research_artifact.artifact_id}")

            research_task_result = await self.assign_task_and_wait(
                research_agent,
                f"Research recent news on the topic: {topic}",
                input_artifacts=[initial_research_artifact]
            )
            if research_task_result.status != TaskStatus.COMPLETED or not research_task_result.output_artifacts:
                logger.error(f"Orchestrator: Research task {research_task_result.task_id} failed or produced no artifacts. Status: {research_task_result.status}")
                return None
            researched_content_artifact = research_task_result.output_artifacts[0]
            logger.info(f"Orchestrator: Research completed. Artifact ID: {researched_content_artifact.artifact_id}")

            writing_agents = self.discover_agents_with_capability("write_content")
            if not writing_agents:
                logger.error("Orchestrator: No WritingAgent found.")
                return None
            writing_agent = writing_agents[0]
            drafting_task_result = await self.assign_task_and_wait(writing_agent, f"Write a blog post: {topic}", [researched_content_artifact])
            if drafting_task_result.status != TaskStatus.COMPLETED or not drafting_task_result.output_artifacts:
                logger.error(f"Orchestrator: Drafting task failed. Status: {drafting_task_result.status}")
                return None
            draft_artifact = drafting_task_result.output_artifacts[0]
            logger.info(f"Orchestrator: Drafting completed. Artifact: {draft_artifact.artifact_id}")

            seo_agents = self.discover_agents_with_capability("optimize_seo")
            if not seo_agents:
                logger.warning("Orchestrator: No SEOAgent found. Skipping SEO.")
                seo_optimized_artifact = draft_artifact
            else:
                seo_agent = seo_agents[0]
                seo_task_result = await self.assign_task_and_wait(seo_agent, f"Optimize SEO: {topic}", [draft_artifact])
                if seo_task_result.status != TaskStatus.COMPLETED or not seo_task_result.output_artifacts:
                    logger.warning(f"Orchestrator: SEO task failed. Status: {seo_task_result.status}. Using unoptimized draft.")
                    seo_optimized_artifact = draft_artifact
                else:
                    seo_optimized_artifact = seo_task_result.output_artifacts[0]
                    logger.info(f"Orchestrator: SEO optimization completed. Artifact: {seo_optimized_artifact.artifact_id}")
            
            image_agents = self.discover_agents_with_capability("find_images_openai")
            if not image_agents:
                logger.warning("Orchestrator: No ImageAgent with OpenAI found. Skipping image generation.")
                final_blog_post_artifact = seo_optimized_artifact
            else:
                image_agent = image_agents[0]
                image_task_result = await self.assign_task_and_wait(image_agent, f"Generate images for: {topic}", [seo_optimized_artifact])
                if image_task_result.status != TaskStatus.COMPLETED or not image_task_result.output_artifacts:
                    logger.warning(f"Orchestrator: Image task failed. Status: {image_task_result.status}. Using content without new images.")
                    final_blog_post_artifact = seo_optimized_artifact
                else:
                    final_blog_post_artifact = image_task_result.output_artifacts[0]
                    logger.info(f"Orchestrator: Image processing completed. Artifact: {final_blog_post_artifact.artifact_id}")
            
            logger.info(f"--- Orchestrator: Blog Post Creation Workflow for topic: '{topic}' COMPLETED ---")
            logger.info(f"Final Blog Post Artifact ID: {final_blog_post_artifact.artifact_id if final_blog_post_artifact else 'N/A'}")
            return final_blog_post_artifact

        except Exception as e:
            logger.error(f"Orchestrator: Error during blog post workflow for topic '{topic}': {e}", exc_info=True)
            return None

    async def process_task(self, task: Task):
        # The base class logs the start of processing this task.
        # await super().process_task(task) # Don't call super if we are overriding its full logic

        logger.info(f"Orchestrator (as agent) processing its own assigned task ID {task.task_id}: {task.description}")
        self.update_task_status(task, TaskStatus.IN_PROGRESS) # Already done by base, but good to be explicit if overriding

        if task.description.startswith("Create a blog post on topic:"):
            topic = task.description.replace("Create a blog post on topic:", "").strip()
            final_artifact = await self.execute_blog_post_workflow(topic)
            if final_artifact:
                self.add_output_artifact_to_task(task, final_artifact)
                self.update_task_status(task, TaskStatus.COMPLETED)
            else:
                logger.error(f"Orchestrator: Blog post workflow for '{topic}' did not produce a final artifact or failed.")
                self.update_task_status(task, TaskStatus.FAILED)
        else:
            logger.warning(f"Orchestrator doesn't know how to directly process task: {task.description}")
            self.update_task_status(task, TaskStatus.FAILED)

        # Notify initiator of this orchestrator-level task (e.g., system in main.py)
        # BaseAgent.process_task already has logic to send this update, but only if initiator_id is different.
        # If orchestrator is assigned a task by main.py (system_initiator_id), we want to send an update.
        if task.initiator_agent_id and task.initiator_agent_id != self.agent_id and self.message_handler:
             logger.info(f"Orchestrator sending completion status for its own task {task.task_id} to initiator {task.initiator_agent_id}.")
             self.send_message(
                 receiver_agent_id=task.initiator_agent_id,
                 message_type="task_status_update",
                 payload=task.model_dump()
             )
        elif task.initiator_agent_id == self.agent_id:
             logger.debug(f"Orchestrator's task {task.task_id} was self-initiated or handled by base class logic, no explicit final update sent here.") 