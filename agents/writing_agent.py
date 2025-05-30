import asyncio
# import os # Unused
import logging
from openai import AsyncOpenAI
from typing import Optional

from agents.base_agent import BaseAgent, Task, TaskStatus, Artifact
# from protocols.a2a_schemas import AgentMessage # Unused
from core.agent_prompt_builder import generate_prompt as build_llm_prompt

logger = logging.getLogger(f"agentsAI.{__name__}")

class WritingAgent(BaseAgent):
    def __init__(self, agent_id: str = "writing_agent_001", name: str = "Writing Agent",
                 description: str = "Writes blog post drafts using OpenAI based on research.", **kwargs):
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)
        self.register_capability(
            skill_name="write_content",
            description="Writes a blog post draft based on provided research findings.",
            input_schema={"type": "object", "properties": {"research_artifact_id": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"draft_artifact_id": {"type": "string"}}}
        )
        self.openai_client: Optional[AsyncOpenAI] = None
        try:
            self.openai_client = AsyncOpenAI()
            logger.info(f"OpenAI client initialized for {self.card.name}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client for {self.card.name}: {e}. Writing tasks will be simulated.", exc_info=True)
            # self.openai_client remains None

    async def generate_blog_draft_with_openai(self, research_findings: str, topic: str) -> str:
        if not self.openai_client:
            logger.warning(f"{self.card.name}: OpenAI client not available. Using SIMULATED draft generation for topic '{topic}'.")
            await asyncio.sleep(1)
            return (
                f"## Simulated Blog Post: {topic}\n\n"
                f"This is a *simulated* blog post draft about {topic}, generated because the OpenAI API is not available.\n\n"
                f"### Key Points from Research (Simulated Integration):\n"
                f"{research_findings[:300]}... (simulated summary)"
                f"\n\nFurther details and engaging content would be added here."
            )

        prompt_specs = {
            "task": f"Write an expert blog post about the topic: '{topic}'",
            "input_type": "Research Findings (provided below, in text/markdown format)",
            "output_format": (
                "A complete, well-structured blog post in Markdown format "
                "(approximately 500-800 words). Must include: "
                "1. A catchy H1 title. "
                "2. An engaging introduction. "
                "3. A body that elaborates on key points from the research, "
                "citing or incorporating them naturally. "
                "4. A conclusion summarizing main takeaways and offering a final thought. "
                "Ensure content is based strictly on the provided research findings; "
                "do not invent information. If research is sparse, elaborate on available points well."
            ),
            "style": "Professional, engaging, and accessible to a general audience interested in technology and AI.",
            "creativity": "medium"
        }
        
        prompt_build_result = build_llm_prompt(prompt_specs)
        user_prompt_template = prompt_build_result["raw_prompt"]
        
        if "[PASTE THE INPUT HERE]" in user_prompt_template:
            full_user_prompt = user_prompt_template.replace("[PASTE THE INPUT HERE]", research_findings)
        else:
            logger.warning("Prompt builder did not include '[PASTE THE INPUT HERE]'. Appending research findings directly.")
            full_user_prompt = (
                f"{user_prompt_template}\n\n"
                f"--- RESEARCH FINDINGS START ---\n{research_findings}\n"
                f"--- RESEARCH FINDINGS END ---"
            )

        system_prompt_content = "You are an expert blog post writer specializing in creating engaging and informative content based on provided research. You adhere strictly to formatting instructions and content constraints."
        
        logger.debug(f"{self.card.name}: Sending OpenAI request for topic '{topic}'. System prompt: '{system_prompt_content}'. User prompt snippet: {full_user_prompt[:250]}...")

        try:
            completion = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": full_user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            draft_content = completion.choices[0].message.content.strip()
            logger.info(f"{self.card.name}: Successfully generated blog draft for '{topic}' using OpenAI and prompt builder.")
            logger.debug(f"Prompt estimated tokens (from builder): {prompt_build_result.get('estimated_tokens', 'N/A')}")
            if completion.usage:
                 logger.debug(f"OpenAI usage: Prompt tokens: {completion.usage.prompt_tokens}, Completion tokens: {completion.usage.completion_tokens}, Total tokens: {completion.usage.total_tokens}")
            return draft_content
        except Exception as e:
            logger.error(f"{self.card.name}: Error calling OpenAI API for topic '{topic}': {e}", exc_info=True)
            await asyncio.sleep(0.5)
            return (
                f"## Error-Fallback Blog Post: {topic}\n\n"
                f"An error occurred while trying to generate this blog post about {topic} using OpenAI (with prompt builder).\n"
                f"Error details: {str(e)}\n\n"
                f"Based on research (fallback summary):\n{research_findings[:300]}..."
            )

    async def process_task(self, task: Task):
        logger.info(f"{self.card.name} ({self.agent_id}) starting task: {task.description}")
        self.update_task_status(task, TaskStatus.IN_PROGRESS)

        if not task.input_artifacts:
            logger.error(f"Error: Writing task {task.task_id} for {self.card.name} has no input research artifact.")
            self.update_task_status(task, TaskStatus.FAILED)
            if task.initiator_agent_id and self.message_handler:
                await self.send_message(receiver_agent_id=task.initiator_agent_id, message_type="task_status_update", payload=task.model_dump())
            return

        research_artifact = task.input_artifacts[0]
        research_findings = str(research_artifact.data) # Ensure data is string
        topic_desc = str(research_artifact.description) # Ensure description is string
        topic = "the provided research"
        
        # Make the check case-insensitive for "topic:"
        lower_topic_desc = topic_desc.lower()
        if "topic:" in lower_topic_desc:
            # Split original string by finding the index of "topic:" case-insensitively
            prefix_idx = lower_topic_desc.find("topic:")
            # Get the part after "topic:" from the original string to preserve case
            topic_segment = topic_desc[prefix_idx + len("topic:"):].strip()
            # Remove trailing details like " (generated by...)" or " (detailed)"
            topic = topic_segment.split(" (", 1)[0].strip()

        logger.info(f"{self.card.name} generating blog draft for topic: '{topic}' based on research artifact {research_artifact.artifact_id}")
        
        draft_content = await self.generate_blog_draft_with_openai(research_findings, topic)
        
        output_artifact = self.create_artifact(
            task_id=task.task_id,
            content_type="text/markdown",
            data=draft_content,
            description=f"Blog post draft for topic: {topic} (generated by OpenAI, prompt builder)"
        )
        self.add_output_artifact_to_task(task, output_artifact)
        self.update_task_status(task, TaskStatus.COMPLETED)
        logger.info(f"{self.card.name} ({self.agent_id}) completed task: {task.description}. Artifact: {output_artifact.artifact_id}")

        # Send the final status update
        if task.initiator_agent_id and self.message_handler and task.initiator_agent_id != self.agent_id:
            await self._send_status_update(task)

        # This final status update is handled by BaseAgent._handle_task_assignment's finally block
        # if task.initiator_agent_id and self.message_handler:
        #     self.send_message(
        #         receiver_agent_id=task.initiator_agent_id,
        #         message_type="task_status_update",
        #         payload=task.model_dump()
        #     ) 