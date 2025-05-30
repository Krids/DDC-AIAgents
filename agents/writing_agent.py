import asyncio
# import os # Unused
import logging
from openai import AsyncOpenAI, OpenAIError
from typing import Optional
import json
from datetime import datetime
from utils.json_utils import convert_datetime_to_iso_string
import os

from agents.base_agent import BaseAgent, Task, TaskStatus, Artifact
# from protocols.a2a_schemas import AgentMessage # Unused
from core.agent_prompt_builder import generate_prompt as build_llm_prompt

logger = logging.getLogger(f"agentsAI.{__name__}")

class WritingAgent(BaseAgent):
    AGENT_DATA_SUBFOLDER = "writing_openai"

    def __init__(self, agent_id: str = "writing_agent_001", name: str = "Writing Agent",
                 description: str = "Writes blog post drafts using OpenAI based on research.",
                 data_dir_override: Optional[str] = None, **kwargs):
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)
        self.data_dir_override = data_dir_override
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

    async def _generate_draft_with_openai(self, topic: str, research_findings: str, task_id_for_log: Optional[str] = None) -> Optional[str]:
        if not self.openai_client:
            logger.warning(f"{self.card.name}: OpenAI client not available. Cannot generate draft for '{topic}'. Returning placeholder.")
            return f"Placeholder draft for {topic} - OpenAI client not initialized."

        logger.info(f"{self.card.name} generating blog draft for topic: '{topic}' based on research artifact")
        
        # Prepare input for the agent_prompt_builder
        prompt_input_data = {
            "task": f"Write a blog post about: {topic}",
            "input_type": "research findings (markdown text)",
            "output_format": "A single block of Markdown text representing the full blog post. Do not include any pre-amble or conversational text outside the blog post itself. The blog post should be at least 500 words.",
            "style": "informative yet accessible",
            "creativity": "medium" # Or make this configurable
        }
        
        # Add detailed instructions from the original implementation to the task description
        detailed_instructions = [ 
            "Start with a compelling introduction that grabs the reader's attention.",
            "Develop the main points with clear explanations and supporting details from the research.",
            "Organize the content logically with headings and subheadings.",
            "Maintain an informative yet accessible tone.",
            "Conclude with a summary that reinforces the key takeaways and perhaps offers a forward-looking perspective."
        ]
        prompt_input_data["task"] += "\n\nKey instructions to follow:\n" + "\n".join([f"- {instr}" for instr in detailed_instructions])
        
        prompt_dict = build_llm_prompt(input_data=prompt_input_data)
        # The actual research_findings will be appended to this raw_prompt later
        # where [PASTE THE INPUT HERE] is located.
        llm_prompt_template = prompt_dict.get("raw_prompt", "Failed to generate prompt.")
        
        # Replace the placeholder with the actual research findings
        final_llm_prompt = llm_prompt_template.replace("[PASTE THE INPUT HERE]", research_findings)

        openai_response = None
        try:
            logger.debug(f"{self.card.name}: Sending request to OpenAI for topic '{topic}'. Prompt (first 100 chars): {final_llm_prompt[:100]}...")
            completion = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    # The system message is now part of the raw_prompt from build_llm_prompt
                    # We can send the whole thing as a user message, or try to parse system/user parts.
                    # For simplicity, sending all as user message after the system message built into the prompt template.
                    # However, the current build_llm_prompt already includes [System], [Role] etc. which are not standard OpenAI API message roles.
                    # Option 1: Send entire final_llm_prompt as user content. This is simpler.
                    # {"role": "system", "content": "You are a helpful AI assistant specialized in writing insightful blog posts."}, # This might be redundant if prompt_dict has system part
                    {"role": "user", "content": final_llm_prompt}
                ],
                temperature=0.7,
                max_tokens=1500 
            )
            openai_response = completion # Store for saving

            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                generated_text = completion.choices[0].message.content.strip()
                logger.info(f"{self.card.name}: Successfully generated blog draft for '{topic}' using OpenAI and prompt builder.")
                return generated_text
            else:
                logger.error(f"{self.card.name}: OpenAI response for '{topic}' did not contain expected content. Response: {completion}")
                return None
        except OpenAIError as e:
            logger.error(f"{self.card.name}: OpenAI API error while generating draft for '{topic}': {e}", exc_info=True)
            if not isinstance(openai_response, dict) and openai_response is not None:
                 openai_response = {"error": str(e), "traceback": logging.Formatter().formatException(logging.sys.exc_info())}
            elif openai_response is None:
                 openai_response = {"error": str(e), "message": "OpenAI call failed before response object was obtained."}
            return None
        except Exception as e:
            logger.error(f"{self.card.name}: Unexpected error while generating draft for '{topic}': {e}", exc_info=True)
            if not isinstance(openai_response, dict) and openai_response is not None:
                 openai_response = {"error": str(e), "traceback": logging.Formatter().formatException(logging.sys.exc_info())}
            elif openai_response is None:
                 openai_response = {"error": str(e), "message": "OpenAI call failed before response object was obtained."}
            return None
        finally:
            if openai_response is not None:
                try:
                    base_save_path = self.data_dir_override
                    if base_save_path is None:
                        base_save_path = os.path.join("data", self.AGENT_DATA_SUBFOLDER)
                    
                    os.makedirs(base_save_path, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_id = task_id_for_log if task_id_for_log else "unknown_task"
                    filename = os.path.join(base_save_path, f"openai_writing_{self.agent_id}_{log_id}_{ts}.json")
                    
                    data_to_save = openai_response
                    if hasattr(data_to_save, 'model_dump'): # OpenAI v1.x Pydantic model
                        data_to_save = data_to_save.model_dump()

                    serializable_openai_response = convert_datetime_to_iso_string(data_to_save)
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(serializable_openai_response, f, indent=4, ensure_ascii=False)

                    logger.info(f"Saved OpenAI writing response to {filename}")
                except Exception as log_e:
                    logger.error(f"Failed to save OpenAI writing response to file: {log_e}", exc_info=True)

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
        topic = "the provided research" # Default topic
        
        # Make the check case-insensitive for "topic:"
        lower_topic_desc = topic_desc.lower()
        if "topic:" in lower_topic_desc:
            # Split original string by finding the index of "topic:" case-insensitively
            prefix_idx = lower_topic_desc.find("topic:")
            # Get the part after "topic:" from the original string to preserve case
            topic_segment = topic_desc[prefix_idx + len("topic:"):].strip()
            # Remove trailing details like " (generated by...)" or " (detailed)"
            extracted_topic_candidate = topic_segment.split(" (", 1)[0].strip()
            if extracted_topic_candidate: # Only update if extraction is non-empty
                topic = extracted_topic_candidate
            # If extracted_topic_candidate is empty, 'topic' remains "the provided research"

        logger.info(f"{self.card.name} generating blog draft for topic: '{topic}' based on research artifact {research_artifact.artifact_id}")
        
        # Pass task.task_id for logging
        generated_draft = await self._generate_draft_with_openai(topic, research_findings, task_id_for_log=task.task_id)
        
        if generated_draft:
            output_artifact = self.create_artifact(
                task_id=task.task_id,
                content_type="text/markdown",
                data=generated_draft,
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
        else:
            logger.error(f"{self.card.name}: Failed to generate blog draft for topic '{topic}'")
            self.update_task_status(task, TaskStatus.FAILED)
            if task.initiator_agent_id and self.message_handler:
                await self.send_message(receiver_agent_id=task.initiator_agent_id, message_type="task_status_update", payload=task.model_dump())

        # This final status update is handled by BaseAgent._handle_task_assignment's finally block
        # if task.initiator_agent_id and self.message_handler:
        #     self.send_message(
        #         receiver_agent_id=task.initiator_agent_id,
        #         message_type="task_status_update",
        #         payload=task.model_dump()
        #     ) 