import asyncio
# import os # Unused
import logging
from openai import AsyncOpenAI
# import base64 # Unused
# import requests # Unused
from typing import List, Optional
import re

from agents.base_agent import BaseAgent, Task, TaskStatus, Artifact
# from protocols.a2a_schemas import AgentMessage # Unused

logger = logging.getLogger(f"agentsAI.{__name__}")

class ImageAgent(BaseAgent):
    """
    Agent responsible for generating images using OpenAI DALL-E.
    """
    def __init__(self, agent_id: str = "image_agent_001", name: str = "Image Agent",
                 description: str = "Generates images using OpenAI DALL-E and adds them to content.", **kwargs):
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)
        self.register_capability(
            skill_name="find_images_openai",
            description="Generates images using OpenAI DALL-E based on content/topic.",
            input_schema={"type": "object", "properties": {"content_artifact_id": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"content_with_images_artifact_id": {"type": "string"}}}
        )
        self.openai_client: Optional[AsyncOpenAI] = None
        try:
            self.openai_client = AsyncOpenAI()
            logger.info(f"OpenAI client initialized for {self.card.name}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client for {self.card.name}: {e}. Image generation will use placeholders.", exc_info=True)
            # self.openai_client remains None

    async def generate_image_with_dalle(self, prompt: str, image_size: str = "1024x1024", quality:str="standard", n_images: int = 1) -> List[str]:
        """
        Generates an image using DALL-E and returns a list of image URLs.
        DALL-E 3 typically returns URLs.
        """
        if not self.openai_client:
            logger.warning(f"{self.card.name}: OpenAI client not available. Cannot generate images for prompt: '{prompt[:100]}...'.")
            return []
        
        try:
            logger.info(f"{self.card.name}: Sending DALL-E request for prompt (first 100 chars): '{prompt[:100]}...'")
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=image_size, # DALL-E 3 options: 1024x1024, 1792x1024, or 1024x1792
                quality=quality, # "standard" or "hd"
                n=n_images,
                response_format="url" # "url" or "b64_json"
            )
            
            image_urls = [item.url for item in response.data if item.url]
            
            logger.info(f"{self.card.name}: Received {len(image_urls)} image URL(s) from DALL-E for prompt: '{prompt[:100]}...'")
            if image_urls:
                for i, url in enumerate(image_urls):
                    logger.debug(f"DALL-E Image URL {i+1}: {url}")
            return image_urls
        except Exception as e:
            logger.error(f"{self.card.name}: Error calling DALL-E API for prompt '{prompt[:100]}...': {e}", exc_info=True)
            return []

    async def process_task(self, task: Task):
        logger.info(f"{self.card.name} ({self.agent_id}) starting task: {task.description}")
        self.update_task_status(task, TaskStatus.IN_PROGRESS)

        if not self.openai_client: # Check moved here as it's critical for this agent's core function
            logger.error(f"Error: OpenAI client not initialized for ImageAgent {self.agent_id}. Cannot proceed with image generation.")
            self.update_task_status(task, TaskStatus.FAILED)
            if task.initiator_agent_id and self.message_handler:
                await self.send_message(receiver_agent_id=task.initiator_agent_id, message_type="task_status_update", payload=task.model_dump())
            return

        if not task.input_artifacts:
            logger.error(f"Error: Image task {task.task_id} for {self.card.name} has no input content artifact.")
            self.update_task_status(task, TaskStatus.FAILED)
            if task.initiator_agent_id and self.message_handler:
                await self.send_message(receiver_agent_id=task.initiator_agent_id, message_type="task_status_update", payload=task.model_dump())
            return

        content_artifact = task.input_artifacts[0]
        original_content = str(content_artifact.data) # Ensure string
        artifact_desc = str(content_artifact.description) # Ensure string
        
        topic_desc_part = "the blog post content" # Default
        # Try to extract topic from "topic: Actual Topic (details)"
        if "topic:" in artifact_desc.lower(): # Case insensitive check for "topic:"
            parts = artifact_desc.split("topic:", 1)
            if len(parts) > 1:
                segment = parts[1].strip()
                topic_desc_part = segment.split(" (", 1)[0].strip()
        # Fallback for "Content with DALL-E image for topic: X" or "SEO optimized draft for topic: Y"
        elif artifact_desc.lower().startswith("content with") or artifact_desc.lower().startswith("seo optimized draft for") or artifact_desc.lower().startswith("blog post draft for") or artifact_desc.lower().startswith("final output for"):
             # Split by "for topic:" or just "for" then take the rest
             if "for topic:" in artifact_desc.lower():
                 parts = artifact_desc.lower().split("for topic:", 1)
             else:
                 parts = artifact_desc.lower().split("for ", 1)
                 
             if len(parts) > 1:
                 segment = parts[1].strip()
                 topic_desc_part = segment.split(" (", 1)[0].strip()
                 # Capitalize if it was lowercased by split, but respect original casing from segment if possible
                 # This part is tricky; for now, let's assume we want title case from the extracted segment
                 original_segment_parts = artifact_desc.split(parts[0], 1) # re-split original to get casing
                 if len(original_segment_parts) > 1:
                     original_topic_match_segment = original_segment_parts[1].strip().split(" (", 1)[0].strip()
                     # Heuristic: if the lowercased topic is found in original segment, use original casing part
                     if topic_desc_part.lower() in original_topic_match_segment.lower(): 
                        # find the start index of topic_desc_part in original_topic_match_segment (case insensitive)
                        start_index = original_topic_match_segment.lower().find(topic_desc_part.lower())
                        if start_index != -1:
                            topic_desc_part = original_topic_match_segment[start_index : start_index + len(topic_desc_part)]

        # Fallback for "Some text artifact about Actual Topic"
        elif " about " in artifact_desc.lower():
            parts = artifact_desc.lower().split(" about ", 1)
            if len(parts) > 1:
                segment = parts[1].strip()
                topic_desc_part = segment.split(" (", 1)[0].strip().split(", which",1)[0].strip().split(". ",1)[0].strip()
                # Try to get original casing for this more general extraction
                # Use re.split if flags are needed, otherwise str.split is fine
                # For a simple prefix split to get original casing, str.split might be sufficient if prefix is known
                # However, the original code was trying to use re.IGNORECASE effectively.
                # If parts[0] is derived from lowercased string, we need a case-insensitive split or find on original.
                
                # Simpler approach to find the segment in original description and extract with original casing:
                lower_artifact_desc = artifact_desc.lower()
                lower_segment_to_find = parts[0].lower() + " about " + topic_desc_part.lower()
                start_index = lower_artifact_desc.find(parts[0].lower() + " about ")
                if start_index != -1:
                    # Find where the actual topic starts after "about "
                    topic_start_index = start_index + len(parts[0]) + len(" about ")
                    # Extract the segment corresponding to topic_desc_part from original string
                    original_topic_extraction = artifact_desc[topic_start_index : topic_start_index + len(topic_desc_part)]
                    if original_topic_extraction.lower() == topic_desc_part.lower(): # Verify we got the right part
                        topic_desc_part = original_topic_extraction
                    else: # Fallback to title case if precise original casing extraction fails
                         topic_desc_part = ' '.join(word.capitalize() for word in topic_desc_part.split() if word)
                else: # Fallback to title case if prefix not found in original (should not happen if parts[0] came from it)
                    topic_desc_part = ' '.join(word.capitalize() for word in topic_desc_part.split() if word) 

        image_prompt = f"A compelling and professional main illustration for a blog post about '{topic_desc_part}'. The image should be visually engaging, conceptually relevant to the topic, and suitable for a featured image. Avoid text in the image. Digital art, vibrant, and modern style."
        logger.info(f"{self.card.name} generating image with DALL-E for topic: '{topic_desc_part}'. Prompt snippet: '{image_prompt[:150]}...'")
        
        image_urls = await self.generate_image_with_dalle(image_prompt, n_images=1)

        image_markdown_references = ""
        if image_urls:
            for i, url in enumerate(image_urls):
                # Ensure title in markdown is escaped properly if topic contains special characters
                escaped_topic_desc = topic_desc_part.replace('"', '\\"')
                image_markdown_references += f"\n![Generated illustration for {escaped_topic_desc}]({url} \"{escaped_topic_desc} - AI Generated Image {i+1}\")\n"
                logger.info(f"{self.card.name}: Generated DALL-E image URL for '{topic_desc_part}': {url}")
        else:
            logger.warning(f"{self.card.name}: Failed to generate images with DALL-E for '{topic_desc_part}'. Using placeholder text.")
            image_markdown_references = f"\n*[Image generation failed for '{topic_desc_part}'. Placeholder for a relevant image.]*\n"

        content_with_images = f"""
{original_content}

## Featured Image
{image_markdown_references}
*[Image created by {self.card.name} using OpenAI DALL-E for '{topic_desc_part}']*
"""

        output_artifact = self.create_artifact(
            task_id=task.task_id,
            content_type="text/markdown",
            data=content_with_images.strip(), # Strip to remove newlines from triple quotes
            description=f"Content with DALL-E image for topic: {topic_desc_part}"
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
