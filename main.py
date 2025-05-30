import asyncio
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
import logging
from typing import Optional

import core.logger # Initialize logger configuration
from core.agent_factory import AgentFactory

from agents import (
    BaseAgent, # For type hint and isinstance checks
    OrchestratorAgent, # For isinstance checks and specific methods
    ContentResearchAgent # For isinstance checks and set_web_search_tool
    # WritingAgent, SEOAgent, ImageAgent removed as specific types not directly used here
)
from protocols.a2a_schemas import Task, TaskStatus

logger = logging.getLogger(f"agentsAI.{__name__}")

class MockDefaultAPI:
    async def web_search(self, query: str, explanation: str): # explanation default removed
        logger.info(f"MOCK web_search called for query: {query} (Explanation: {explanation}). Returning simulated data.")
        await asyncio.sleep(0.1)
        return {
            "web_search_response": {
                "results": [
                    {"title": f"Mock Result 1 for {query}", "url": "http://mock.example.com/1", "content": "Simulated content for query 1."},
                    {"title": f"Mock Result 2 for {query}", "url": "http://mock.example.com/2", "content": "Simulated content for query 2."}
                ]
            }
        }

default_api_instance = None
try:
    # Attempt to import the actual default_api if available in the environment
    import default_api # type: ignore
    default_api_instance = default_api
    logger.info("Successfully imported and using real default_api.")
except ImportError:
    logger.warning("Warning: default_api not found or not usable. Using MockDefaultAPI for web_search.")
    default_api_instance = MockDefaultAPI() # Fallback to mock

async def run_blog_creation_workflow():
    load_dotenv()
    logger.info("Environment variables loaded from .env file")

    try:
        orchestrator: OrchestratorAgent = AgentFactory.create_agent("orchestrator") # type: ignore
        research_agent: ContentResearchAgent = AgentFactory.create_agent("content_research") # type: ignore
        writing_agent = AgentFactory.create_agent("writing")
        seo_agent = AgentFactory.create_agent("seo")
        image_agent = AgentFactory.create_agent("image")
        
        # Ensure correct types for agents where specific methods are called.
        # Factory should return correct types, but this confirms for linters/safety.
        if not isinstance(orchestrator, OrchestratorAgent):
            logger.critical(f"Orchestrator is not of type OrchestratorAgent, but {type(orchestrator)}. Aborting.")
            return
        if not isinstance(research_agent, ContentResearchAgent):
            logger.critical(f"Research agent is not of type ContentResearchAgent, but {type(research_agent)}. Aborting.")
            return

    except ValueError as e:
        logger.critical(f"Failed to create agents using AgentFactory: {e}", exc_info=True)
        return
    except Exception as e: # Catch any other exception during agent creation
        logger.critical(f"An unexpected error occurred during agent creation: {e}", exc_info=True)
        return
    
    # Equip the research agent with the web search tool
    if hasattr(default_api_instance, 'web_search'):
        if hasattr(research_agent, 'set_web_search_tool'):
            # Check if the tool is async or needs wrapping
            if asyncio.iscoroutinefunction(default_api_instance.web_search):
                research_agent.set_web_search_tool(default_api_instance.web_search)
                logger.info("Async web search tool has been set for ContentResearchAgent.")
            else:
                # Wrap synchronous tool to be awaitable
                async def async_web_search_wrapper(*args, **kwargs):
                    loop = asyncio.get_event_loop()
                    logger.debug("Using synchronous web_search_tool wrapped in async executor.")
                    # Ensure the lambda captures args and kwargs correctly for the executor
                    return await loop.run_in_executor(None, lambda: default_api_instance.web_search(*args, **kwargs))
                research_agent.set_web_search_tool(async_web_search_wrapper)
                logger.info("Synchronous web search tool (wrapped to async) has been set for ContentResearchAgent.")
        else:
            logger.warning(f"Research agent (type {type(research_agent)}) does not have 'set_web_search_tool' method.")
    else:
        logger.warning("default_api_instance.web_search not available. ContentResearchAgent may use simulated search.")

    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(writing_agent)
    orchestrator.register_agent(seo_agent)
    orchestrator.register_agent(image_agent)

    blog_topic = "The Future of Multi-Agent AI Systems"
    logger.info(f"Starting blog post generation for topic: '{blog_topic}'")

    # The orchestrator needs its own message handler to process status updates for tasks it assigned to itself
    # or tasks it assigned to other agents when those agents send updates back.
    orchestrator.set_message_handler(orchestrator.route_message)
    logger.debug("Orchestrator's message handler set to its own route_message method.")

    # Create the primary task for the orchestrator to manage the entire workflow
    # This task is what triggers orchestrator.process_task -> execute_blog_post_workflow
    system_initiator_id = f"system_initiator_{uuid.uuid4()}"
    orchestrator_workflow_task = orchestrator.create_task(
        description=f"Create a blog post on topic: {blog_topic}", # This description is key for orchestrator.process_task
        initiator_agent_id=system_initiator_id, # Marks the "system" or main.py as the ultimate initiator
        assigned_to_agent_id=orchestrator.agent_id # Task is assigned to the orchestrator itself
    )
    logger.info(f"Master workflow task {orchestrator_workflow_task.task_id} created for orchestrator.")

    # Send this master task to the orchestrator for processing
    # This is slightly different from assign_task_and_wait as this is the *initial* trigger.
    # We will simulate this by directly calling its process_task or a similar entry point
    # For the existing setup, assign_task_and_wait on itself also works, which uses its own process_task.
    
    logger.info(f"Triggering orchestrator to process its master task {orchestrator_workflow_task.task_id}...")
    
    # The Orchestrator's process_task is designed to pick up "Create a blog post on topic: X"
    # and then call execute_blog_post_workflow. We assign this task to itself and wait for it.
    # This uses the orchestrator's own task management and callback system.
    final_workflow_task_result: Optional[Task] = await orchestrator.assign_task_and_wait(
        agent=orchestrator, # The orchestrator agent itself
        task_description=orchestrator_workflow_task.description, # Pass the description of the task it should execute
        # input_artifacts can be None if process_task extracts topic from description
    )
    
    if final_workflow_task_result and final_workflow_task_result.status == TaskStatus.COMPLETED and final_workflow_task_result.output_artifacts:
        final_artifact = final_workflow_task_result.output_artifacts[0]
        logger.info(f"Blog post workflow COMPLETED successfully! Final artifact ID: {final_artifact.artifact_id}, Content Type: {final_artifact.content_type}")
        
        output_dir = "outputs" # Ensure this matches project structure
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() else "_" for c in blog_topic).lower()
        filename = os.path.join(output_dir, f"{safe_topic}_{timestamp}.md")
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Blog Post: {blog_topic}\n\n")
                f.write(f"*(Generated on: {datetime.now().isoformat()})*\n\n")
                if isinstance(final_artifact.data, str):
                    f.write(final_artifact.data)
                else:
                    f.write(str(final_artifact.data)) # Fallback to string conversion
            logger.info(f"Blog post saved to: {filename}")
        except IOError as e:
            logger.error(f"Error saving blog post to file {filename}: {e}", exc_info=True)

    else:
        logger.error("Blog post workflow FAILED or produced no output.")
        if final_workflow_task_result:
            # Log the entire task object for details if it failed or has no artifacts
            logger.error(f"Final task details: {final_workflow_task_result.model_dump_json(indent=2)}")
        else:
            logger.error("The workflow task assigned to the orchestrator did not return a valid task result object.")

if __name__ == "__main__":
    logger.info("Starting main application execution.")
    try:
        asyncio.run(run_blog_creation_workflow())
        logger.info("Main application execution finished successfully.")
    except Exception as e: # Catch any other unhandled exception
        logger.critical(f"Unhandled exception at top level of main application: {e}", exc_info=True)
