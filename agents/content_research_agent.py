import asyncio
import logging
import os # Added for os.getenv
from apify_client import ApifyClientAsync # Added
import json # Added for JSON saving
from datetime import datetime # Added for timestamp
from agents.base_agent import BaseAgent, Task, TaskStatus, Artifact
from typing import List, Dict, Optional # Callable removed
from utils.json_utils import convert_datetime_to_iso_string # Added

logger = logging.getLogger(f"agentsAI.{__name__}")

class ContentResearchAgent(BaseAgent):
    """
    Researches topics and gathers information using an Apify actor.
    """
    AGENT_DATA_SUBFOLDER = "content_research"

    def __init__(self, agent_id: str = "research_agent_001", name: str = "Content Research Agent",
                 description: str = "Researches topics and gathers information using an Apify Actor.",
                 data_dir_override: Optional[str] = None, **kwargs): # Added data_dir_override
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)
        self.data_dir_override = data_dir_override # Store it
        self.register_capability(
            skill_name="research_topic_apify", # Changed capability name
            description="Performs research on a given topic using an Apify actor and returns a summary.",
            input_schema={"type": "object", "properties": {"topic_artifact_id": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"research_summary_artifact_id": {"type": "string"}}}
        )
        # self.web_search_tool: Optional[Callable] = None # Removed
        self.apify_client: Optional[ApifyClientAsync] = None
        try:
            apify_token = os.getenv("APIFY_API_TOKEN")
            if not apify_token:
                logger.warning(f"{self.card.name}: APIFY_API_TOKEN not found in environment. Content Research Agent will use fallback/simulated data.")
            else:
                self.apify_client = ApifyClientAsync(apify_token)
                logger.info(f"ApifyClientAsync initialized for {self.card.name}")
        except Exception as e:
            logger.error(f"Error initializing ApifyClientAsync for {self.card.name}: {e}. Will use fallback/simulated data.", exc_info=True)

    async def get_research_from_apify(self, research_query: str, task_id_for_log: Optional[str] = None, max_results: int = 3) -> List[Dict[str, str]]:
        if not self.apify_client:
            logger.warning(f"{self.card.name}: Apify client not available. Returning SIMULATED research for query '{research_query}'.")
            await asyncio.sleep(0.5)
            return [
                {"title": f"Simulated Apify Result 1: {research_query}", "url": "http://sim.example.com/1", "summary": f"This is simulated detailed content about {research_query} from an Apify actor."},
                {"title": f"Simulated Apify Result 2: Overview of {research_query}", "url": "http://sim.example.com/2", "summary": f"Another piece of simulated research focusing on key aspects of {research_query}."},
            ][:max_results]

        ACTOR_ID = "uNMHGOGRawDYkIXmg" # User specified Apify actor for research
        logger.info(f"{self.card.name}: Calling Apify actor '{ACTOR_ID}' for research query: '{research_query}'")
        
        run_details_to_save = None # Initialize to None
        actor_run_details = None # Ensure initialized for the finally block

        try:
            # Input schema based on user feedback and error message
            actor_input = { 
                "feeds": ["http://feeds.bbci.co.uk/news/technology/rss.xml"], # Default feed
                "query": research_query, # Use the agent's research query for Apify's query field
                # "maxItems": max_results, # Assuming this actor might not use maxItems if it processes feeds
                                         # Or it might have a different parameter for result limits per feed or overall.
                                         # For now, let's omit it and see if the actor has its own default or if it processes all feed items.
            }
            # If max_results is important, you'd need to find the correct parameter name for this specific actor.
            # Common names could be maxItems, maxResults, limit, etc.
            if max_results > 0: # A common way to specify a limit if the actor supports it
                actor_input["maxArticles"] = max_results # Example: trying "maxArticles", actual param may vary
                # actor_input["maxItemsPerFeed"] = max_results # Another common possibility

            logger.debug(f"{self.card.name}: Apify actor input: {actor_input}")
            
            actor_client = self.apify_client.actor(ACTOR_ID)
            logger.debug(f"{self.card.name}: Apify actor client for {ACTOR_ID} obtained. Calling actor with input: {actor_input}")
            run = await actor_client.call(run_input=actor_input, memory_mbytes=512, timeout_secs=180) # Increased memory/timeout
            actor_run_details = run

            if not run or not run.get("defaultDatasetId"):
                logger.error(f"{self.card.name}: Apify actor run {run.get('id') if run else 'N/A'} for query '{research_query}' did not return a valid defaultDatasetId. Run details: {run}")
                return self._get_fallback_research(research_query, max_results)

            logger.info(f"{self.card.name}: Apify actor run {run['id']} for '{research_query}' completed with status {run.get('status')}. Fetching dataset {run['defaultDatasetId']}.")
            dataset_client = self.apify_client.dataset(run["defaultDatasetId"])
            
            results = []
            async for item in dataset_client.iterate_items():
                if isinstance(item, dict):
                    # Try to extract common fields for research results
                    title = item.get("title", item.get("name", "N/A"))
                    url = item.get("url", item.get("source_url", "N/A"))
                    # Prefer 'summary' or 'text', then 'content', then 'description'
                    summary_fields = ["summary", "text", "content", "description", "snippet"]
                    summary = "N/A"
                    for field in summary_fields:
                        if item.get(field) and isinstance(item[field], str):
                            summary = item[field][:500] + "..." if len(item[field]) > 500 else item[field]
                            break
                    
                    if title != "N/A" or url != "N/A" or summary != "N/A":
                        results.append({"title": title, "url": url, "summary": summary})
                        logger.debug(f"Extracted research item from Apify: {title[:50]}...")
                
                if len(results) >= max_results:
                    logger.debug(f"Reached max_results ({max_results}) for Apify research. Stopping item iteration.")
                    break
            
            if not results:
                 logger.warning(f"No structured results extracted from Apify for query '{research_query}'. Using fallback.")
                 return self._get_fallback_research(research_query, max_results)
            
            logger.info(f"{self.card.name}: Extracted {len(results)} research items from Apify for query '{research_query}'.")
            return results

        except Exception as e:
            logger.error(f"{self.card.name}: Error calling Apify actor '{ACTOR_ID}' or processing research results for query '{research_query}': {e}", exc_info=True)
            # run_details_to_save might be set if error happened after actor_client.call() but before return
            # or it might be an exception object if call itself failed.
            # Ensure actor_run_details is a dict for consistent saving, even if it was an exception.
            if e is not None: # If an exception occurred and was caught
                if actor_run_details is None:
                    actor_run_details = {} # Ensure it's a dict
                actor_run_details["error"] = str(e)
                actor_run_details["traceback"] = logging.Formatter().formatException(logging.sys.exc_info())

            return self._get_fallback_research(research_query, max_results)
        finally:
            if actor_run_details is not None:
                try:
                    # Determine the base path for saving data
                    base_save_path = self.data_dir_override
                    if base_save_path is None: # Normal operation, not overridden by test
                        base_save_path = os.path.join("data", self.AGENT_DATA_SUBFOLDER)
                    
                    os.makedirs(base_save_path, exist_ok=True) # Create directory if it doesn't exist

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_id = task_id_for_log if task_id_for_log else "unknown_task"
                    filename = os.path.join(base_save_path, f"apify_research_{self.agent_id}_{log_id}_{ts}.json") # Use base_save_path
                    
                    serializable_data_to_save = {}
                    if isinstance(actor_run_details, dict):
                        serializable_data_to_save = convert_datetime_to_iso_string(actor_run_details)
                    elif hasattr(actor_run_details, 'status') and hasattr(actor_run_details, 'id'): # Apify Run like object
                        temp_dict = {attr: getattr(actor_run_details, attr) for attr in dir(actor_run_details) if not attr.startswith('_') and not callable(getattr(actor_run_details, attr))}
                        serializable_data_to_save = convert_datetime_to_iso_string(temp_dict)
                    else: # Fallback for other types
                        serializable_data_to_save = {"raw_data": str(actor_run_details)}

                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(serializable_data_to_save, f, indent=4, ensure_ascii=False)
                    logger.info(f"Saved Apify research actor response/details to {filename}")
                except Exception as log_e:
                    logger.error(f"Failed to save Apify research actor response to file: {log_e}", exc_info=True)

    def _get_fallback_research(self, query: str, num_results: int) -> List[Dict[str, str]]:
        logger.warning(f"{self.card.name}: Providing fallback/simulated research data for query: '{query}'.")
        return [
                {"title": f"Fallback Result 1: Exploring {query}", "url": "http://fallback.example.com/1", "summary": f"General fallback information about {query}, as Apify interaction failed or yielded no results."},
                {"title": f"Fallback Result 2: Understanding {query}", "url": "http://fallback.example.com/2", "summary": f"Key concepts and overview related to {query}, provided as a fallback measure."},
            ][:num_results]

    async def process_task(self, task: Task):
        logger.info(f"{self.card.name} ({self.agent_id}) starting task: {task.description} (using Apify research)")
        self.update_task_status(task, TaskStatus.IN_PROGRESS)

        if not task.input_artifacts:
            logger.error(f"Error: Research task {task.task_id} for {self.card.name} has no input artifacts (topic).")
            self.update_task_status(task, TaskStatus.FAILED)
            if task.initiator_agent_id and self.message_handler:
                await self.send_message(receiver_agent_id=task.initiator_agent_id, message_type="task_status_update", payload=task.model_dump())
            return

        topic_artifact = task.input_artifacts[0]
        original_topic = str(topic_artifact.data)
        
        # For Apify, the query might be the topic itself, or a question about it.
        # Let's use a slightly more direct query for an actor that might expect it.
        research_query = f"Comprehensive overview and recent developments in {original_topic}"
        
        logger.info(f"{self.card.name} performing Apify research for query: '{research_query}' (related to original topic: '{original_topic}')")
        
        # Pass task.task_id for logging purposes in the filename
        apify_results = await self.get_research_from_apify(research_query, task_id_for_log=task.task_id, max_results=3)
        await asyncio.sleep(0.1) 

        research_summary_content = f"## Apify Research Summary for: {original_topic}\\n\\n"
        research_summary_content += f"Based on research using Apify actor for the query **'{research_query}'**, here are the findings:\\n\\n"
        
        if apify_results:
            for i, result in enumerate(apify_results):
                research_summary_content += f"### Result {i+1}: {result.get('title', 'N/A')}\\n"
                research_summary_content += f"- **Source URL**: <{result.get('url', 'N/A')}>\\n" # Keep URL if available
                research_summary_content += f"- **Summary**: {result.get('summary', 'N/A')}\\n\\n"
            logger.info(f"Processed {len(apify_results)} results from Apify for query '{research_query}'.")
        else:
            logger.warning(f"No relevant information found or Apify research failed for query '{research_query}'.")
            research_summary_content += "No relevant information found via Apify actor, or the process encountered an error.\\n"
            # Fallback to a more generic statement if Apify yields nothing
            research_summary_content += f"Further investigation or alternative research methods may be needed for **{original_topic}**.\\n"

        research_summary_content += f"\\nThis research provides an overview of findings via Apify for content creation on '{original_topic}'."

        output_artifact = self.create_artifact(
            task_id=task.task_id,
            content_type="text/markdown",
            data=research_summary_content,
            description=f"Apify research summary for topic: {original_topic}"
        )
        self.add_output_artifact_to_task(task, output_artifact)
        self.update_task_status(task, TaskStatus.COMPLETED)
        logger.info(f"{self.card.name} ({self.agent_id}) completed Apify research task: {task.description}. Artifact: {output_artifact.artifact_id}")

        if task.initiator_agent_id and self.message_handler and task.initiator_agent_id != self.agent_id:
            await self._send_status_update(task) 