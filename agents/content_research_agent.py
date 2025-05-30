import asyncio
import logging # Import logging
from agents.base_agent import BaseAgent, Task, TaskStatus, Artifact
# AgentMessage removed as it was only for a comment
from typing import List, Dict, Optional, Callable 

logger = logging.getLogger(f"agentsAI.{__name__}") # Child logger

# Comments about web_search tool moved into class docstring or specific methods

class ContentResearchAgent(BaseAgent):
    """
    Researches topics and gathers recent information. 
    Requires a web search tool to be set via `set_web_search_tool`.
    """
    def __init__(self, agent_id: str = "research_agent_001", name: str = "Content Research Agent",
                 description: str = "Researches topics and gathers recent information using web search.", **kwargs):
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)
        self.register_capability(
            skill_name="research_topic_web", 
            description="Performs web research for recent news on a given topic and returns a summary.",
            input_schema={"type": "object", "properties": {"topic_artifact_id": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"research_summary_artifact_id": {"type": "string"}}}
        )
        self.web_search_tool: Optional[Callable] = None

    def set_web_search_tool(self, tool_function: Callable):
        """Sets the callable web search tool for this agent."""
        self.web_search_tool = tool_function
        logger.info(f"Web search tool has been set for {self.card.name}")

    async def perform_web_search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        if not self.web_search_tool:
            logger.warning(f"{self.card.name}: Web search tool not set. Using SIMULATED search for: '{query}'")
            await asyncio.sleep(0.5) 
            sim_results_list = [
                {"title": f"Recent breakthrough in {query}", "url": f"https://example.com/news/{query.replace(' ', '-')}-1", "content": f"Details about the latest advancements in {query}..."},
                {"title": f"The impact of {query} on industry", "url": f"https://example.com/articles/{query.replace(' ', '-')}-impact", "content": f"An analysis of how {query} is changing various sectors..."},
                {"title": f"Future trends in {query} for 2024", "url": f"https://example.com/blog/{query.replace(' ', '-')}-trends", "content": f"Experts discuss what to expect from {query} in the coming year..."}
            ]
            return [
                {
                    "title": r.get("title", "N/A"),
                    "url": r.get("url", "N/A"),
                    "snippet": (r.get("content") or r.get("snippet", ""))[:200] + "...",
                }
                for r in sim_results_list[:num_results]
            ]
        try:
            logger.info(f"{self.card.name}: Performing web search via provided tool for: '{query}'")
            search_result_dict = await self.web_search_tool(query=query, explanation=f"Content research for topic {query}")
            
            if search_result_dict and \
               isinstance(search_result_dict.get("web_search_response"), dict) and \
               isinstance(search_result_dict["web_search_response"].get("results"), list):
                
                raw_results = search_result_dict["web_search_response"]["results"]
                formatted_results = [
                    {
                        "title": r.get("title", "N/A"),
                        "url": r.get("url", "N/A"),
                        "snippet": (r.get("content") or r.get("snippet", ""))[:200] + "...",
                    }
                    for r in raw_results[:num_results]
                ]
                logger.debug(f"{self.card.name}: Web search for '{query}' returned {len(formatted_results)} results.")
                return formatted_results
            else:
                logger.warning(f"{self.card.name}: Web search for '{query}' returned unexpected structure or no results. Response snippet: {str(search_result_dict)[:200]}...")
                return []

        except Exception as e:
            logger.error(f"{self.card.name}: Error during web search for '{query}': {e}", exc_info=True)
            return []

    async def process_task(self, task: Task):
        logger.info(f"{self.card.name} ({self.agent_id}) starting task: {task.description} (using web search)")
        self.update_task_status(task, TaskStatus.IN_PROGRESS)

        if not task.input_artifacts:
            logger.error(f"Error: Research task {task.task_id} for {self.card.name} has no input artifacts (topic).")
            self.update_task_status(task, TaskStatus.FAILED)
            if task.initiator_agent_id and self.message_handler:
                await self.send_message(receiver_agent_id=task.initiator_agent_id, message_type="task_status_update", payload=task.model_dump())
            return

        topic_artifact = task.input_artifacts[0]
        original_topic = str(topic_artifact.data) # Ensure string for f-string formatting
        
        search_query = f"recent news on {original_topic}"
        
        logger.info(f"{self.card.name} performing web search for query: '{search_query}' (related to original topic: '{original_topic}')")
        search_results = await self.perform_web_search(search_query, num_results=3)
        await asyncio.sleep(0.1) 

        research_summary = f"## Web Research Summary for: {original_topic}\n\n"
        research_summary += f"Based on a web search for **'{search_query}'**, here are the key findings:\n\n"
        if search_results:
            for i, result in enumerate(search_results):
                research_summary += f"### Result {i+1}: {result.get('title', 'N/A')}\n"
                research_summary += f"- **Source URL**: <{result.get('url', 'N/A')}>\n"
                research_summary += f"- **Snippet**: {result.get('snippet', 'N/A')}\n\n"
            logger.info(f"Found {len(search_results)} web results for query '{search_query}'.")
        else:
            logger.warning(f"No relevant recent news found or web search failed for query '{search_query}'.")
            research_summary += "No relevant recent news found or web search failed.\n"
            research_summary += f"Falling back to general information about **{original_topic}** (simulated content - web search did not yield results for current news):\n"
            research_summary += (
                f"- **General Point 1**: Information about {original_topic} indicates its growing importance in the current technological landscape.\n"
                f"- **General Statistic**: Data suggests that adoption of {original_topic} related technologies has increased by X% in the last year.\n"
                f"- **Key Insight**: Experts believe that {original_topic} will continue to be a major driver of innovation in the coming years.\n"
            )

        research_summary += f"\nThis research provides an overview of recent findings for content creation on '{original_topic}'."

        output_artifact = self.create_artifact(
            task_id=task.task_id,
            content_type="text/markdown",
            data=research_summary,
            description=f"Web research summary for topic: {original_topic}"
        )
        self.add_output_artifact_to_task(task, output_artifact)
        self.update_task_status(task, TaskStatus.COMPLETED)
        logger.info(f"{self.card.name} ({self.agent_id}) completed web research task: {task.description}. Artifact: {output_artifact.artifact_id}")

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