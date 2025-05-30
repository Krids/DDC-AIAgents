import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from agents.content_research_agent import ContentResearchAgent
from agents.base_agent import Task, Artifact, TaskStatus
from protocols.a2a_schemas import AgentMessage

@pytest.fixture
def research_agent_instance():
    """Provides a ContentResearchAgent instance for testing."""
    return ContentResearchAgent()

@pytest.mark.asyncio
async def test_content_research_agent_initialization(research_agent_instance: ContentResearchAgent):
    assert research_agent_instance.card.name == "Content Research Agent"
    assert "research_topic_web" in [cap.skill_name for cap in research_agent_instance.card.capabilities]
    assert research_agent_instance.web_search_tool is None

def test_set_web_search_tool(research_agent_instance: ContentResearchAgent):
    mock_tool = AsyncMock()
    research_agent_instance.set_web_search_tool(mock_tool)
    assert research_agent_instance.web_search_tool == mock_tool

@pytest.mark.asyncio
async def test_perform_web_search_with_tool_success(research_agent_instance: ContentResearchAgent):
    mock_search_tool = AsyncMock(return_value={
        "web_search_response": {
            "results": [
                {"title": "Test Title 1", "url": "http://example.com/1", "content": "Test snippet 1"},
                {"title": "Test Title 2", "url": "http://example.com/2", "snippet": "Test snippet 2 long content to be truncated" * 10},
            ]
        }
    })
    research_agent_instance.set_web_search_tool(mock_search_tool)
    results = await research_agent_instance.perform_web_search("test query", num_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "Test Title 1"
    assert results[0]["snippet"] == "Test snippet 1..." # Appends ...
    assert results[1]["title"] == "Test Title 2"
    assert results[1]["snippet"].startswith("Test snippet 2 long content")
    assert results[1]["snippet"].endswith("...")
    assert len(results[1]["snippet"]) <= 203 # 200 + "..."
    mock_search_tool.assert_awaited_once_with(query="test query", explanation="Content research for topic test query")

@pytest.mark.asyncio
async def test_perform_web_search_with_tool_api_error(research_agent_instance: ContentResearchAgent, caplog):
    import logging
    caplog.set_level(logging.ERROR)
    mock_search_tool = AsyncMock(side_effect=Exception("API Communication Error"))
    research_agent_instance.set_web_search_tool(mock_search_tool)
    
    results = await research_agent_instance.perform_web_search("error query")
    assert len(results) == 0
    assert "Error during web search for 'error query'" in caplog.text
    assert "API Communication Error" in caplog.text

@pytest.mark.asyncio
async def test_perform_web_search_with_tool_malformed_response(research_agent_instance: ContentResearchAgent, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    mock_search_tool = AsyncMock(return_value={"some_other_key": "no_results_here"})
    research_agent_instance.set_web_search_tool(mock_search_tool)

    results = await research_agent_instance.perform_web_search("malformed query")
    assert len(results) == 0
    assert "Web search for 'malformed query' returned unexpected structure or no results." in caplog.text
    
@pytest.mark.asyncio
async def test_perform_web_search_no_tool_simulated(research_agent_instance: ContentResearchAgent, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    research_agent_instance.web_search_tool = None # Ensure no tool
    
    results = await research_agent_instance.perform_web_search("simulated query", num_results=1)
    assert len(results) == 1
    assert results[0]["title"] == "Recent breakthrough in simulated query"
    assert "Web search tool not set. Using SIMULATED search" in caplog.text

@pytest.mark.asyncio
async def test_process_task_success_with_web_search(research_agent_instance: ContentResearchAgent):
    mock_search_tool = AsyncMock(return_value={
        "web_search_response": {
            "results": [{"title": "AI News", "url": "http://news.ai", "content": "Latest AI developments."}]
        }
    })
    research_agent_instance.set_web_search_tool(mock_search_tool)
    research_agent_instance.set_message_handler(AsyncMock())

    topic_artifact = Artifact(
        artifact_id="topic_artifact_1", task_id="t1", creator_agent_id="o",
        content_type="text/plain", data="AI in 2024", description="Blog topic",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = research_agent_instance.create_task(
        description="Research AI in 2024",
        initiator_agent_id="orchestrator",
        input_artifacts=[topic_artifact]
    )

    await research_agent_instance.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_artifact = task.output_artifacts[0]
    assert output_artifact.content_type == "text/markdown"
    assert "## Web Research Summary for: AI in 2024" in output_artifact.data
    assert "AI News" in output_artifact.data
    assert "http://news.ai" in output_artifact.data
    mock_search_tool.assert_awaited_once()
    research_agent_instance.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_simulated_search_if_no_tool(research_agent_instance: ContentResearchAgent, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    research_agent_instance.web_search_tool = None # No tool
    research_agent_instance.set_message_handler(AsyncMock())

    topic_artifact = Artifact(
        artifact_id="topic_sim", task_id="t_sim", creator_agent_id="o",
        data="Future Tech", description="Blog topic",
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = research_agent_instance.create_task(
        description="Research Future Tech",
        initiator_agent_id="orchestrator",
        input_artifacts=[topic_artifact]
    )

    await research_agent_instance.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert "Using SIMULATED search" in caplog.text # From perform_web_search
    assert "## Web Research Summary for: Future Tech" in task.output_artifacts[0].data
    assert "Recent breakthrough in recent news on Future Tech" in task.output_artifacts[0].data # Simulated result title
    research_agent_instance.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_no_input_artifact(research_agent_instance: ContentResearchAgent, caplog):
    import logging
    caplog.set_level(logging.ERROR)
    research_agent_instance.set_message_handler(AsyncMock())
    task = research_agent_instance.create_task(
        description="Research without topic",
        initiator_agent_id="orchestrator"
    )

    await research_agent_instance.process_task(task)

    assert task.status == TaskStatus.FAILED
    assert f"Research task {task.task_id} for {research_agent_instance.card.name} has no input artifacts" in caplog.text
    research_agent_instance.message_handler.assert_awaited_once() # Status update should still be sent

@pytest.mark.asyncio
async def test_process_task_web_search_fails(research_agent_instance: ContentResearchAgent, caplog):
    import logging
    caplog.set_level(logging.WARNING) # To catch no results warning
    mock_search_tool = AsyncMock(return_value={"web_search_response": {"results": []}}) # Tool returns no results
    research_agent_instance.set_web_search_tool(mock_search_tool)
    research_agent_instance.set_message_handler(AsyncMock())

    topic_artifact = Artifact(
        artifact_id="topic_fail", task_id="t_fail", creator_agent_id="o",
        data="Obscure Topic", description="Blog topic",
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = research_agent_instance.create_task(
        description="Research Obscure Topic",
        initiator_agent_id="orchestrator",
        input_artifacts=[topic_artifact]
    )

    await research_agent_instance.process_task(task)
    
    assert task.status == TaskStatus.COMPLETED # Still completes with fallback
    assert len(task.output_artifacts) == 1
    output_data = task.output_artifacts[0].data
    assert "No relevant recent news found or web search failed" in output_data
    assert "Falling back to general information about **Obscure Topic** (simulated content - web search did not yield results for current news):" in output_data
    assert "Information about Obscure Topic indicates its growing importance" in output_data # Check for part of the fallback content
    assert "This research provides an overview of recent findings for content creation on 'Obscure Topic'." in output_data

    mock_search_tool.assert_awaited_once_with(query="recent news on Obscure Topic", explanation="Content research for topic recent news on Obscure Topic")
    research_agent_instance.message_handler.assert_awaited_once() 