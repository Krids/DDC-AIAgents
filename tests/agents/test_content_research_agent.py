import pytest
import asyncio
import os
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from pathlib import Path # For tmp_path

from apify_client import ApifyClientAsync
from apify_client._errors import ApifyApiError

from agents.content_research_agent import ContentResearchAgent
from agents.base_agent import Task, Artifact, TaskStatus
from core.agent_factory import create_agent # For using the factory

@pytest.fixture
def content_research_agent_mock_apify(monkeypatch, tmp_path: Path): # Added tmp_path
    monkeypatch.setenv("APIFY_API_TOKEN", "fake_token_for_research_tests")
    
    mock_apify_client_instance = AsyncMock(spec=ApifyClientAsync)
    mock_actor_sub_client = AsyncMock()
    mock_apify_client_instance.actor = MagicMock(return_value=mock_actor_sub_client)
    mock_dataset_sub_client = AsyncMock()
    mock_apify_client_instance.dataset = MagicMock(return_value=mock_dataset_sub_client)

    # Patch the ApifyClientAsync that would be instantiated inside the agent
    with patch("agents.content_research_agent.ApifyClientAsync", return_value=mock_apify_client_instance) as mock_apify_constructor:
        agent = create_agent(
            "ContentResearchAgent", 
            use_tmp_path=True, 
            tmp_path=tmp_path
        )
    assert agent is not None, "Failed to create ContentResearchAgent via factory"
    # Since ApifyClientAsync is initialized within the agent's __init__ based on env var,
    # and we want to control the mock instance, we ensure the patched constructor was called if token was present,
    # and then we can directly assign our fully mocked client if needed, or rely on the patch.
    # For this fixture, the token IS present via monkeypatch.setenv.
    mock_apify_constructor.assert_called_once() 
    # The agent.apify_client should be the one from the patch.
    # If further control is needed (e.g. specific return values for actor/dataset not set above), set them on agent.apify_client here.
    return agent

@pytest.fixture
def content_research_agent_no_apify_token(caplog, tmp_path: Path): # Added tmp_path
    caplog.set_level(logging.WARNING)
    with patch('agents.content_research_agent.os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key: None if key == "APIFY_API_TOKEN" else os.environ.get(key)
        instance = create_agent(
            "ContentResearchAgent",
            use_tmp_path=True,
            tmp_path=tmp_path
        )
    assert instance is not None, "Failed to create agent using factory in no_apify_token fixture"
    # No need to yield, instance is returned directly. Fixture is not async.
    return instance

@pytest.fixture
def research_agent_instance(content_research_agent_mock_apify: ContentResearchAgent):
    """Provides a ContentResearchAgent instance for testing, with Apify mocked."""
    return content_research_agent_mock_apify

@pytest.mark.asyncio
async def test_content_research_agent_initialization(research_agent_instance: ContentResearchAgent):
    assert research_agent_instance.card.name == "Content Research Agent"
    assert "research_topic_apify" in [cap.skill_name for cap in research_agent_instance.card.capabilities]
    assert research_agent_instance.apify_client is not None

@pytest.mark.asyncio
async def test_content_research_agent_initialization_no_token(content_research_agent_no_apify_token: ContentResearchAgent, caplog):
    agent = content_research_agent_no_apify_token
    
    assert "research_topic_apify" in [cap.skill_name for cap in agent.card.capabilities]
    assert agent.apify_client is None
    
    # Check logs from the setup phase of the fixture where the agent is initialized
    init_warnings = [r for r in caplog.get_records(when='setup') if r.levelname == 'WARNING' and "APIFY_API_TOKEN not found" in r.message]
    assert len(init_warnings) > 0, "APIFY_API_TOKEN not found warning was not logged during agent initialization"

@pytest.mark.asyncio
async def test_get_research_from_apify_success(content_research_agent_mock_apify: ContentResearchAgent):
    agent = content_research_agent_mock_apify
    query = "test query"
    mock_run_result = {"id": "run_id_1", "defaultDatasetId": "dataset_id_1", "status": "SUCCEEDED"}
    agent.apify_client.actor.return_value.call = AsyncMock(return_value=mock_run_result)
    
    mock_items = [{"title": "Mocked Apify Result 1 for test query", "url": "http://mock.example.com/1"}, 
                  {"title": "Mocked Apify Result 2 for test query", "url": "http://mock.example.com/2"}]
    async def mock_iterate_items_func():
        for item in mock_items: yield item
    agent.apify_client.dataset.return_value.iterate_items = MagicMock(return_value=mock_iterate_items_func())

    results = await agent.get_research_from_apify(query, max_results=2, task_id_for_log="task_log_1")
    assert len(results) == 2
    assert results[0]["title"] == "Mocked Apify Result 1 for test query"
    assert "http://mock.example.com/1" in results[0]["url"]
    agent.apify_client.actor.assert_called_once_with("uNMHGOGRawDYkIXmg")
    agent.apify_client.actor.return_value.call.assert_awaited_once()
    actor_call_args = agent.apify_client.actor.return_value.call.call_args
    assert actor_call_args.kwargs['run_input']['query'] == query
    assert actor_call_args.kwargs['run_input']['maxArticles'] == 2
    agent.apify_client.dataset.assert_called_once_with("dataset_id_1")
    agent.apify_client.dataset.return_value.iterate_items.assert_called_once()

@pytest.mark.asyncio
async def test_get_research_from_apify_api_error(content_research_agent_mock_apify: ContentResearchAgent, caplog):
    agent = content_research_agent_mock_apify
    
    mock_error_response = MagicMock()
    mock_error_response.text = "Mocked Apify API Error Details"
    agent.apify_client.actor.return_value.call = AsyncMock(side_effect=ApifyApiError(mock_error_response, attempt=1))
    caplog.set_level(logging.WARNING)
    query = "error query"
    results = await agent.get_research_from_apify(query, task_id_for_log="task_log_2")
    assert len(results) == 2
    assert "Fallback Result 1: Exploring error query" in results[0]["title"]
    assert "Error calling Apify actor uNMHGOGRawDYkIXmg for query 'error query'" in caplog.text
    assert "Mocked Apify API Error Details" in caplog.text

@pytest.mark.asyncio
async def test_get_research_from_apify_no_dataset_id(content_research_agent_mock_apify: ContentResearchAgent, caplog):
    agent = content_research_agent_mock_apify
    agent.apify_client.actor.return_value.call = AsyncMock(return_value={"id": "run123", "defaultDatasetId": None, "status": "SUCCEEDED"})
    caplog.set_level(logging.WARNING)
    query = "no dataset id query"
    results = await agent.get_research_from_apify(query, task_id_for_log="task_log_3")
    assert len(results) == 2
    assert "Fallback Result 1: Exploring no dataset id query" in results[0]["title"]
    assert "Apify actor run run123 for query 'no dataset id query' did not return a valid defaultDatasetId." in caplog.text
    assert "Providing fallback/simulated research data for query: 'no dataset id query'" in caplog.text

@pytest.mark.asyncio
async def test_get_research_from_apify_no_items(content_research_agent_mock_apify: ContentResearchAgent, caplog):
    agent = content_research_agent_mock_apify
    mock_run_result_empty_ds = {"id": "run_empty_ds_id", "defaultDatasetId": "dataset_empty_id_actual", "status": "SUCCEEDED"}
    agent.apify_client.actor.return_value.call = AsyncMock(return_value=mock_run_result_empty_ds)
    async def mock_empty_iterate_items():
        if False:
            yield {}
    agent.apify_client.dataset.return_value.iterate_items = MagicMock(return_value=mock_empty_iterate_items())
    caplog.set_level(logging.WARNING)
    query = "no items query"
    results = await agent.get_research_from_apify(query, task_id_for_log="task_log_4")
    assert len(results) == 2
    assert "Fallback Result 1: Exploring no items query" in results[0]["title"]
    assert "No structured results extracted from Apify for query 'no items query'. Using fallback." in caplog.text
    assert "Providing fallback/simulated research data for query: 'no items query'" in caplog.text

@pytest.mark.asyncio
async def test_get_research_from_apify_client_none_uses_fallback(content_research_agent_no_apify_token: ContentResearchAgent, caplog):
    agent = content_research_agent_no_apify_token
    assert agent.apify_client is None
    caplog.set_level(logging.WARNING)
    query = "client none query"
    results = await agent.get_research_from_apify(query, max_results=1, task_id_for_log="task_log_5")
    assert len(results) == 1
    # This path in agent.get_research_from_apify returns SIMULATED data, not the _get_fallback_research data
    assert "Simulated Apify Result 1: client none query" in results[0]["title"] 
    assert "Apify client not available. Returning SIMULATED research for query 'client none query'" in caplog.text

@pytest.mark.asyncio
async def test_process_task_success_with_apify(content_research_agent_mock_apify: ContentResearchAgent):
    agent = content_research_agent_mock_apify
    agent.set_message_handler(AsyncMock())

    mock_apify_data = [{"title": "Mocked Processed Result", "url": "http://mock.proc/1"}]
    agent.get_research_from_apify = AsyncMock(return_value=mock_apify_data)

    original_topic = "AI in Education"
    topic_artifact = Artifact(
        artifact_id="topic_artifact", task_id="t_parent", creator_agent_id="test",
        data=original_topic, content_type="text/plain",
        description="Initial topic for research", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description=f"Research {original_topic} with Apify",
        initiator_agent_id="orchestrator",
        input_artifacts=[topic_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_artifact = task.output_artifacts[0]
    
    # The agent's process_task uses original_topic for the summary header
    expected_query_for_apify = f"Comprehensive overview and recent developments in {original_topic}" # This is for the Apify call
    
    assert f"Apify Research Summary for: {original_topic}" in output_artifact.data # Header uses original_topic
    assert "Mocked Processed Result" in output_artifact.data # Check for the mocked Apify result
    
    # Verify get_research_from_apify was called with the agent-modified topic
    agent.get_research_from_apify.assert_awaited_once_with(
        expected_query_for_apify, 
        max_results=5, 
        task_id_for_log=task.task_id
    )

@pytest.mark.asyncio
async def test_process_task_apify_fails_uses_fallback(content_research_agent_mock_apify: ContentResearchAgent, caplog):
    agent = content_research_agent_mock_apify
    agent.set_message_handler(AsyncMock())
    agent.get_research_from_apify = AsyncMock(return_value=[])
    caplog.set_level(logging.WARNING)

    original_topic = "Quantum Physics"
    topic_artifact = Artifact(
        artifact_id="topic_artifact_fail", task_id="t_parent_f", creator_agent_id="test_f",
        data=original_topic, content_type="text/plain",
        description="Initial topic for failure", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description=f"Research {original_topic} - expect Apify fail",
        initiator_agent_id="orchestrator",
        input_artifacts=[topic_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_data = task.output_artifacts[0].data
    
    expected_query_for_apify = f"Comprehensive overview and recent developments in {original_topic}"
    
    assert f"Apify Research Summary for: {original_topic}" in output_data # Header uses original_topic
    # The fallback within process_task after get_research_from_apify returns [] will generate simulated data
    # The _get_fallback_research method is called by get_research_from_apify when it returns []
    # process_task calls get_research_from_apify(expected_query_for_apify, ...)
    # If that returns [], then process_task itself constructs the "Simulated Research Result..." string.
    # Let's check the agent code for what it does when get_research_from_apify returns an empty list.
    # agents/content_research_agent.py L194: if apify_results: ... else: research_summary_content += "No relevant..."
    # This is then used. The test mock agent.get_research_from_apify = AsyncMock(return_value=[])
    # So, the "No relevant information found..." path is taken.
    assert "No relevant information found or Apify research failed" in output_data
    assert f"Further investigation or alternative research methods may be needed for **{original_topic}**" in output_data

    # The caplog should reflect that get_research_from_apify (the mock) was called, and returned empty,
    # leading to the process_task logic for empty results.
    # The test mocks agent.get_research_from_apify, so its internal logs won't appear unless the mock calls the original.
    # The log "No relevant information found..." is from process_task directly.
    assert f"No relevant information found or Apify research failed for query '{expected_query_for_apify}'." in caplog.text

@pytest.mark.asyncio
async def test_process_task_apify_client_none(content_research_agent_no_apify_token: ContentResearchAgent, caplog):
    agent = content_research_agent_no_apify_token
    agent.set_message_handler(AsyncMock())
    caplog.set_level(logging.WARNING)

    original_topic = "Gardening Tips"
    topic_artifact = Artifact(
        artifact_id="topic_artifact_none", task_id="t_parent_n", creator_agent_id="test_n",
        data=original_topic, content_type="text/plain",
        description="Initial topic, no client", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Research Gardening - expect no Apify client",
        initiator_agent_id="orchestrator",
        input_artifacts=[topic_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_data = task.output_artifacts[0].data
    
    expected_query_for_apify = f"Comprehensive overview and recent developments in {original_topic}"
    
    assert f"Apify Research Summary for: {original_topic}" in output_data # Header uses original_topic
    # Fallback from get_research_from_apify when client is None (this uses SIMULATED)
    assert f"Simulated Apify Result 1: {expected_query_for_apify}" in output_data
    assert f"Apify client not available. Returning SIMULATED research for query \\'{expected_query_for_apify}\\'" in caplog.text

@pytest.mark.asyncio
async def test_process_task_no_input_artifact(content_research_agent_mock_apify: ContentResearchAgent, caplog):
    agent = content_research_agent_mock_apify
    caplog.set_level(logging.ERROR)
    agent.set_message_handler(AsyncMock())
    task = agent.create_task(
        description="Research without topic",
        initiator_agent_id="orchestrator"
    )

    await agent.process_task(task)

    assert task.status == TaskStatus.FAILED
    assert f"Research task {task.task_id} for {agent.card.name} has no input artifact (topic)." in caplog.text
    agent.message_handler.assert_awaited_once() 