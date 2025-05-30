import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone
import logging
from apify_client import ApifyClientAsync
from apify_client._errors import ApifyApiError
from pathlib import Path # For tmp_path

from agents.seo_agent import SEOAgent
from agents.base_agent import Task, Artifact, TaskStatus
from core.agent_factory import create_agent # For using the factory

@pytest.fixture
def seo_agent_instance_mock_apify(monkeypatch, tmp_path: Path): # Added tmp_path
    monkeypatch.setenv("APIFY_API_TOKEN", "fake_token_for_seo_tests")
    
    mock_apify_client_instance = AsyncMock(spec=ApifyClientAsync) # Main client is async

    # actor() and dataset() methods on ApifyClientAsync are synchronous, returning respective clients.
    # So, these attributes on the mock_apify_client_instance should be MagicMock (sync mocks).
    
    mock_actor_sub_client = AsyncMock() 
    mock_apify_client_instance.actor = MagicMock(return_value=mock_actor_sub_client)

    mock_dataset_sub_client = AsyncMock()
    mock_apify_client_instance.dataset = MagicMock(return_value=mock_dataset_sub_client)

    with patch("agents.seo_agent.ApifyClientAsync", return_value=mock_apify_client_instance) as mock_apify_constructor:
        agent = create_agent(
            "SEOAgent",
            use_tmp_path=True,
            tmp_path=tmp_path
        )
    assert agent is not None, "Failed to create SEOAgent via factory"
    mock_apify_constructor.assert_called_once()
    # agent.apify_client will be the one from the patch.
    return agent

@pytest.fixture
def seo_agent_no_apify_token(caplog, tmp_path: Path): # Changed from async, added tmp_path
    """Provides an SEOAgent instance where APIFY_API_TOKEN is not set."""
    caplog.set_level(logging.WARNING)
    with patch('agents.seo_agent.os.getenv') as mock_getenv:
        # Ensure getenv returns None ONLY for APIFY_API_TOKEN
        original_getenv = os.getenv
        def side_effect(key, default=None):
            if key == "APIFY_API_TOKEN":
                return None
            return original_getenv(key, default)
        mock_getenv.side_effect = side_effect
        
        instance = create_agent(
            "SEOAgent",
            use_tmp_path=True,
            tmp_path=tmp_path
        )
    assert instance is not None, "Failed to create SEOAgent via factory (no_apify_token)"
    # No yield, just return. If caplog needs reset, do it in test or calling fixture.
    # caplog.set_level(logging.NOTSET) # Removed, should be handled by test if needed
    return instance

@pytest.mark.asyncio
async def test_seo_agent_initialization_with_token(seo_agent_instance_mock_apify: SEOAgent):
    agent = seo_agent_instance_mock_apify
    assert agent.card.name == "SEO Agent"
    assert "optimize_seo" in [cap.skill_name for cap in agent.card.capabilities]
    assert agent.apify_client is not None
    assert isinstance(agent.apify_client, AsyncMock) # The main client is an AsyncMock

@pytest.mark.asyncio
async def test_seo_agent_initialization_no_token(seo_agent_no_apify_token: SEOAgent, caplog):
    agent = seo_agent_no_apify_token
    assert agent.apify_client is None
    # Check caplog for records created *during the setup phase of the fixture*
    setup_warnings = [r for r in caplog.get_records(when='setup') if r.levelname == 'WARNING']
    assert any("APIFY_API_TOKEN not found" in r.message for r in setup_warnings)
    assert any("SEO Agent keyword research will use fallback." in r.message for r in setup_warnings)

@pytest.mark.asyncio
async def test_get_keywords_from_apify_success(seo_agent_instance_mock_apify: SEOAgent):
    agent = seo_agent_instance_mock_apify
    task_id_for_log = "seo_task_success_123"

    mock_run_result = {
        "id": "run_success_1", "defaultDatasetId": "ds_success_1", "status": "SUCCEEDED"
    }
    # .actor is MagicMock, .call on its return_value (mock_actor_sub_client) is AsyncMock
    agent.apify_client.actor.return_value.call = AsyncMock(return_value=mock_run_result)

    mock_dataset_items = [{"keyword": "k1"}, {"search_term": "k2 "}, {"value": "k3"}]
    async def mock_iterate_items_func():
        for item in mock_dataset_items: yield item
    # .dataset is MagicMock, .iterate_items on its return_value (mock_dataset_sub_client) is MagicMock here
    agent.apify_client.dataset.return_value.iterate_items = MagicMock(return_value=mock_iterate_items_func())

    keywords = await agent.get_keywords_from_apify("topic1", task_id_for_log=task_id_for_log, max_keywords=3)
    assert keywords == ["k1", "k2", "k3"]
    
    agent.apify_client.actor.assert_called_once_with("zrikMXxBEbEj3a6Pc")
    agent.apify_client.actor.return_value.call.assert_awaited_once()
    agent.apify_client.dataset.assert_called_once_with("ds_success_1")
    agent.apify_client.dataset.return_value.iterate_items.assert_called_once()

@pytest.mark.asyncio
async def test_get_keywords_from_apify_api_error(seo_agent_instance_mock_apify: SEOAgent, caplog):
    agent = seo_agent_instance_mock_apify
    caplog.set_level(logging.ERROR)
    task_id_for_log = "seo_task_api_err"

    mock_error_response = MagicMock()
    mock_error_response.text = "Mocked SEO Apify API Error"
    agent.apify_client.actor.return_value.call = AsyncMock(side_effect=ApifyApiError(mock_error_response, attempt=1))

    keywords = await agent.get_keywords_from_apify("topic_api_error", task_id_for_log=task_id_for_log)
    assert keywords == ["topic_api_error", "topic_api_error insights", "learn topic_api_error"] # Fallback
    assert f"{agent.card.name}: Apify API error while fetching keywords for 'topic_api_error': Unexpected error: Mocked SEO Apify API Error. Task ID: {task_id_for_log}" in caplog.text
    assert "Mocked SEO Apify API Error" in caplog.text 
    agent.apify_client.actor.assert_called_once_with("zrikMXxBEbEj3a6Pc")
    agent.apify_client.actor.return_value.call.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_keywords_from_apify_no_dataset_id(seo_agent_instance_mock_apify: SEOAgent, caplog):
    agent = seo_agent_instance_mock_apify
    caplog.set_level(logging.ERROR)
    task_id_for_log = "seo_task_no_ds_id"
    mock_run_no_dataset = {"id": "run_no_ds", "defaultDatasetId": None, "status": "SUCCEEDED"}
    agent.apify_client.actor.return_value.call = AsyncMock(return_value=mock_run_no_dataset)

    keywords = await agent.get_keywords_from_apify("topic_no_ds", task_id_for_log=task_id_for_log)
    assert keywords == ["topic_no_ds", "topic_no_ds error fallback", "Apify issue topic_no_ds"]
    assert f"Apify actor run {mock_run_no_dataset['id']} for query 'topic_no_ds' did not return a valid defaultDatasetId." in caplog.text
    agent.apify_client.actor.assert_called_once_with("zrikMXxBEbEj3a6Pc")
    agent.apify_client.actor.return_value.call.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_keywords_from_apify_no_items(seo_agent_instance_mock_apify: SEOAgent, caplog):
    agent = seo_agent_instance_mock_apify
    caplog.set_level(logging.WARNING) # Fallback is a warning
    task_id_for_log = "seo_task_no_items"
    mock_run_empty_dataset = {"id": "run_empty_ds", "defaultDatasetId": "ds_empty_actual", "status": "SUCCEEDED"}
    agent.apify_client.actor.return_value.call = AsyncMock(return_value=mock_run_empty_dataset)

    async def mock_empty_iterate_items_func():
        if False: yield {} # Correct way for empty async generator
    agent.apify_client.dataset.return_value.iterate_items = MagicMock(return_value=mock_empty_iterate_items_func())

    keywords = await agent.get_keywords_from_apify("topic_no_items", task_id_for_log=task_id_for_log)
    assert keywords == ["topic_no_items"] # Fallback is just the topic
    assert "No keywords extracted from Apify" in caplog.text
    agent.apify_client.actor.assert_called_once_with("zrikMXxBEbEj3a6Pc")
    agent.apify_client.actor.return_value.call.assert_awaited_once()
    agent.apify_client.dataset.assert_called_once_with("ds_empty_actual")
    agent.apify_client.dataset.return_value.iterate_items.assert_called_once()

@pytest.mark.asyncio
async def test_get_keywords_from_apify_client_none(seo_agent_no_apify_token: SEOAgent, caplog):
    agent = seo_agent_no_apify_token
    caplog.set_level(logging.WARNING)
    keywords = await agent.get_keywords_from_apify("no client topic")
    assert keywords == ["no client topic", "no client topic insights", "learn no client topic"]
    assert "Apify client not available. Returning fallback keywords" in caplog.text

@pytest.mark.asyncio
async def test_process_task_success(seo_agent_instance_mock_apify: SEOAgent):
    agent = seo_agent_instance_mock_apify
    agent.get_keywords_from_apify = AsyncMock(return_value=["seo keyword1", "seo keyword2", "seo keyword3"])
    agent.set_message_handler(AsyncMock())

    draft_artifact = Artifact(
        artifact_id="draft_artifact_1", task_id="t1", creator_agent_id="w", 
        content_type="text/markdown", data="# Main Title\nSome content here.", 
        description="Blog post draft for topic: SEO Test Topic (generated)",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Optimize SEO for SEO Test Topic", 
        initiator_agent_id="orchestrator",
        input_artifacts=[draft_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_artifact = task.output_artifacts[0]
    assert output_artifact.content_type == "text/markdown"
    assert "<!-- SEO Analysis for: SEO Test Topic -->" in output_artifact.data
    assert "seo keyword1" in output_artifact.data
    assert "seo keyword2" in output_artifact.data
    assert "# Main Title" in output_artifact.data
    agent.get_keywords_from_apify.assert_awaited_once_with("SEO Test Topic", task_id_for_log=task.task_id, max_keywords=10)
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_no_input_artifact(seo_agent_instance_mock_apify: SEOAgent, caplog):
    agent = seo_agent_instance_mock_apify
    caplog.set_level(logging.ERROR) 
    agent.set_message_handler(AsyncMock())
    task = agent.create_task(
        initiator_agent_id="orchestrator", 
        description="Optimize without draft"
    )
    # If task_id is needed for assertion, get it from created task: task.task_id

    await agent.process_task(task)

    assert task.status == TaskStatus.FAILED
    assert f"SEO task {task.task_id} for {agent.card.name} has no input draft artifact" in caplog.text
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_apify_fails_uses_fallback_keywords(seo_agent_instance_mock_apify: SEOAgent, caplog):
    agent = seo_agent_instance_mock_apify
    caplog.set_level(logging.WARNING)
    agent.get_keywords_from_apify = AsyncMock(return_value=[])
    agent.set_message_handler(AsyncMock())

    draft_artifact = Artifact(
        artifact_id="draft_fallback", task_id="t_fallback", creator_agent_id="w", 
        data="Content.", description="Blog post draft for topic: Fallback Topic",
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Optimize Fallback Topic", 
        initiator_agent_id="orchestrator", 
        input_artifacts=[draft_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    output_data = task.output_artifacts[0].data
    assert "Not enough keywords from Apify for topic 'Fallback Topic'" in caplog.text
    assert "Fallback Topic trends" in output_data
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "description, expected_topic",
    [
        ("Blog post draft for topic: My Detailed Topic (generated by X)", "My Detailed Topic"),
        ("SEO optimized draft for topic: Another Cool Topic (Apify keywords: 3)", "Another Cool Topic"),
        ("Draft for: Simple Topic", "Simple Topic"),
        ("Unknown format content", "the analyzed content")
    ]
)
async def test_topic_extraction_in_process_task(seo_agent_instance_mock_apify: SEOAgent, description: str, expected_topic: str):
    agent = seo_agent_instance_mock_apify
    agent.get_keywords_from_apify = AsyncMock(return_value=[expected_topic, "trend1", "trend2"])
    agent.set_message_handler(AsyncMock())

    draft_artifact = Artifact(
        artifact_id="draft_topic_test", task_id="t_topic", creator_agent_id="w", 
        data="Content.", description=description,
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description=f"Optimize {expected_topic}", 
        initiator_agent_id="orchestrator", 
        input_artifacts=[draft_artifact]
    )
    await agent.process_task(task)

    agent.get_keywords_from_apify.assert_awaited_once_with(expected_topic, task_id_for_log=task.task_id, max_keywords=10)
    output_data = task.output_artifacts[0].data
    assert f"<!-- SEO Analysis for: {expected_topic} -->" in output_data