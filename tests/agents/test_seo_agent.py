import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone
import logging

from agents.seo_agent import SEOAgent
from agents.base_agent import Task, Artifact, TaskStatus
from protocols.a2a_schemas import AgentMessage

@pytest.fixture
async def seo_agent_instance_mock_apify(caplog):
    """Provides an SEOAgent instance with a mocked Apify client."""
    with caplog.at_level(logging.INFO): # For SEOAgent init logs
        with patch('agents.seo_agent.os.getenv', return_value="fake_token_for_testing") as mock_getenv:
            with patch('agents.seo_agent.ApifyClientAsync') as MockApifyClass:
                mock_apify_client_instance = AsyncMock()
                MockApifyClass.return_value = mock_apify_client_instance
                instance = SEOAgent()
                # instance.apify_client is now set by __init__ to mock_apify_client_instance
                yield instance

@pytest.fixture
async def seo_agent_no_apify_token(caplog):
    """Provides an SEOAgent instance where APIFY_API_TOKEN is not set."""
    caplog.set_level(logging.WARNING) # Set level directly for init log
    with patch('agents.seo_agent.os.getenv', return_value=None):
        with patch('agents.seo_agent.ApifyClientAsync') as MockApifyClass:
            instance = SEOAgent()
            yield instance
    caplog.set_level(logging.NOTSET) # Reset level

@pytest.mark.asyncio
async def test_seo_agent_initialization_with_token(seo_agent_instance_mock_apify: SEOAgent):
    agent = seo_agent_instance_mock_apify
    assert agent.card.name == "SEO Agent"
    assert "optimize_seo" in [cap.skill_name for cap in agent.card.capabilities]
    assert agent.apify_client is not None
    assert isinstance(agent.apify_client, AsyncMock)

@pytest.mark.asyncio
async def test_seo_agent_initialization_no_token(seo_agent_no_apify_token: SEOAgent, caplog):
    agent = seo_agent_no_apify_token
    assert agent.apify_client is None
    # The relevant log is captured by the fixture's caplog instance
    warning_logs_during_setup = [r for r in caplog.get_records(when='setup') if r.levelname == 'WARNING']
    assert len(warning_logs_during_setup) > 0
    assert "APIFY_API_TOKEN not found in environment. SEO Agent keyword research will use fallback." in warning_logs_during_setup[-1].message

@pytest.mark.asyncio
async def test_get_keywords_from_apify_success(seo_agent_instance_mock_apify: SEOAgent):
    agent = seo_agent_instance_mock_apify
    
    mock_run_result = {
        "id": "run_id_123",
        "datasetId": "dataset_id_456"
    }

    # Mock the chain: apify_client.actor(ACTOR_ID).call()
    mock_actor_call_method = AsyncMock(return_value=mock_run_result)
    mock_actor_client_instance = AsyncMock() # This represents the Apify V2 ActorClient
    mock_actor_client_instance.call = mock_actor_call_method
    # agent.apify_client is the top-level AsyncMock from the fixture.
    # Its 'actor' attribute (which is a method) should return mock_actor_client_instance when called.
    agent.apify_client.actor.return_value = mock_actor_client_instance
    
    # Mock the chain: apify_client.dataset(DATASET_ID).iterate_items()
    mock_dataset_items = [
        {"keyword": "keyword1"}, 
        {"search_term": "keyword2 "}, # Note the space, will be stripped
        {"value": "keyword3"},
        {"text": "keyword1"}, # Duplicate, should be handled by set
        {"query": "keyword4"}
    ]
    async def mock_iterate_items_func(): # Needs to be an async generator
        for item in mock_dataset_items:
            yield item
    
    mock_dataset_client_instance = AsyncMock() # This represents the Apify V2 DatasetClient
    # Assign a MagicMock to iterate_items, its return_value is the async generator
    mock_dataset_client_instance.iterate_items = MagicMock(return_value=mock_iterate_items_func())
    # Its 'dataset' attribute (method) should return mock_dataset_client_instance.
    agent.apify_client.dataset.return_value = mock_dataset_client_instance

    keywords = await agent.get_keywords_from_apify("test topic", max_keywords=3)
    assert keywords == ["keyword1", "keyword2", "keyword3"]
    
    agent.apify_client.actor.assert_called_once_with("kocourek/keyword-research-tool")
    mock_actor_client_instance.call.assert_awaited_once() # Check the call on the actor client instance
    # The call to actor() itself doesn't take datasetId, dataset() does.
    agent.apify_client.dataset.assert_called_once_with("dataset_id_456")
    # Check that iterate_items was actually entered (though its call count is tricky with async iter)
    mock_dataset_client_instance.iterate_items.assert_called_once() # Check the mock method was called

@pytest.mark.asyncio
async def test_get_keywords_from_apify_api_error(seo_agent_instance_mock_apify: SEOAgent, caplog):
    agent = seo_agent_instance_mock_apify
    caplog.set_level(logging.ERROR)

    # Mock apify_client.actor(ACTOR_ID).call() to raise an error
    mock_actor_call_method = AsyncMock(side_effect=Exception("Apify API Error"))
    mock_actor_client_instance = AsyncMock()
    mock_actor_client_instance.call = mock_actor_call_method
    agent.apify_client.actor.return_value = mock_actor_client_instance

    keywords = await agent.get_keywords_from_apify("error topic")
    assert keywords == ["error topic", "error topic insights", "learn error topic"]
    assert "Error calling Apify actor" in caplog.text
    assert "Apify API Error" in caplog.text
    
    agent.apify_client.actor.assert_called_once_with("kocourek/keyword-research-tool")
    mock_actor_client_instance.call.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_keywords_from_apify_no_client(seo_agent_no_apify_token: SEOAgent, caplog):
    agent = seo_agent_no_apify_token
    caplog.set_level(logging.WARNING) # For the warning log in get_keywords_from_apify
    keywords = await agent.get_keywords_from_apify("no client topic")
    assert keywords == ["no client topic", "no client topic trends", "best no client topic practices"]
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
    agent.get_keywords_from_apify.assert_awaited_once_with("SEO Test Topic", max_keywords=10)
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

    agent.get_keywords_from_apify.assert_awaited_once_with(expected_topic, max_keywords=10)
    output_data = task.output_artifacts[0].data
    assert f"<!-- SEO Analysis for: {expected_topic} -->" in output_data