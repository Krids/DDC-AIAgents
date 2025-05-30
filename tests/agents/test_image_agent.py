import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import logging

from agents.image_agent import ImageAgent
from agents.base_agent import Task, Artifact, TaskStatus
from protocols.a2a_schemas import AgentMessage # Though not directly used, good for context

@pytest.fixture
async def image_agent_instance_mock_openai():
    """Provides an ImageAgent instance with a mocked OpenAI client."""
    with patch('agents.image_agent.AsyncOpenAI') as MockOpenAIClass:
        mock_openai_client_instance = AsyncMock()
        MockOpenAIClass.return_value = mock_openai_client_instance
        instance = ImageAgent()
        yield instance

@pytest.fixture
async def image_agent_no_openai_client(caplog):
    """Provides an ImageAgent instance where OpenAI client initialization fails."""
    caplog.set_level(logging.ERROR)
    with patch('agents.image_agent.AsyncOpenAI', side_effect=Exception("OpenAI Init Error")):
        instance = ImageAgent()
        yield instance
    caplog.set_level(logging.NOTSET)

@pytest.mark.asyncio
async def test_image_agent_initialization_success(image_agent_instance_mock_openai: ImageAgent):
    agent = image_agent_instance_mock_openai
    assert agent.card.name == "Image Agent"
    assert "find_images_openai" in [cap.skill_name for cap in agent.card.capabilities]
    assert agent.openai_client is not None
    assert isinstance(agent.openai_client, AsyncMock)

@pytest.mark.asyncio
async def test_image_agent_initialization_failure(image_agent_no_openai_client: ImageAgent, caplog):
    agent = image_agent_no_openai_client
    assert agent.openai_client is None
    error_logs_during_setup = [r for r in caplog.get_records(when='setup') if r.levelname == 'ERROR']
    assert len(error_logs_during_setup) > 0
    assert "Error initializing OpenAI client for Image Agent: OpenAI Init Error" in error_logs_during_setup[-1].message

@pytest.mark.asyncio
async def test_generate_image_with_dalle_success(image_agent_instance_mock_openai: ImageAgent):
    agent = image_agent_instance_mock_openai
    mock_dalle_response = MagicMock()
    mock_dalle_response.data = [MagicMock(url="http://example.com/image1.png")]
    agent.openai_client.images.generate = AsyncMock(return_value=mock_dalle_response)

    urls = await agent.generate_image_with_dalle("A cat playing chess")
    assert urls == ["http://example.com/image1.png"]
    agent.openai_client.images.generate.assert_awaited_once_with(
        model="dall-e-3",
        prompt="A cat playing chess",
        size="1024x1024",
        quality="standard",
        n=1,
        response_format="url"
    )

@pytest.mark.asyncio
async def test_generate_image_with_dalle_api_error(image_agent_instance_mock_openai: ImageAgent, caplog):
    caplog.set_level(logging.ERROR)
    agent = image_agent_instance_mock_openai
    agent.openai_client.images.generate = AsyncMock(side_effect=Exception("DALL-E API Error"))

    urls = await agent.generate_image_with_dalle("A complex prompt")
    assert urls == []
    assert "Error calling DALL-E API" in caplog.text
    assert "DALL-E API Error" in caplog.text

@pytest.mark.asyncio
async def test_generate_image_with_dalle_no_client(image_agent_no_openai_client: ImageAgent, caplog):
    caplog.set_level(logging.WARNING)
    agent = image_agent_no_openai_client
    
    urls = await agent.generate_image_with_dalle("A prompt with no client")
    assert urls == []
    assert "OpenAI client not available. Cannot generate images" in caplog.text

@pytest.mark.asyncio
async def test_process_task_success(image_agent_instance_mock_openai: ImageAgent):
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    
    mock_dalle_response = MagicMock()
    mock_dalle_response.data = [MagicMock(url="http://generated.images.ai/final_image.png")]
    agent.openai_client.images.generate = AsyncMock(return_value=mock_dalle_response)

    content_artifact = Artifact(
        artifact_id="content_1", task_id="t1", creator_agent_id="seo", 
        content_type="text/markdown", data="# Blog Title\nIntro paragraph.", 
        description="SEO optimized draft for topic: AI Ethics",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Generate images for AI Ethics blog post", 
        initiator_agent_id="orchestrator",
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_artifact = task.output_artifacts[0]
    assert output_artifact.content_type == "text/markdown"
    assert "http://generated.images.ai/final_image.png" in output_artifact.data
    assert "![Generated illustration for AI Ethics]" in output_artifact.data
    assert "# Blog Title" in output_artifact.data
    agent.openai_client.images.generate.assert_awaited_once()
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_no_openai_client(image_agent_no_openai_client: ImageAgent, caplog):
    caplog.set_level(logging.ERROR)
    agent = image_agent_no_openai_client
    agent.set_message_handler(AsyncMock())

    content_artifact = Artifact(
        artifact_id="c1", task_id="t1", creator_agent_id="s", 
        data="Content", description="Content for topic: No Client Test",
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Image task", 
        initiator_agent_id="o", 
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.FAILED
    assert "OpenAI client not initialized" in caplog.text
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_no_input_artifact(image_agent_instance_mock_openai: ImageAgent, caplog):
    caplog.set_level(logging.ERROR)
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    task = agent.create_task(
        initiator_agent_id="o", 
        description="Image task no input"
    )

    await agent.process_task(task)
    
    assert task.status == TaskStatus.FAILED
    assert f"Image task {task.task_id} for {agent.card.name} has no input content artifact" in caplog.text
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_dalle_fails_uses_placeholder(image_agent_instance_mock_openai: ImageAgent, caplog):
    caplog.set_level(logging.WARNING)
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    agent.openai_client.images.generate = AsyncMock(return_value=MagicMock(data=[]))

    content_artifact = Artifact(
        artifact_id="c_fail", task_id="t_fail", creator_agent_id="s", 
        data="Content.", description="Draft for topic: DALL-E Failure Test",
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Image task DALL-E fail", 
        initiator_agent_id="o", 
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    output_data = task.output_artifacts[0].data
    assert "Failed to generate images with DALL-E for 'DALL-E Failure Test'" in caplog.text
    assert "*[Image generation failed for 'DALL-E Failure Test'. Placeholder for a relevant image.]*" in output_data
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifact_description, expected_topic_for_prompt",
    [
        ("Content with DALL-E image for topic: Quantum Computing Today", "Quantum Computing Today"),
        ("SEO optimized draft for topic: The Future of AI in Healthcare (details)", "The Future of AI in Healthcare"),
        ("Blog post draft for topic: Advanced Python Techniques", "Advanced Python Techniques"),
        ("Final output for topic: Renewable Energy Sources", "Renewable Energy Sources"),
        ("Some text artifact about Ancient Civilizations", "Ancient Civilizations"),
        ("A generic document", "the blog post content")
    ]
)
async def test_topic_extraction_for_dalle_prompt(image_agent_instance_mock_openai: ImageAgent, artifact_description: str, expected_topic_for_prompt: str):
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    agent.openai_client.images.generate = AsyncMock(return_value=MagicMock(data=[MagicMock(url="http://example.com/img.png")]))

    content_artifact = Artifact(
        artifact_id="c_topic", task_id="t_topic", creator_agent_id="s", 
        data="Content.", description=artifact_description,
        content_type="text/plain",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Image task topic test", 
        initiator_agent_id="o", 
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)
    
    expected_prompt_segment = f"A compelling and professional main illustration for a blog post about '{expected_topic_for_prompt}'."
    _, kwargs = agent.openai_client.images.generate.call_args
    actual_prompt = kwargs.get('prompt', '')
    
    assert expected_prompt_segment in actual_prompt
    assert task.status == TaskStatus.COMPLETED