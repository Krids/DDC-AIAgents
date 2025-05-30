import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import logging
from pathlib import Path # For tmp_path

from agents.image_agent import ImageAgent
from agents.base_agent import Task, Artifact, TaskStatus
# from protocols.a2a_schemas import AgentMessage # Not directly used, but context is fine
from core.agent_factory import create_agent # For using the factory

@pytest.fixture
async def image_agent_instance_mock_openai(tmp_path: Path): # Added tmp_path, fixture is async so agent creation will be too if it were async
    """Provides an ImageAgent instance with a mocked OpenAI client."""
    # This fixture needs to be non-async if create_agent is non-async and ImageAgent init is non-async
    # Let's make it non-async for consistency, as agent init is sync.
    with patch('agents.image_agent.AsyncOpenAI') as MockOpenAIClass:
        mock_openai_client_instance = AsyncMock()
        mock_openai_client_instance.images = AsyncMock()
        mock_openai_client_instance.images.generate = AsyncMock()
        MockOpenAIClass.return_value = mock_openai_client_instance
        
        instance = create_agent(
            "ImageAgent",
            use_tmp_path=True,
            tmp_path=tmp_path
        )
        assert instance is not None, "Failed to create ImageAgent via factory"
        # Ensure the mocked client is indeed set if the patch worked as expected during init
        # This check might be redundant if the factory correctly passes kwargs for AsyncOpenAI, 
        # but it's good for verifying the mock setup when AsyncOpenAI is instantiated inside the agent.
        # However, ImageAgent's __init__ directly calls AsyncOpenAI(), so the patch on the module is key.
        # instance.openai_client will be the mock_openai_client_instance due to the patch.
        yield instance # Yield because the original fixture used yield

@pytest.fixture
async def image_agent_no_openai_client(caplog, tmp_path: Path): # Added tmp_path
    """Provides an ImageAgent instance where OpenAI client initialization fails."""
    caplog.set_level(logging.ERROR)
    with patch('agents.image_agent.AsyncOpenAI', side_effect=Exception("OpenAI Init Error")) as mock_init_fail:
        instance = create_agent(
            "ImageAgent",
            use_tmp_path=True,
            tmp_path=tmp_path
        )
    assert instance is not None, "Failed to create ImageAgent via factory (no_openai_client)"
    yield instance # Yield because the original fixture used yield
    caplog.set_level(logging.NOTSET) # This was in the original, keep if specific reason exists

@pytest.mark.asyncio
async def test_image_agent_initialization_success(image_agent_instance_mock_openai: ImageAgent):
    agent = image_agent_instance_mock_openai
    assert agent.card.name == "Image Agent"
    assert "find_images_openai" in [cap.skill_name for cap in agent.card.capabilities]
    assert agent.openai_client is not None
    assert isinstance(agent.openai_client, AsyncMock)
    assert isinstance(agent.openai_client.images.generate, AsyncMock) # Verify it's an AsyncMock

@pytest.mark.asyncio
async def test_image_agent_initialization_failure(image_agent_no_openai_client: ImageAgent, caplog):
    agent = image_agent_no_openai_client
    assert agent.openai_client is None
    # Check caplog for records created *during the setup phase of the fixture* or agent init
    error_logs = [r for r in caplog.get_records(when='call') if r.levelname == 'ERROR' and "Error initializing OpenAI client" in r.message]
    # The fixture also logs, so check setup records if call phase is empty
    if not error_logs:
        error_logs = [r for r in caplog.get_records(when='setup') if r.levelname == 'ERROR' and "Error initializing OpenAI client" in r.message]

    assert len(error_logs) > 0, "OpenAI client initialization error was not logged"
    assert "OpenAI Init Error" in error_logs[-1].message

@pytest.mark.asyncio
async def test_process_task_success(image_agent_instance_mock_openai: ImageAgent):
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    
    mock_dalle_response = MagicMock()
    mock_dalle_response.data = [MagicMock(url="http://generated.images.ai/final_image.png")]
    mock_dalle_response.model_dump = MagicMock(return_value={"data": [{"url": "http://generated.images.ai/final_image.png"}], "created": int(datetime.now(timezone.utc).timestamp())})
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
    # caplog.set_level(logging.ERROR) # Already set by fixture
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
    # The agent logs an error during __init__ if client fails, and another in process_task
    assert "OpenAI client not initialized for ImageAgent" in caplog.text # Logged in process_task
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
    caplog.set_level(logging.WARNING) # Agent logs warning for this case
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    
    # Simulate DALL-E returning no data or an error scenario leading to no URL
    mock_dalle_response_empty = MagicMock()
    mock_dalle_response_empty.data = [] # No image URL
    mock_dalle_response_empty.model_dump = MagicMock(return_value={"data": [], "created": int(datetime.now(timezone.utc).timestamp())})
    agent.openai_client.images.generate = AsyncMock(return_value=mock_dalle_response_empty)

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

    assert task.status == TaskStatus.COMPLETED # Task completes with placeholder
    output_data = task.output_artifacts[0].data
    # Check for the specific warning log from _generate_image_with_dalle when it fails
    # and also the one from process_task if it's distinct
    assert "Failed to generate image with DALL-E for 'DALL-E Failure Test'. Using placeholder text." in caplog.text # Logged in process_task
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
        ("Some text artifact about Ancient Civilizations, which is cool.", "Ancient Civilizations"), # Test stripping after "about"
        ("Research summary for topic: Topic with (parentheses)", "Topic with"), # Corrected expected: parentheses are stripped by agent logic
        ("topic: Direct Topic from User (ignore this part)", "Direct Topic from User"),
        ("A generic document", "the blog post content") # Fallback
    ]
)
async def test_topic_extraction_for_dalle_prompt(image_agent_instance_mock_openai: ImageAgent, artifact_description: str, expected_topic_for_prompt: str):
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    
    mock_dalle_response = MagicMock()
    mock_dalle_response.data = [MagicMock(url="http://example.com/img.png")]
    mock_dalle_response.model_dump = MagicMock(return_value={"data": [{"url": "http://example.com/img.png"}], "created": int(datetime.now(timezone.utc).timestamp())})
    agent.openai_client.images.generate = AsyncMock(return_value=mock_dalle_response)

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
    
    # The prompt construction is inside _generate_image_with_dalle, called by process_task
    # So we check the arguments passed to the mocked generate method
    agent.openai_client.images.generate.assert_awaited_once()
    _, kwargs = agent.openai_client.images.generate.call_args
    actual_prompt = kwargs.get('prompt', '')
    
    # The prompt is more complex now, check for the key part
    # Prompt: f"A compelling and professional main illustration for a blog post about '{topic}'. ..."
    assert f"'{expected_topic_for_prompt}'" in actual_prompt # Check if the topic is in the prompt
    assert task.status == TaskStatus.COMPLETED

@pytest.mark.asyncio
async def test_process_task_calls_dalle_success(image_agent_instance_mock_openai: ImageAgent):
    """Test that process_task successfully calls DALL-E and creates an artifact."""
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    
    mock_dalle_response = MagicMock(data=[MagicMock(url="http://mocked.dalle.url/image.png")])
    mock_dalle_response.model_dump = MagicMock(return_value={"data": [{"url": "http://mocked.dalle.url/image.png"}], "created": int(datetime.now(timezone.utc).timestamp())})
    agent.openai_client.images.generate = AsyncMock(return_value=mock_dalle_response)

    content_artifact = Artifact(
        artifact_id="content_1", task_id="t_dalle_succ", creator_agent_id="test",
        data="Some blog content about space.", description="Blog post draft for topic: Exploring Mars",
        content_type="text/markdown", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Generate image for Mars post",
        initiator_agent_id="orchestrator",
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_artifact = task.output_artifacts[0]
    assert "http://mocked.dalle.url/image.png" in output_artifact.data
    assert "![Generated illustration for Exploring Mars]" in output_artifact.data
    agent.openai_client.images.generate.assert_awaited_once()
    call_args = agent.openai_client.images.generate.call_args
    assert "Exploring Mars" in call_args.kwargs['prompt']

@pytest.mark.asyncio
async def test_process_task_dalle_api_error(image_agent_instance_mock_openai: ImageAgent, caplog):
    caplog.set_level(logging.ERROR) # Agent logs error for DALL-E API issues
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())
    
    # Simulate DALL-E API error
    agent.openai_client.images.generate = AsyncMock(side_effect=Exception("DALL-E API Unit Test Error"))

    content_artifact = Artifact(
        artifact_id="content_err", task_id="t_dalle_err", creator_agent_id="test_err",
        data="Content for error test.", description="Blog post draft for topic: DALL-E Error Scenario",
        content_type="text/markdown", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Generate image, expect DALL-E error",
        initiator_agent_id="orchestrator",
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED # Completes with placeholder
    output_data = task.output_artifacts[0].data
    # The log from _generate_image_with_dalle for an API error is more generic now
    assert "Unexpected error during DALL-E image generation for 'DALL-E Error Scenario'" in caplog.text
    assert "DALL-E API Unit Test Error" in caplog.text # This is the specific exception message
    assert "*[Image generation failed for 'DALL-E Error Scenario'. Placeholder for a relevant image.]*" in output_data
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_dalle_no_client(image_agent_no_openai_client: ImageAgent, caplog):
    # caplog.set_level(logging.ERROR) # Fixture sets this
    agent = image_agent_no_openai_client
    agent.set_message_handler(AsyncMock())

    content_artifact = Artifact(
        artifact_id="content_no_client", task_id="t_no_client", creator_agent_id="test_nc",
        data="Content for no client test.", description="Blog post draft for topic: No OpenAI Client Available",
        content_type="text/markdown", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Generate image, no OpenAI client",
        initiator_agent_id="orchestrator",
        input_artifacts=[content_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.FAILED # Fails early if client is None at process_task start
    assert "OpenAI client not initialized for ImageAgent" in caplog.text # Logged in process_task
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_main_success_scenario(image_agent_instance_mock_openai: ImageAgent):
    """A more integrated test for the main success path of process_task."""
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())

    mock_dalle_url = "http://live.images.ai/a-beautiful-image.png"
    mock_dalle_response = MagicMock(data=[MagicMock(url=mock_dalle_url)])
    mock_dalle_response.model_dump = MagicMock(return_value={"data": [{"url": mock_dalle_url}], "created": int(datetime.now(timezone.utc).timestamp())})
    agent.openai_client.images.generate = AsyncMock(return_value=mock_dalle_response)
    
    original_markdown = "# My Great Blog Post\n\nThis is the introduction.\n\n## Section 1\nDetails here."
    input_artifact = Artifact(
        artifact_id="orig_md_1",
        task_id="parent_task_1",
        creator_agent_id="writing_agent",
        content_type="text/markdown",
        data=original_markdown,
        description="Blog post draft for topic: AI in Modern Art",
        created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Add DALL-E image to 'AI in Modern Art' post",
        initiator_agent_id="orchestrator",
        input_artifacts=[input_artifact]
    )

    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    assert len(task.output_artifacts) == 1
    output_artifact = task.output_artifacts[0]

    # Check that original content is preserved
    assert original_markdown in output_artifact.data
    # Check that image markdown is added
    assert f"![Generated illustration for AI in Modern Art]({mock_dalle_url} \"AI in Modern Art - AI Generated Image\")" in output_artifact.data
    assert "*[Image created by Image Agent using OpenAI DALL-E for 'AI in Modern Art']*" in output_artifact.data
    
    agent.openai_client.images.generate.assert_awaited_once()
    args, kwargs = agent.openai_client.images.generate.call_args
    assert "AI in Modern Art" in kwargs['prompt'] # Check the topic in prompt
    agent.message_handler.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_task_dalle_call_fails_uses_placeholder_text(image_agent_instance_mock_openai: ImageAgent, caplog):
    """Ensures placeholder text is used if _generate_image_with_dalle returns None."""
    caplog.set_level(logging.WARNING)
    agent = image_agent_instance_mock_openai
    agent.set_message_handler(AsyncMock())

    # Mock _generate_image_with_dalle directly to simulate failure
    agent._generate_image_with_dalle = AsyncMock(return_value=None)

    input_artifact = Artifact(
        artifact_id="input_for_fail_1", task_id="task_fail_1", creator_agent_id="writer",
        data="Some content here.", description="Blog post draft for topic: Abstract Concepts",
        content_type="text/markdown", created_at=datetime.now(timezone.utc).isoformat()
    )
    task = agent.create_task(
        description="Generate image for Abstract Concepts, expect DALL-E call failure",
        initiator_agent_id="orchestrator", input_artifacts=[input_artifact]
    )
    await agent.process_task(task)

    assert task.status == TaskStatus.COMPLETED
    output_artifact = task.output_artifacts[0]
    assert "*[Image generation failed for 'Abstract Concepts'. Placeholder for a relevant image.]*" in output_artifact.data
    
    # Ensure the specific warning from process_task is logged
    assert "Failed to generate image with DALL-E for 'Abstract Concepts'. Using placeholder text." in caplog.text
    
    agent._generate_image_with_dalle.assert_awaited_once_with("Abstract Concepts", task_id_for_log=task.task_id)
    agent.message_handler.assert_awaited_once()