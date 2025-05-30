import pytest
from typing import Type # For type hinting agent classes
from core.agent_factory import AgentFactory
from agents import (
    BaseAgent,
    OrchestratorAgent,
    ContentResearchAgent,
    WritingAgent,
    SEOAgent,
    ImageAgent
)

# Fixture to ensure a clean AgentFactory registry for each test
@pytest.fixture(autouse=True)
def reset_agent_factory_registry():
    """Resets the agent factory's registry before each test that uses it."""
    original_registry = AgentFactory._registry.copy()
    yield
    AgentFactory._registry = original_registry

def test_create_known_agents():
    """Test creating all known agent types from the factory."""
    agent_types_to_test = {
        "orchestrator": OrchestratorAgent,
        "content_research": ContentResearchAgent,
        "writing": WritingAgent,
        "seo": SEOAgent,
        "image": ImageAgent,
    }
    for key, agent_class in agent_types_to_test.items():
        agent = AgentFactory.create_agent(key)
        assert isinstance(agent, agent_class)
        assert isinstance(agent, BaseAgent)
        # Check default ID (or that it was constructed without error for defaults)
        assert agent.agent_id is not None 

def test_create_agent_with_custom_id_and_kwargs():
    """Test creating an agent with a custom ID and other kwargs."""
    custom_id = "custom_writer_123"
    custom_name = "My Custom Writer"
    custom_description = "Writes custom stories."
    agent = AgentFactory.create_agent(
        "writing", 
        agent_id=custom_id,
        name=custom_name, 
        description=custom_description,
        version="1.1.0"
    )
    assert isinstance(agent, WritingAgent)
    assert agent.agent_id == custom_id
    assert agent.card.name == custom_name
    assert agent.card.description == custom_description
    assert agent.card.version == "1.1.0"

def test_create_unknown_agent_type():
    """Test that creating an unknown agent type raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        AgentFactory.create_agent("non_existent_agent_type")
    assert "not registered in factory" in str(excinfo.value)
    assert "non_existent_agent_type" in str(excinfo.value)

# Test dynamic registration
class MockDerivedAgent(BaseAgent):
    def __init__(self, agent_id: str = "mock_derived_001", name: str = "Mock Derived Agent", description: str = "A mock derived agent."):
        super().__init__(agent_id, name, description)

class NotABaseAgent:
    pass

def test_register_agent_type_success():
    """Test successfully registering a new agent type."""
    AgentFactory.register_agent_type("mock_derived", MockDerivedAgent)
    agent = AgentFactory.create_agent("mock_derived")
    assert isinstance(agent, MockDerivedAgent)

def test_register_agent_type_already_exists_warning(caplog):
    """Test that re-registering an agent type logs a warning."""
    AgentFactory.register_agent_type("writing", WritingAgent) # First registration (or ensure it's there)
    # Capture logs at warning level
    import logging
    caplog.set_level(logging.WARNING)
    AgentFactory.register_agent_type("writing", SEOAgent) # Re-register with a different class
    assert "Agent type 'writing' is already registered. Overwriting with SEOAgent." in caplog.text
    # Ensure it was actually overwritten
    agent = AgentFactory.create_agent("writing")
    assert isinstance(agent, SEOAgent)

def test_register_agent_type_not_subclass_of_baseagent():
    """Test that registering a class not derived from BaseAgent raises TypeError."""
    with pytest.raises(TypeError) as excinfo:
        AgentFactory.register_agent_type("invalid_agent", NotABaseAgent) # type: ignore
    assert "must be a subclass of BaseAgent" in str(excinfo.value)

def test_create_agent_passes_extra_kwargs():
    """Test that extra kwargs are passed to the agent constructor if it accepts them."""
    class SpecialAgent(BaseAgent):
        def __init__(self, agent_id="special_001", name="Default Special Name", description="Default Special Desc", special_param=None, **kwargs):
            # Ensure all required BaseAgent params are provided, using defaults if necessary
            # kwargs might override name/description if provided by factory
            final_kwargs = {'name': name, 'description': description}
            final_kwargs.update(kwargs) # Factory kwargs can override defaults here
            super().__init__(agent_id=agent_id, **final_kwargs) 
            self.special_param = special_param
    
    AgentFactory.register_agent_type("special", SpecialAgent)
    agent = AgentFactory.create_agent("special", special_param="unique_value", name="Actual Special Name", description="Actual Special Description")
    assert isinstance(agent, SpecialAgent)
    assert agent.agent_id == "special_001" # Default from SpecialAgent's init
    assert agent.card.name == "Actual Special Name" # Overridden by factory
    assert agent.card.description == "Actual Special Description" # Overridden by factory
    assert agent.special_param == "unique_value"

    # Test with agent_id override as well
    agent2 = AgentFactory.create_agent("special", agent_id="custom_special_002", special_param="another", name="Another", description="Desc another")
    assert agent2.agent_id == "custom_special_002"
    assert agent2.special_param == "another"
    assert agent2.card.name == "Another" 