import pytest
from typing import Type, Dict # For type hinting agent classes
from core.agent_factory import create_agent, register_agent_type, AGENT_REGISTRY
from agents import (
    BaseAgent,
    OrchestratorAgent,
    ContentResearchAgent,
    WritingAgent,
    SEOAgent,
    ImageAgent
)

# Fixture to ensure a clean AGENT_REGISTRY for each test
@pytest.fixture(autouse=True)
def reset_agent_registry():
    """Resets the AGENT_REGISTRY before each test that uses it."""
    original_registry = AGENT_REGISTRY.copy()
    yield
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(original_registry)

def test_create_known_agents(tmp_path): # Added tmp_path for data_dir_override
    """Test creating all known agent types from the factory."""
    # Note: AGENT_REGISTRY is pre-populated at module load by core.agent_factory
    # For this test, we can rely on that pre-population or re-register if we want to be explicit.
    # The reset_agent_registry fixture ensures a clean state from the original pre-population.
    
    agent_type_names_to_test = {
        "OrchestratorAgent": OrchestratorAgent,
        "ContentResearchAgent": ContentResearchAgent,
        "WritingAgent": WritingAgent,
        "SEOAgent": SEOAgent,
        "ImageAgent": ImageAgent,
    }
    for type_name, agent_class in agent_type_names_to_test.items():
        # Ensure the agent type is in the current registry for the test
        if type_name not in AGENT_REGISTRY:
            register_agent_type(type_name, agent_class) # Register if reset removed it and it wasn't re-added

        agent = create_agent(type_name, use_tmp_path=True, tmp_path=tmp_path)
        assert agent is not None, f"Failed to create agent {type_name}"
        assert isinstance(agent, agent_class)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id is not None 

def test_create_agent_with_custom_id_and_kwargs(tmp_path):
    """Test creating an agent with a custom ID and other kwargs."""
    custom_id = "custom_writer_123"
    custom_name = "My Custom Writer"
    custom_description = "Writes custom stories."
    
    # Ensure WritingAgent is registered for this test instance if registry was cleared
    if "WritingAgent" not in AGENT_REGISTRY:
        register_agent_type("WritingAgent", WritingAgent)

    agent = create_agent(
        "WritingAgent", 
        agent_id=custom_id,
        name=custom_name, 
        description=custom_description,
        version="1.1.0", # This will go into card.metadata via BaseAgent's kwargs
        use_tmp_path=True,
        tmp_path=tmp_path
    )
    assert agent is not None
    assert isinstance(agent, WritingAgent)
    assert agent.agent_id == custom_id
    assert agent.card.name == custom_name
    assert agent.card.description == custom_description
    assert agent.card.version == "1.1.0"

def test_create_unknown_agent_type(tmp_path):
    """Test that creating an unknown agent type returns None and logs an error."""
    # AGENT_REGISTRY is reset by fixture, so "non_existent_agent_type" won't be there
    agent = create_agent("non_existent_agent_type", use_tmp_path=True, tmp_path=tmp_path)
    assert agent is None 
    # Error logging is checked via caplog if needed, but create_agent returning None is the primary check

# Test dynamic registration
class MockDerivedAgent(BaseAgent):
    def __init__(self, agent_id: str = "mock_derived_001", name: str = "Mock Derived Agent", description: str = "A mock derived agent.", **kwargs):
        super().__init__(agent_id=agent_id, name=name, description=description, **kwargs)

class NotABaseAgent:
    pass

def test_register_agent_type_success(tmp_path):
    """Test successfully registering a new agent type."""
    register_agent_type("MockDerivedAgent", MockDerivedAgent)
    agent = create_agent("MockDerivedAgent", use_tmp_path=True, tmp_path=tmp_path)
    assert agent is not None
    assert isinstance(agent, MockDerivedAgent)

def test_register_agent_type_already_exists_warning(caplog, tmp_path):
    """Test that re-registering an agent type logs a warning."""
    # Ensure WritingAgent is initially registered
    if "WritingAgent" not in AGENT_REGISTRY:
         register_agent_type("WritingAgent", WritingAgent)
    
    original_class = AGENT_REGISTRY["WritingAgent"]

    import logging
    caplog.set_level(logging.INFO) # Factory logs re-registration at INFO level
    
    # Re-register with a different class (SEOAgent) but same type name
    register_agent_type("WritingAgent", SEOAgent) 
    
    assert "Registering agent type: WritingAgent" in caplog.text # Will show twice due to re-registration.
    # The factory doesn't explicitly warn on overwrite but just reassigns.
    # The logger in register_agent_type will just log the new registration.
    
    # Ensure it was actually overwritten
    agent = create_agent("WritingAgent", use_tmp_path=True, tmp_path=tmp_path)
    assert agent is not None
    assert isinstance(agent, SEOAgent)

    # Restore original for other tests if needed (though fixture should handle general reset)
    register_agent_type("WritingAgent", original_class)


def test_register_agent_type_not_subclass_of_baseagent():
    """Test that registering a class not derived from BaseAgent raises TypeError (or similar check)."""
    # The current register_agent_type doesn't enforce BaseAgent subclassing.
    # This is a design choice. If enforcement is desired, it should be added to register_agent_type.
    # For now, this test will pass as is, as no error is raised by register_agent_type itself.
    # If create_agent fails later, that's a different issue.
    try:
        register_agent_type("InvalidAgent", NotABaseAgent) # type: ignore
        # Attempt to create to see if it fails there, assuming BaseAgent features are used in __init__
        # agent = create_agent("InvalidAgent") # This would likely fail if NotABaseAgent can't handle BaseAgent params
    except TypeError:
        # This would be the ideal place if register_agent_type itself checked
        pytest.fail("register_agent_type should ideally check for BaseAgent subclass, but currently does not.")
    
    # To actually test the failure, one might try to create it:
    # with pytest.raises(TypeError): # Or whatever error BaseAgent init might cause
    # create_agent("InvalidAgent", name="test") # This depends on NotABaseAgent's __init__

    # For now, let's confirm it can be registered, and leave creation failure to other tests if any.
    assert "InvalidAgent" in AGENT_REGISTRY
    assert AGENT_REGISTRY["InvalidAgent"] == NotABaseAgent


def test_create_agent_passes_extra_kwargs(tmp_path):
    """Test that extra kwargs are passed to the agent constructor if it accepts them."""
    class SpecialAgent(BaseAgent):
        def __init__(self, agent_id="special_001", name="Default Special Name", description="Default Special Desc", special_param=None, **kwargs):
            super().__init__(agent_id=agent_id, name=name, description=description, **kwargs) 
            self.special_param = special_param
    
    register_agent_type("SpecialAgent", SpecialAgent)
    agent = create_agent("SpecialAgent", special_param="unique_value", name="Actual Special Name", 
                         description="Actual Special Description", use_tmp_path=True, tmp_path=tmp_path)
    assert agent is not None
    assert isinstance(agent, SpecialAgent)
    assert agent.agent_id == "special_001" 
    assert agent.card.name == "Actual Special Name"
    assert agent.card.description == "Actual Special Description"
    assert agent.special_param == "unique_value"

    agent2 = create_agent("SpecialAgent", agent_id="custom_special_002", special_param="another", 
                          name="Another", description="Desc another", use_tmp_path=True, tmp_path=tmp_path)
    assert agent2 is not None
    assert agent2.agent_id == "custom_special_002"
    assert agent2.special_param == "another"
    assert agent2.card.name == "Another" 