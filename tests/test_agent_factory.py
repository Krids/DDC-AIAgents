import pytest
from core.agent_factory import AgentFactory
from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    def run(self, *args, **kwargs):
        return "dummy"

def test_agent_factory_register_and_create():
    """
    Test that AgentFactory can register and create a new agent.
    """
    AgentFactory._registry["dummy"] = DummyAgent
    agent = AgentFactory.create_agent("dummy")
    assert isinstance(agent, DummyAgent)
    assert agent.run() == "dummy"

def test_agent_factory_invalid_agent_type():
    """
    Test that AgentFactory raises ValueError for unknown agent type.
    """
    with pytest.raises(ValueError):
        AgentFactory.create_agent("nonexistent")
