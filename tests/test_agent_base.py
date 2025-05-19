import pytest
from agents.base_agent import BaseAgent

def test_base_agent_run_is_abstract():
    """
    Test that BaseAgent.run is abstract and raises TypeError if not implemented.
    """
    with pytest.raises(TypeError):
        class DummyAgent(BaseAgent):
            pass
        DummyAgent()
