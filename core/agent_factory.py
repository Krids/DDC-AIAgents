"""
Agent Factory to create and register various agent types.
"""

from typing import Optional

from agents import (
    OrchestratorAgent,
    ContentResearchAgent,
    WritingAgent,
    SEOAgent,
    ImageAgent,
    BaseAgent # For type hinting if needed, or specific agent types
)

# For logging
import logging
logger = logging.getLogger(f"agentsAI.{__name__}")

class AgentFactory:
    _registry = {
        "orchestrator": OrchestratorAgent,
        "content_research": ContentResearchAgent,
        "writing": WritingAgent,
        "seo": SEOAgent,
        "image": ImageAgent,
    }

    @classmethod
    def register_agent_type(cls, agent_type_key: str, agent_class: type[BaseAgent]):
        """Allows dynamic registration of agent types."""
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"Agent class {agent_class.__name__} must be a subclass of BaseAgent.")
        if agent_type_key in cls._registry:
            logger.warning(f"Agent type '{agent_type_key}' is already registered. Overwriting with {agent_class.__name__}.")
        cls._registry[agent_type_key] = agent_class
        logger.info(f"Agent type '{agent_type_key}' registered to class {agent_class.__name__}.")

    @classmethod
    def create_agent(cls, agent_type_key: str, agent_id: Optional[str] = None, **kwargs) -> BaseAgent:
        """
        Creates an agent instance of the specified type.

        Args:
            agent_type_key (str): The key for the agent type to create (e.g., "orchestrator").
            agent_id (Optional[str]): Specific agent ID. If None, the agent's default ID will be used.
            **kwargs: Additional keyword arguments to pass to the agent's constructor (e.g., name, description).

        Returns:
            BaseAgent: An instance of the requested agent.

        Raises:
            ValueError: If the agent_type_key is not registered.
        """
        agent_class = cls._registry.get(agent_type_key)
        if not agent_class:
            logger.error(f"Agent type '{agent_type_key}' not found in factory registry. Available types: {list(cls._registry.keys())}")
            raise ValueError(f"Agent type '{agent_type_key}' not registered in factory.")
        
        # Prepare constructor arguments
        # Our agents typically take agent_id, name, description. Kwargs can override defaults or add more.
        constructor_args = {}
        if agent_id:
            constructor_args['agent_id'] = agent_id
        
        # Merge with other kwargs, allowing user-provided kwargs to override defaults set by agent_id logic
        constructor_args.update(kwargs) 

        logger.info(f"Creating agent of type '{agent_type_key}' (Class: {agent_class.__name__}) with ID '{agent_id if agent_id else 'default'}' and args: {constructor_args}")
        try:
            return agent_class(**constructor_args)
        except Exception as e:
            logger.error(f"Error instantiating agent '{agent_type_key}' with class {agent_class.__name__} and args {constructor_args}: {e}", exc_info=True)
            raise # Re-raise the exception after logging

# Example of how one might dynamically register later, if needed:
# class CustomAgent(BaseAgent):
#     pass
# AgentFactory.register_agent_type("custom", CustomAgent)
