"""
Agent Factory to create and register various agent types.
"""

from typing import Optional, Dict, Type, Any

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

# Global agent registry
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {}

def register_agent_type(name: str, agent_class: Type[BaseAgent]):
    logger.info(f"Registering agent type: {name}")
    AGENT_REGISTRY[name] = agent_class

# Register existing agent types
register_agent_type("OrchestratorAgent", OrchestratorAgent)
register_agent_type("ContentResearchAgent", ContentResearchAgent)
register_agent_type("WritingAgent", WritingAgent)
register_agent_type("SEOAgent", SEOAgent)
register_agent_type("ImageAgent", ImageAgent)

def create_agent(agent_type_name: str, agent_id: Optional[str] = None, 
                   name: Optional[str] = None, description: Optional[str] = None,
                   use_tmp_path: bool = False, tmp_path: Optional[Any] = None, # Added tmp_path and use_tmp_path
                   **kwargs) -> Optional[BaseAgent]:
    logger.debug(f"Attempting to create agent of type: {agent_type_name} with ID: {agent_id}")
    agent_class = AGENT_REGISTRY.get(agent_type_name)
    if agent_class:
        params = {}
        if agent_id: params['agent_id'] = agent_id
        if name: params['name'] = name
        if description: params['description'] = description
        
        if use_tmp_path and tmp_path:
            params['data_dir_override'] = str(tmp_path) # Pass tmp_path as string for override
            
        params.update(kwargs) # Add any other specific kwargs
        try:
            instance = agent_class(**params)
            logger.info(f"Agent of type '{agent_type_name}' created successfully with ID '{agent_id if agent_id else 'default'}'")
            return instance
        except Exception as e:
            logger.error(f"Error instantiating agent '{agent_type_name}' with class {agent_class.__name__} and args {params}: {e}", exc_info=True)
            return None

# Example of how one might dynamically register later, if needed:
# class CustomAgent(BaseAgent):
#     pass
# AgentFactory.register_agent_type("custom", CustomAgent)
