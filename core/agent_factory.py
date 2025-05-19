from agents.keyword_agent import KeywordAgent
from agents.keyword_agent import KeywordAgent
from agents.title_agent import TitleAgent
from agents.intro_agent import IntroAgent
from agents.body_agent import BodyAgent
from agents.conclusion_agent import ConclusionAgent
from agents.review_agent import ReviewAgent
from agents.sources_agent import SourcesAgent
from agents.image_agent import ImageAgent

class AgentFactory:
    _registry = {
        "keyword": KeywordAgent,
        "keyword": KeywordAgent,
        "title": TitleAgent,
        "intro": IntroAgent,
        "body": BodyAgent,
        "conclusion": ConclusionAgent,
        "review": ReviewAgent,
        "sources": SourcesAgent,
        "image": ImageAgent,
    }

    @classmethod
    def create_agent(cls, agent_type: str, *args, **kwargs):
        agent_cls = cls._registry.get(agent_type)
        if not agent_cls:
            raise ValueError(f"Agent '{agent_type}' not registered in factory.")
        return agent_cls(*args, **kwargs)
