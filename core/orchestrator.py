from core.agent_factory import AgentFactory

class Orchestrator:
    def __init__(self):
        self.title_agent = AgentFactory.create_agent("title")
        self.intro_agent = AgentFactory.create_agent("intro")
        self.body_agent = AgentFactory.create_agent("body")
        self.conclusion_agent = AgentFactory.create_agent("conclusion")
        self.review_agent = AgentFactory.create_agent("review")
        self.sources_agent = AgentFactory.create_agent("sources")
        self.image_agent = AgentFactory.create_agent("image")

    def run(self):
        pass
