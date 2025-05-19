import pytest
from agents.keyword_agent import KeywordAgent

def test_keyword_agent_returns_keywords():
    """
    KeywordAgent should return a list of keywords related to a topic.
    """
    agent = KeywordAgent()
    keywords = agent.run(topic="LLM Agents")
    assert isinstance(keywords, list)
    assert all(isinstance(k, str) for k in keywords)
    assert len(keywords) > 0
