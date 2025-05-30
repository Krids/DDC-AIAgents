"""
Demonstration of A2A (Agent-to-Agent) communication protocol.
Shows how agents can collaborate to accomplish goals.
"""

import asyncio
import os
from typing import Dict, Any
from core.a2a_protocol import A2AProtocol, MessagePriority
from agents.keyword_agent import KeywordAgent
from agents.a2a_base_agent import A2ABaseAgent
from core.logger import logger


class ResearchAgent(A2ABaseAgent):
    """
    Example research agent that can gather information and collaborate with other agents.
    """
    
    def __init__(self, agent_id: str, protocol: A2AProtocol):
        super().__init__(agent_id, protocol)
        
        # Register capabilities
        self.register_capability("research_topic", "Research a topic and provide summary")
        self.register_capability("suggest_keywords", "Suggest keywords based on research")
        
        # Subscribe to relevant topics
        self.subscribe("keywords_found")
        
    async def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research actions"""
        if action == "research_topic":
            topic = data.get("topic")
            research = await self.research_topic(topic)
            return {"topic": topic, "research": research}
            
        elif action == "suggest_keywords":
            topic = data.get("topic")
            keywords = await self.suggest_keywords_for_topic(topic)
            return {"keywords": keywords}
            
        else:
            raise ValueError(f"Unknown action: {action}")
            
    async def research_topic(self, topic: str) -> str:
        """Simulate researching a topic"""
        logger.info(f"[{self.agent_id}] Researching topic: {topic}")
        
        # In a real implementation, this would search databases, APIs, etc.
        # For demo, we'll return a simulated research summary
        await asyncio.sleep(1)  # Simulate work
        
        return f"Research summary for '{topic}': This is a comprehensive topic covering various aspects..."
        
    async def suggest_keywords_for_topic(self, topic: str) -> list:
        """Suggest keywords based on research"""
        logger.info(f"[{self.agent_id}] Suggesting keywords for: {topic}")
        
        # Simulate keyword suggestions based on research
        base_keywords = topic.lower().split()
        suggested = []
        
        for word in base_keywords:
            suggested.extend([
                f"{word} tutorial",
                f"{word} guide",
                f"best {word}",
                f"{word} examples"
            ])
            
        return suggested
        
    async def run_async(self, *args, **kwargs):
        """Async run method"""
        # This agent primarily responds to requests
        pass


class ContentAgent(A2ABaseAgent):
    """
    Example content generation agent that creates content based on keywords and research.
    """
    
    def __init__(self, agent_id: str, protocol: A2AProtocol):
        super().__init__(agent_id, protocol)
        
        # Register capabilities
        self.register_capability("generate_outline", "Generate content outline")
        self.register_capability("write_section", "Write a content section")
        
        # Subscribe to keyword updates
        self.subscribe("keywords_found")
        
        # Store keywords for content generation
        self.available_keywords = []
        
    async def execute_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content generation actions"""
        if action == "generate_outline":
            topic = data.get("topic")
            keywords = data.get("keywords", [])
            outline = await self.generate_outline(topic, keywords)
            return {"outline": outline}
            
        elif action == "write_section":
            section = data.get("section")
            content = await self.write_section(section)
            return {"content": content}
            
        else:
            raise ValueError(f"Unknown action: {action}")
            
    async def generate_outline(self, topic: str, keywords: list) -> Dict[str, Any]:
        """Generate content outline based on topic and keywords"""
        logger.info(f"[{self.agent_id}] Generating outline for: {topic}")
        
        outline = {
            "title": f"Comprehensive Guide to {topic}",
            "sections": [
                {"name": "Introduction", "keywords": keywords[:2] if keywords else []},
                {"name": "Main Concepts", "keywords": keywords[2:4] if len(keywords) > 2 else []},
                {"name": "Practical Applications", "keywords": keywords[4:6] if len(keywords) > 4 else []},
                {"name": "Conclusion", "keywords": keywords[6:] if len(keywords) > 6 else []}
            ]
        }
        
        return outline
        
    async def write_section(self, section: str) -> str:
        """Write content for a specific section"""
        logger.info(f"[{self.agent_id}] Writing section: {section}")
        
        # Simulate content writing
        await asyncio.sleep(0.5)
        
        return f"Content for {section}: This section covers important aspects..."
        
    async def run_async(self, *args, **kwargs):
        """Async run method"""
        pass
        
    async def _handle_request(self, message):
        """Override to handle broadcasts"""
        await super()._handle_request(message)
        
        # Handle keyword broadcasts
        if message.payload.get("action") == "keywords_found":
            self.available_keywords = message.payload.get("data", {}).get("keywords", [])
            logger.info(f"[{self.agent_id}] Received keywords: {self.available_keywords}")


async def demonstrate_a2a_collaboration():
    """
    Demonstrate how agents collaborate using A2A protocol.
    """
    # Initialize protocol
    protocol = A2AProtocol()
    
    # Create agents
    keyword_agent = KeywordAgent("keyword_agent", protocol)
    research_agent = ResearchAgent("research_agent", protocol)
    content_agent = ContentAgent("content_agent", protocol)
    
    # Start protocol processing
    protocol_task = asyncio.create_task(protocol.process_messages())
    
    try:
        # Scenario 1: Direct agent-to-agent communication
        logger.info("\n=== Scenario 1: Direct Communication ===")
        
        # Research agent requests keywords from keyword agent
        keywords_result = await research_agent.request_capability(
            "keyword_agent",
            "find_keywords",
            {"topic": "machine learning"},
            timeout=10.0
        )
        
        logger.info(f"Keywords received by research agent: {keywords_result}")
        
        # Scenario 2: Collaborative content generation
        logger.info("\n=== Scenario 2: Collaborative Content Generation ===")
        
        topic = "artificial intelligence"
        
        # Step 1: Get keywords
        logger.info("Step 1: Getting keywords...")
        keywords = await keyword_agent.find_keywords(topic)
        
        # Step 2: Research the topic
        logger.info("Step 2: Researching topic...")
        research_result = await content_agent.request_capability(
            "research_agent",
            "research_topic",
            {"topic": topic}
        )
        
        # Step 3: Generate content outline
        logger.info("Step 3: Generating content outline...")
        outline_result = await research_agent.request_capability(
            "content_agent",
            "generate_outline",
            {"topic": topic, "keywords": keywords}
        )
        
        logger.info(f"Generated outline: {outline_result}")
        
        # Scenario 3: Multi-agent collaboration
        logger.info("\n=== Scenario 3: Multi-Agent Collaboration ===")
        
        # Content agent requests help from multiple agents
        collaboration_results = await content_agent.collaborate(
            ["keyword_agent", "research_agent"],
            "suggest_keywords",
            {"topic": "deep learning"}
        )
        
        logger.info(f"Collaboration results: {collaboration_results}")
        
        # Scenario 4: Broadcasting and subscriptions
        logger.info("\n=== Scenario 4: Broadcasting ===")
        
        # Keyword agent broadcasts found keywords
        await keyword_agent.broadcast(
            "new_trending_keywords",
            {"keywords": ["AI ethics", "responsible AI", "AI governance"]},
            priority=MessagePriority.HIGH
        )
        
        # Give time for messages to process
        await asyncio.sleep(2)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        
    finally:
        # Clean up
        protocol.stop()
        await asyncio.sleep(0.5)  # Allow time for cleanup
        protocol_task.cancel()
        
        
async def demonstrate_workflow():
    """
    Demonstrate a complete workflow using A2A orchestrator.
    """
    from core.a2a_orchestrator import A2AOrchestrator
    
    logger.info("\n=== A2A Orchestrated Workflow ===")
    
    orchestrator = A2AOrchestrator()
    
    # Start protocol
    protocol_task = asyncio.create_task(orchestrator.run_protocol())
    
    try:
        # Execute keyword research workflow
        results = await orchestrator.execute_workflow(
            "keyword_research",
            {"topic": "blockchain technology"}
        )
        
        logger.info(f"Workflow completed with results: {results}")
        
    finally:
        orchestrator.protocol.stop()
        protocol_task.cancel()


async def main():
    """Main demonstration function"""
    logger.info("Starting A2A Protocol Demonstration")
    
    # Set dummy credentials for demo (if not set)
    if not os.getenv("DATAFORSEO_LOGIN"):
        os.environ["DATAFORSEO_LOGIN"] = "demo"
        os.environ["DATAFORSEO_PASSWORD"] = "demo"
    
    # Run demonstrations
    await demonstrate_a2a_collaboration()
    await demonstrate_workflow()
    
    logger.info("\nA2A Protocol Demonstration Complete!")


if __name__ == "__main__":
    asyncio.run(main())
