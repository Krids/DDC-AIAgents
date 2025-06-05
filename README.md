# AgentsAI

A sophisticated multi-agent AI system for automated blog post creation using intelligent agent collaboration. This project demonstrates a complete workflow from research to final content creation through coordinated AI agents.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinated system of specialized AI agents
- **Automated Blog Post Generation**: Complete end-to-end content creation pipeline
- **Research Integration**: Automated web research using Apify actors
- **SEO Optimization**: Intelligent keyword research and content optimization
- **Image Generation**: DALL-E powered image creation for blog posts
- **Orchestrated Workflows**: Smart task delegation and management
- **Extensible Design**: Easy to add new agents and capabilities

## 🏗️ Architecture

The system consists of specialized agents working together:

### Core Agents

1. **OrchestratorAgent** - Manages and coordinates all other agents
2. **ContentResearchAgent** - Performs topic research using Apify actors
3. **WritingAgent** - Generates blog content using OpenAI GPT models
4. **SEOAgent** - Optimizes content for search engines with keyword research
5. **ImageAgent** - Creates relevant images using OpenAI DALL-E

### Agent Communication

- **A2A Protocol**: Agent-to-Agent communication system
- **Task Management**: Structured task assignment and status tracking
- **Artifact System**: Standardized data exchange between agents
- **Message Routing**: Intelligent message delivery and handling

## 📋 Prerequisites

- Python 3.13+
- OpenAI API key
- Apify API token (for research and SEO agents)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentsAI
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   APIFY_API_TOKEN=your_apify_token_here
   ```

## 🚀 Quick Start

### Basic Usage

Run the main blog post generation workflow:

```bash
python main.py
```

This will:
1. Research the topic "The Future of Multi-Agent AI Systems"
2. Generate a comprehensive blog post
3. Optimize it for SEO
4. Add relevant images
5. Save the final result to `outputs/`

### Custom Topics

To generate content for a different topic, modify the `blog_topic` variable in `main.py`:

```python
blog_topic = "Your Custom Topic Here"
```

## 📁 Project Structure

```
agentsAI/
├── agents/                    # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py         # Base agent class
│   ├── orchestrator.py       # Main orchestrator agent
│   ├── content_research_agent.py  # Research agent
│   ├── writing_agent.py      # Content writing agent
│   ├── seo_agent.py         # SEO optimization agent
│   └── image_agent.py       # Image generation agent
├── core/                     # Core system components
│   ├── agent_factory.py     # Agent creation and management
│   ├── agent_prompt_builder.py  # LLM prompt generation
│   └── logger.py           # Logging configuration
├── protocols/               # Communication protocols
│   ├── __init__.py
│   └── a2a_schemas.py      # Agent-to-Agent schemas
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── json_utils.py       # JSON handling utilities
├── tests/                   # Comprehensive test suite
├── outputs/                 # Generated blog posts
├── data/                   # Agent operation logs and data
├── logs/                   # System logs
├── main.py                 # Main entry point
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🔧 Configuration

### Agent Configuration

Each agent can be configured through the `core/agent_factory.py`:

```python
# Create custom agent instances
research_agent = create_agent(
    "ContentResearchAgent",
    agent_id="custom_research_001",
    name="Custom Research Agent"
)
```

### Logging

Logging is configured in `core/logger.py`. Logs are written to:
- Console (INFO level)
- `logs/agentsAI.log` (INFO level)

### Data Storage

Agent operations are automatically logged to the `data/` directory:
- Research results: `data/content_research/`
- Writing responses: `data/writing_openai/`
- SEO analysis: `data/seo_apify/`
- Image generation: `data/image_openai_dalle/`

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/agents/test_base_agent.py
pytest tests/core/test_agent_factory.py

# Run with coverage
pytest --cov=agents --cov=core
```

### Test Structure

- `tests/agents/` - Agent-specific tests
- `tests/core/` - Core system tests
- `tests/conftest.py` - Shared test fixtures

## 📊 Workflow Example

Here's how the system generates a complete blog post:

1. **Research Phase**
   ```python
   # ContentResearchAgent searches for recent information
   research_results = await research_agent.get_research_from_apify(
       "Comprehensive overview of AI trends", 
       max_results=3
   )
   ```

2. **Writing Phase**
   ```python
   # WritingAgent creates content using OpenAI
   blog_draft = await writing_agent._generate_draft_with_openai(
       topic="AI Trends",
       research_findings=research_results
   )
   ```

3. **SEO Optimization**
   ```python
   # SEOAgent optimizes for search engines
   keywords = await seo_agent.get_keywords_from_apify("AI Trends")
   optimized_content = seo_agent.optimize_content(blog_draft, keywords)
   ```

4. **Image Generation**
   ```python
   # ImageAgent creates relevant visuals
   image_url = await image_agent._generate_image_with_dalle("AI Trends")
   final_content = image_agent.add_images_to_content(optimized_content, image_url)
   ```

## 🔌 API Integration

### OpenAI Integration

The system uses OpenAI for:
- **Text Generation**: GPT models for blog content creation
- **Image Generation**: DALL-E for relevant visual content

### Apify Integration

Apify actors are used for:
- **Content Research**: Web scraping and information gathering
- **SEO Keywords**: Keyword research and analysis

## 🎯 Advanced Usage

### Creating Custom Agents

1. **Inherit from BaseAgent**
   ```python
   from agents.base_agent import BaseAgent
   
   class CustomAgent(BaseAgent):
       def __init__(self, agent_id="custom_001", **kwargs):
           super().__init__(agent_id=agent_id, **kwargs)
           self.register_capability(
               skill_name="custom_skill",
               description="Performs custom operations"
           )
   ```

2. **Register with Factory**
   ```python
   from core.agent_factory import register_agent_type
   register_agent_type("CustomAgent", CustomAgent)
   ```

### Custom Workflows

Create custom orchestration workflows:

```python
async def custom_workflow(orchestrator, topic):
    # Custom agent coordination logic
    research_task = await orchestrator.assign_task_and_wait(
        research_agent, 
        f"Research {topic}"
    )
    # Additional workflow steps...
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: OpenAI client not initialized
   ```
   Solution: Ensure `OPENAI_API_KEY` is set in your `.env` file

2. **Apify Connection Issues**
   ```
   Warning: APIFY_API_TOKEN not found
   ```
   Solution: Add your Apify token to the `.env` file

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'agents'
   ```
   Solution: Ensure you're running from the project root directory

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("agentsAI").setLevel(logging.DEBUG)
```

## 📈 Performance

### Optimization Tips

1. **Concurrent Operations**: Agents can process tasks concurrently
2. **Caching**: Research results are automatically cached
3. **Resource Management**: Agents handle API rate limiting
4. **Error Recovery**: Robust fallback mechanisms for external API failures

### Monitoring

- Check `logs/agentsAI.log` for system operations
- Monitor `data/` directories for agent performance
- Use structured logging for production deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Run tests before committing
pytest

# Run code formatting
ruff format .

# Run linting
ruff check .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT and DALL-E APIs
- [Apify](https://apify.com/) for web scraping and research capabilities
- [Pydantic](https://pydantic.dev/) for data validation and serialization

## 📞 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the logs in `logs/agentsAI.log` for debugging information
- Review the test files for usage examples

---

**Happy automating! 🤖✨**
