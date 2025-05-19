from core.agent_prompt_builder import generate_prompt


def test_generate_prompt_returns_dict():
    input_data = {
        "task": "Summarize scientific articles",
        "input_type": "long text",
        "output_format": "summary with up to 5 sentences",
        "style": "formal and concise",
        "creativity": "low",
    }
    result = generate_prompt(input_data)
    assert isinstance(result, dict)


def test_generate_prompt_contains_raw_prompt():
    input_data = {
        "task": "Summarize scientific articles",
        "input_type": "long text",
        "output_format": "summary with up to 5 sentences",
        "style": "formal and concise",
        "creativity": "low",
    }
    result = generate_prompt(input_data)
    assert "raw_prompt" in result
    assert isinstance(result["raw_prompt"], str)
    assert "[System]" in result["raw_prompt"]
    assert "[INPUT]" in result["raw_prompt"]


def test_generate_prompt_contains_notes():
    input_data = {
        "task": "Summarize scientific articles",
        "input_type": "long text",
        "output_format": "summary with up to 5 sentences",
        "style": "formal and concise",
        "creativity": "low",
    }
    result = generate_prompt(input_data)
    assert "notes" in result
    assert isinstance(result["notes"], str)
    assert "The prompt was structured" in result["notes"]


def test_generate_prompt_estimated_tokens_is_int():
    input_data = {
        "task": "Summarize scientific articles",
        "input_type": "long text",
        "output_format": "summary with up to 5 sentences",
        "style": "formal and concise",
        "creativity": "low",
    }
    result = generate_prompt(input_data)
    assert "estimated_tokens" in result
    assert isinstance(result["estimated_tokens"], int)
    assert result["estimated_tokens"] > 0


def test_generate_prompt_default_values():
    # Test missing fields are handled with defaults
    input_data = {}
    result = generate_prompt(input_data)
    assert "Unspecified task" in result["raw_prompt"]
    assert "text" in result["raw_prompt"] or "default response" in result["raw_prompt"]
    assert (
        "neutral" in result["raw_prompt"]
        or "Use a neutral style." in result["raw_prompt"]
    )


def test_generate_prompt_creativity_synonyms():
    # Portuguese synonyms should be accepted
    input_data = {
        "task": "Summarize scientific articles",
        "input_type": "long text",
        "output_format": "summary with up to 5 sentences",
        "style": "formal and concise",
        "creativity": "baixa",
    }
    result = generate_prompt(input_data)
    assert "Be objective" in result["raw_prompt"]
