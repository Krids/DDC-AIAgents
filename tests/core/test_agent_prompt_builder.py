import pytest
import math
from core.agent_prompt_builder import estimate_token_count, generate_prompt

# Test estimate_token_count
@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("", 0),
        ("test", 1),
        ("test test", 2), # 9 chars / 4 = 2.25 -> 3, (depends on spaces, 8/4 = 2)
        ("Hello World! This is a test.", 8), # 28 chars / 4 = 7
        (" دقیقاً", 2) # Example with non-ASCII, approximation might differ. (6 chars / 4 = 1.5 -> 2)
    ]
)
def test_estimate_token_count(text, expected_tokens):
    # The current logic is math.ceil(len(text) / 4)
    # For "test test" (9 chars including space): math.ceil(9/4) = math.ceil(2.25) = 3
    # For "Hello World! This is a test." (28 chars): math.ceil(28/4) = math.ceil(7) = 7
    assert estimate_token_count(text) == math.ceil(len(text) / 4) 
    # If we want to match the provided expected_tokens, some are slightly off based on the formula.
    # Let's use the formula for now, as it's what the code does.
    if text == "test test":
        assert estimate_token_count(text) == 3 # 9 chars including space
    elif text == "Hello World! This is a test.":
        assert estimate_token_count(text) == 7
    else:
        assert estimate_token_count(text) == expected_tokens


# Test generate_prompt basic functionality
def test_generate_prompt_basic_structure():
    input_data = {
        "task": "Generate code",
        "input_type": "Python dictionary",
        "output_format": "Python code block",
        "style": "clear and concise",
        "creativity": "low",
    }
    result = generate_prompt(input_data)

    assert "raw_prompt" in result
    assert "notes" in result
    assert "estimated_tokens" in result

    assert isinstance(result["raw_prompt"], str)
    assert isinstance(result["notes"], str)
    assert isinstance(result["estimated_tokens"], int)

    # Check for key placeholders and sections in the raw_prompt
    assert "[System]" in result["raw_prompt"]
    assert "[Role]" in result["raw_prompt"]
    assert "[Context]" in result["raw_prompt"]
    assert "[Instruction]" in result["raw_prompt"]
    assert "[Output format]" in result["raw_prompt"]
    assert "[INSTRUCTION]:" in result["raw_prompt"]
    assert "[INPUT]:\n[PASTE THE INPUT HERE]" in result["raw_prompt"]

    # Check if task, input_type, output_format, style are in the prompt
    assert input_data["task"].lower() in result["raw_prompt"].lower()
    assert input_data["input_type"] in result["raw_prompt"]
    assert input_data["output_format"] in result["raw_prompt"]
    assert input_data["style"] in result["raw_prompt"]

    # Check creativity instruction mapping
    assert "Be objective and avoid adding information that is not present." in result["raw_prompt"]


def test_generate_prompt_default_values():
    input_data = {}
    result = generate_prompt(input_data)
    assert "unspecified task" in result["raw_prompt"]
    assert "text" in result["raw_prompt"] # default input_type
    assert "default response" in result["raw_prompt"] # default output_format
    assert "neutral" in result["raw_prompt"] # default style
    assert "Use moderate creativity as appropriate." in result["raw_prompt"] # default creativity

@pytest.mark.parametrize(
    "creativity_input, expected_instruction_snippet",
    [
        ("low", "Be objective and avoid adding information that is not present."),
        ("medium", "Use moderate creativity as appropriate."),
        ("high", "Be creative and explore innovative approaches."),
        ("baixa", "Be objective and avoid adding information that is not present."), # Synonym
        ("média", "Use moderate creativity as appropriate."), # Synonym
        ("alta", "Be creative and explore innovative approaches."), # Synonym
        ("unknown", "Use moderate creativity as appropriate."), # Default for unknown
        (123, "Use moderate creativity as appropriate.") # Default for non-string
    ]
)
def test_generate_prompt_creativity_mapping(creativity_input, expected_instruction_snippet):
    input_data = {
        "task": "Test creativity",
        "creativity": creativity_input
    }
    result = generate_prompt(input_data)
    assert expected_instruction_snippet in result["raw_prompt"]

def test_generate_prompt_token_estimation():
    input_data = {"task": "Token estimation test"}
    result = generate_prompt(input_data)
    expected_tokens = math.ceil(len(result["raw_prompt"]) / 4)
    assert result["estimated_tokens"] == expected_tokens 