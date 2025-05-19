"""
Prompt generator agent for LLMs.
Receives a task specification dictionary and returns a structured prompt.
"""

import math
from typing import Dict


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text for LLMs (simple approximation).
    """
    # Approximation: 1 token ≈ 4 characters (adjust as needed for your model)
    return math.ceil(len(text) / 4)


def generate_prompt(input_data: Dict) -> Dict:
    """
    Generates a structured prompt for LLMs based on the given specifications.
    Args:
        input_data (dict):
            - task: task description
            - input_type: expected input type
            - output_format: desired output format
            - style: language tone
            - creativity: creativity level
    Returns:
        dict: {
            'raw_prompt': final prompt as string,
            'notes': explanations about design decisions,
            'estimated_tokens': estimated number of tokens in the prompt
        }
    """
    # Extract fields with default values
    task = input_data.get("task", "Unspecified task")
    input_type = input_data.get("input_type", "text")
    output_format = input_data.get("output_format", "default response")
    style = input_data.get("style", "neutral")
    creativity = input_data.get("creativity", "medium")

    # Map creativity to instruction
    creativity_map = {
        "low": "Be objective and avoid adding information that is not present.",
        "medium": "Use moderate creativity as appropriate.",
        "high": "Be creative and explore innovative approaches.",
    }
    # Accept both English and Portuguese for backward compatibility
    creativity_synonyms = {
        "baixa": "low",
        "média": "medium",
        "alta": "high",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    creativity_key = creativity_synonyms.get(str(creativity).lower(), "medium")
    creativity_instruction = creativity_map.get(
        creativity_key, "Use moderate creativity as appropriate."
    )

    # Build prompt sections
    system = (
        f"You are an expert AI assistant with advanced skills in {task.lower()}. "
        f"Your primary objective is to deliver high-quality, accurate, and contextually appropriate results for the user's needs."
    )
    role = (
        f"Adopt the persona of a highly knowledgeable and reliable specialist in {task.lower()}. "
        f"Demonstrate professionalism, precision, and domain expertise in your responses."
    )
    context = (
        f"You will receive an input of type: {input_type}. "
        f"Carefully analyze this input and consider its nuances to ensure your output is relevant and tailored to the task."
    )
    instruction = (
        f"{creativity_instruction} Carefully follow all instructions and requirements provided below to accomplish the task to the best of your ability."
    )
    output = (
        f"Format your response as follows: {output_format}. Ensure your answer adheres strictly to this format and employs a {style} style throughout."
    )

    # Final prompt
    raw_prompt = (
        f"[System]\n{system}\n"
        f"[Role]\n{role}\n"
        f"[Context]\n{context}\n"
        f"[Instruction]\n{instruction}\n"
        f"[Output format]\n{output}\n"
        "\n[INSTRUCTION]:\nPlease process the input as described above.\n"
        "[INPUT]:\n[PASTE THE INPUT HERE]"
    )

    # Explanatory notes
    notes = (
        "The prompt was structured into five sections to guide the LLM clearly. "
        "The persona was defined based on the task. Style and output format follow the user's preferences. "
        "Creativity level was mapped to explicit instructions."
    )

    # Token estimation
    estimated_tokens = estimate_token_count(raw_prompt)

    return {
        "raw_prompt": raw_prompt,
        "notes": notes,
        "estimated_tokens": estimated_tokens,
    }


# Example usage (remove or comment out in production)
if __name__ == "__main__":
    example = {
        "task": "Generate summaries of scientific articles",
        "input_type": "long text",
        "output_format": "summary with up to 5 sentences",
        "style": "formal and concise",
        "creativity": "low",
    }
    result = generate_prompt(example)
    for k, v in result.items():
        print(f"{k}:\n{v}\n{'-' * 40}")
