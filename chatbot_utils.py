import litellm
from litellm import completion
import re

# Configure LiteLLM for Ollama
litellm.set_verbose = False

# System prompts for the chatbot (as list of strings)
SYSTEM_PROMPTS = [
    "You are a helpful AI assistant powered by Deepseek.",
    "You are knowledgeable, friendly, and provide accurate information.",
    "When you need to think through complex problems, you can use <think></think> tags to show your reasoning process before giving your final answer.",
    "Always be concise but thorough in your responses.",
]

# Convert list to single system prompt
SYSTEM_PROMPT = " ".join(SYSTEM_PROMPTS)


def test_model_connection():
    """Test if the Ollama model is available"""
    try:
        response = completion(
            model="ollama/deepseek-r1:8b",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="http://localhost:11434",
            stream=False,
            max_tokens=10,
        )
        return True, "Model is available"
    except Exception as e:
        return False, f"Model connection failed: {str(e)}"


def extract_current_thinking(text):
    """
    Extract current thinking content from partial response
    Returns the thinking content if <think> is present, even if not closed yet
    """
    if "<think>" not in text:
        return None

    # Find the start of thinking
    start_idx = text.find("<think>") + 7

    # Check if thinking is complete
    if "</think>" in text:
        end_idx = text.find("</think>")
        return text[start_idx:end_idx].strip()
    else:
        # Return partial thinking content
        return text[start_idx:].strip()


def parse_thinking_response(text):
    """
    Parse response to separate thinking content from main response
    Returns: (thinking_content, main_response)
    """
    # Find all thinking blocks
    think_pattern = r"<think>(.*?)</think>"
    thinking_matches = re.findall(think_pattern, text, re.DOTALL)

    # Remove thinking blocks from main response
    main_response = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()

    # Combine all thinking content
    thinking_content = "\n\n".join(thinking_matches) if thinking_matches else None

    return thinking_content, main_response


def get_response_stream(messages):
    """Get streaming response from the model"""
    try:
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_messages.extend(messages)

        response = completion(
            model="ollama/deepseek-r1:8b",
            messages=formatted_messages,
            api_base="http://localhost:11434",
            stream=True,
            temperature=0.7,
            max_tokens=2000,
        )

        full_response = ""
        for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    chunk_content = delta.content
                    full_response += chunk_content
                    yield chunk_content, full_response

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield error_msg, error_msg


def get_non_stream_response(messages):
    """Get non-streaming response from the model (for connection testing)"""
    try:
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_messages.extend(messages)

        response = completion(
            model="ollama/deepseek-r1:8b",
            messages=formatted_messages,
            api_base="http://localhost:11434",
            stream=False,
            temperature=0.7,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"
