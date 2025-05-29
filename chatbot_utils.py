import litellm
from litellm import completion
import re
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from ChatbotFunctions import ChatbotFunctions as ChatFn
from copy import deepcopy

# Configure LiteLLM for Ollama
litellm.set_verbose = False

# === SYSTEM PROMPTS ===

SYSTEM_PROMPTS = [
    """
    You are the responder agent of FoodieSpot - a chain of restaurants across Bengaluru. Always introduce yourself as DineMate. Your job is to recommend users restaurant options by FoodieSpot based on user's preferences.
    
    Once that is done, you are to also take reservations should the user ask for it. To help craft better responses, you may be provided with results of certain tool calls. Use them appropriately.
    """,
    """
    Start by asking the location the user is interested in. Once that you have obtained matching locations by using `get_matching_locations` tool, check if the location specified by user is in the list. If not, show them the list and confirm otherwise proceed.
    """,
    """
    Next show the user the various cuisines available at the location they desire by calling `get_cuisine_by_area` tool (do NOT make up cuisine options). Make sure the area specified by user is present in result of `get_matching_locations` tool call. Once they have chosen their desired cuisine type, we can recommend them suitable restaurants.
    """,
    """
    If the user requests, you can show them all the cuisine types available across all FoodieSpot joints in Bengaluru by using the `get_all_cuisines` tool.
    """,
    """
    You must:
    - Use tools when needed (e.g., user asks for live data or dynamic actions). If you are confused whether user is asking live or stale data, always assume they are asking for live data and use tools to fetch the latest data available.
    - Use tools if and only if the answer is unavailable via built-in knowledge.
    - Do NOT hallucinate tool usage.
    - If a tool is needed, use the most appropriate tool with precise arguments.
    - You can call multiple tools if the query requires it.
    - Reply with tool calls OR an assistant message – not both.
    - Do NOT skip on tool usage by looking at past conversation and coming up with made up response.

    === EXAMPLES ===
    User: "What's the weather in Paris?"
    ✅ Tool call is required.

    User: "Tell me about the Eiffel Tower."
    ✅ Answer directly. Tool call is NOT needed.
    """,
    """
    === GUIDELINES ===

    1. Use **tool outputs** when present to inform your response. DO NOT fabricate or guess information that the tools have not provided.
    2. If there are no tool outputs, or they are insufficient, rely entirely on your own internal knowledge to respond as naturally and usefully as possible.
    3. NEVER mention that tools were used or not used. Just reply normally.
    4. Do NOT hallucinate facts, names, numbers, or results.
    5. If you are unclear on what to say, ask the user a clarifying question — do not make assumptions.
    6. For casual or simple greetings like “hi”, “thanks”, or “bye”, respond politely and briefly. No tools or extra reasoning needed.
    """,
    """
    === ALWAYS FOLLOW THIS STYLE ===
    - Be natural, friendly, and direct.
    - If tool results are available, use only what is in them.
    - If no tools were needed, respond as a normal assistant using your internal knowledge.
    - If something is missing or unclear in tool outputs, politely tell the user what’s missing or ask for more input.

    Your job is to make the conversation feel seamless and human — not robotic or overly literal.
    """,
]
# Directory for storing threads
THREADS_DIR = "threads"


def create_threads_directory():
    """Create threads directory if it doesn't exist"""
    if not os.path.exists(THREADS_DIR):
        os.makedirs(THREADS_DIR)


def get_thread_file_path(thread_id: str) -> str:
    """Get the file path for a thread"""
    return os.path.join(THREADS_DIR, f"{thread_id}.json")


def save_thread(thread_id: str, title: str, messages: List[Dict]) -> bool:
    """Save thread to file"""
    try:
        create_threads_directory()

        thread_data = {
            "id": thread_id,
            "title": title,
            "messages": messages,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # If thread already exists, preserve the original created_at
        file_path = get_thread_file_path(thread_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    thread_data["created_at"] = existing_data.get(
                        "created_at", thread_data["created_at"]
                    )
            except:
                pass  # Use new created_at if we can't read existing file

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(thread_data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error saving thread {thread_id}: {e}")
        return False


def load_thread(thread_id: str) -> Optional[Dict]:
    """Load thread from file"""
    try:
        file_path = get_thread_file_path(thread_id)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading thread {thread_id}: {e}")
        return None


def delete_thread(thread_id: str) -> bool:
    """Delete thread file"""
    try:
        file_path = get_thread_file_path(thread_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return False


def get_all_threads() -> List[Dict]:
    """Get all threads, sorted by update time (most recent first)"""
    threads = []

    try:
        create_threads_directory()

        for filename in os.listdir(THREADS_DIR):
            if filename.endswith(".json"):
                thread_id = filename[:-5]  # Remove .json extension
                thread_data = load_thread(thread_id)
                if thread_data:
                    threads.append(
                        {
                            "id": thread_data["id"],
                            "title": thread_data.get("title", "Untitled Chat"),
                            "created_at": thread_data.get("created_at", "Unknown"),
                            "updated_at": thread_data.get("updated_at", "Unknown"),
                            "message_count": len(thread_data.get("messages", [])),
                        }
                    )

        # Sort by updated_at (most recent first)
        threads.sort(key=lambda x: x["updated_at"], reverse=True)

    except Exception as e:
        print(f"Error getting threads list: {e}")

    return threads


def generate_thread_title(user_message: str, assistant_response: str) -> str:
    """Generate a concise title for the thread based on the first exchange"""
    try:
        # Create a prompt for title generation
        title_prompt = f"""Based on this conversation exchange, generate a concise, descriptive title (max 5 words):

        User: {user_message[:200]}
        Assistant: {assistant_response[:200]}

        Generate only the title, nothing else. Make it descriptive but brief."""

        response = completion(
            model="ollama_chat/llama3-groq-tool-use:8b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise, descriptive titles for conversations. Respond with only the title, no additional text.",
                },
                {"role": "user", "content": title_prompt},
            ],
            api_base="http://localhost:11434",
            stream=False,
            temperature=0.3,
            max_tokens=20,
        )

        if hasattr(response, "choices") and len(response.choices) > 0:
            title = response.choices[0].message.content.strip()
            # Clean up the title
            title = title.replace('"', "").replace("'", "").strip()
            # Limit length and ensure it's reasonable
            if len(title) > 50:
                title = title[:47] + "..."
            if title and len(title) > 3:
                return title

    except Exception as e:
        print(f"Error generating title: {e}")

    # Fallback: create title from first few words of user message
    words = user_message.split()[:4]
    fallback_title = " ".join(words)
    if len(fallback_title) > 30:
        fallback_title = fallback_title[:27] + "..."

    return fallback_title if fallback_title else "New Chat"


def test_model_connection():
    """Test if the Ollama model is available"""
    try:
        response = completion(
            model="ollama_chat/llama3-groq-tool-use:8b",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="http://localhost:11434",
            stream=False,
            max_tokens=10,
        )
        return True, "Model is available"
    except Exception as e:
        return False, f"Model connection failed: {str(e)}"


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


def get_response(messages):
    """Get complete response from the model (non-streaming)"""
    try:
        # Prepare messages with system message
        formatted_messages = [
            {"role": "system", "content": prompt} for prompt in SYSTEM_PROMPTS
        ]
        formatted_messages.extend(messages)

        tools = [
            {"type": "function", "function": description}
            for description in ChatFn.get_descriptions()
        ]

        response = ""

        res = completion(
            model="ollama_chat/llama3-groq-tool-use:8b",
            messages=formatted_messages,
            api_base="http://localhost:11434",
            stream=False,
            tools=tools,
            temperature=0.5,
        )

        content = res.choices[0].message.content
        if res.choices[0].message.tool_calls is None:
            response = content
        else:
            tool_called = True
            res_copy = res
            while tool_called:
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": res.choices[0].message.tool_calls,
                    }
                )

                for tool_call in res.choices[0].message.tool_calls:
                    null = None  # For JSON null
                    print(
                        f"Performing tool call with chosen function: {tool_call.function.name}. The following params were supplied: {tool_call.function.arguments}"
                    )
                    chosen_fn = eval(f"ChatFn.{tool_call.function.name}")
                    params = json.loads(f"{tool_call.function.arguments}")
                    fn_res = chosen_fn(**params)
                    print(f"Function response: {fn_res}")

                    formatted_messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": str(fn_res),
                        }
                    )

                # Attempt to get new completion with tool responses
                max_tries = 3
                tries = 1
                tool_called = False  # Reset before checking
                while tries <= max_tries:
                    res = completion(
                        model="ollama_chat/llama3-groq-tool-use:8b",
                        messages=formatted_messages,
                        api_base="http://localhost:11434",
                        stream=False,
                        tools=tools,
                        temperature=0.5,
                    )

                    # Update based on new response
                    if res.choices[0].message.content is not None:
                        print("Response generated")
                        response = res.choices[0].message.content
                        break

                    if res.choices[0].message.tool_calls:
                        print("More tool calls required")
                        print(formatted_messages)
                        tool_called = True
                        break

                    tries += 1

                if tries > max_tries:
                    response = "There was some problem. Please ask your query again."
                    break

        return response

    except Exception as e:
        return f"Error: {str(e)}"


def export_all_threads() -> str:
    """Export all threads as a single JSON file"""
    try:
        all_threads = []
        threads_list = get_all_threads()

        for thread_info in threads_list:
            thread_data = load_thread(thread_info["id"])
            if thread_data:
                all_threads.append(thread_data)

        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_threads": len(all_threads),
            "threads": all_threads,
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error exporting all threads: {e}")
        return None


def search_threads(query: str) -> List[Dict]:
    """Search threads by title or content"""
    try:
        matching_threads = []
        threads_list = get_all_threads()
        query_lower = query.lower()

        for thread_info in threads_list:
            # Search in title
            if query_lower in thread_info["title"].lower():
                matching_threads.append(thread_info)
                continue

            # Search in messages
            thread_data = load_thread(thread_info["id"])
            if thread_data:
                for message in thread_data.get("messages", []):
                    if query_lower in message.get("content", "").lower():
                        matching_threads.append(thread_info)
                        break

        return matching_threads
    except Exception as e:
        print(f"Error searching threads: {e}")
        return []
