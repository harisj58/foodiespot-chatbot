import litellm
from litellm import completion
import re
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from ChatbotFunctions import ChatbotFunctions as ChatFn
import ollama

# Configure LiteLLM for Ollama
litellm.set_verbose = False

# === COMBINED SYSTEM PROMPT ===
SYSTEM_PROMPT = """
/no_think
You are DineMate, a restaurant assistant for FoodieSpot chain in Bengaluru.

Core Function
Help users find FoodieSpot restaurants and make reservations.

Essential Rules
1. **Always ask location first** - Get user's preferred area in Bengaluru
2. **Verify with tools** - Use `get_matching_locations` to confirm FoodieSpot exists there
3. **Never guess** - Only use tool results, never make up information
4. **Empty results = inform user** - If no results, say so and suggest alternatives
5. **Never indicate ongoing process** - "Let me check the cuisines available at FoodieSpot locations in XYZ. One moment!" This is incorrect. It means you are meant to make a tool call and you skipped it. This will irritate the user. Always make tool calls in such cases to get the relevant information.
6. **Find alternatives** - If a cuisine that a user is interested in is not served at a location, ALWAYS try to find locations that DO serve that cuisine and let the user know about it. DO NOT leave the user asking for more information, try to obtain it automatically.

Step-by-Step Process
1. Ask: "Which area in Bengaluru are you looking for?"
2. Run: `get_matching_locations` with their area
3. If found: Use `get_cuisine_by_area` to show food options
4. When they pick cuisine: Use `recommend_restaurants`
5. Offer reservation if they want

Available Tools
- `get_matching_locations` - Check if FoodieSpot is in their area
- `get_cuisine_by_area` - Show cuisines available in confirmed area
- `get_all_cuisines` - List all cuisines across Bengaluru
- `get_area_by_cuisine` - Find areas with specific cuisine
- `recommend_restaurants` - Get restaurant recommendations

Response Style
- Friendly but direct
- Use numbered lists for options
- Confirm choices before next step
- For simple greetings ("hi", "thanks"), respond briefly without tools
- If tools return nothing, be honest about it

Introduction
- Tell about yourself in your greeting message
- When asked what can you do, specify all the functionalities you have and how that benefits the user (DO NOT mention name of tools to user)
- Try to greet the user in a unique way each time

Reservation making process
- You need the following data to make a reservation for the user:
    * The name of the FoodieSpot joint the user wishes to make a reservation at. Ensure it is the one the user wishes to dine at. Infer this from the conversation.
    * The full name of the user.
    * The phone number of the user. (Must be exactly 10 digits long)
    * The number of people who will be dining.
    * The time slot for which the user is making a booking at the restaurant. Must be 24-hour time only. If unclear, ask the user to specify AM or PM. Convert to 24-hour time format on your own if the user mentions in 12 hour format.
- Ask these questions one-by-one and once everything is collected, use the `make_reservation` tool to complete the job.

Key Reminder
You only know what tools tell you. If unsure, use tools or ask for clarification.
"""

# Configuration constants
MAX_TOOL_CALLS_PER_CONVERSATION = 5
CONTEXT_WINDOW_LIMIT = 20  # Keep last 20 messages
MIN_FUZZY_MATCH_CONFIDENCE = 70

# Directory for storing threads
THREADS_DIR = "threads"


def create_threads_directory():
    """Create threads directory if it doesn't exist"""
    if not os.path.exists(THREADS_DIR):
        os.makedirs(THREADS_DIR)


def get_thread_file_path(thread_id: str) -> str:
    """Get the file path for a thread"""
    return os.path.join(THREADS_DIR, f"{thread_id}.json")


def manage_context_window(messages: List[Dict]) -> List[Dict]:
    """Limit context to recent messages to prevent token overflow"""
    if len(messages) <= CONTEXT_WINDOW_LIMIT:
        return messages

    # Keep the first message if it's important context, then recent messages
    if len(messages) > CONTEXT_WINDOW_LIMIT:
        # Keep last CONTEXT_WINDOW_LIMIT messages
        return messages[-CONTEXT_WINDOW_LIMIT:]

    return messages


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
        title_prompt = f"""
        /no_think
        Based on this conversation exchange, generate a concise, descriptive title (max 5 words):

        User: {user_message[:200]}
        Assistant: {assistant_response[:200]}

        Generate only the title, nothing else. Make it descriptive but brief."""

        response = completion(
            model="ollama_chat/qwen3:8b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise, descriptive titles for conversations. Respond with only the title, no additional text.",
                },
                {"role": "user", "content": title_prompt},
            ],
            api_base="http://localhost:11434",
            stream=False,
            temperature=0.1,  # Lower temperature for consistent titles
            max_tokens=20,
        )

        if hasattr(response, "choices") and len(response.choices) > 0:
            title = response.choices[0].message.content.strip()
            # Clean up the title
            title = (
                title.replace('"', "")
                .replace("'", "")
                .replace("<think>", "")
                .replace("</think>", "")
                .strip("\n")
                .strip()
            )
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
            model="ollama_chat/qwen3:8b",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="http://localhost:11434",
            stream=False,
            temperature=0.1,
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
    """Get complete response from the model with improved error handling and limits"""
    try:
        # Manage context window to prevent token overflow
        managed_messages = manage_context_window(messages)

        # Prepare messages with single system message
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_messages.extend(managed_messages)

        # Format tools for Ollama
        tools = [
            {"type": "function", "function": description}
            for description in ChatFn.get_descriptions()
        ]

        response = ""
        tool_call_count = 0

        # Initial completion call
        res = ollama.chat(
            model="qwen3:8b",
            messages=formatted_messages,
            tools=tools,
            options={
                # "temperature": 0.1,
                "top_p": 0.9,
                # "num_predict": 1000,  # max_tokens equivalent in Ollama
            },
            think=False,
        )

        # Check if there are tool calls in the response
        if (
            "message" not in res
            or "tool_calls" not in res["message"]
            or not res["message"]["tool_calls"]
        ):
            # No tool calls, return the content directly
            response = res["message"]["content"] if "content" in res["message"] else ""
        else:
            # Handle tool calls with proper limits and error handling
            while (
                "message" in res
                and "tool_calls" in res["message"]
                and res["message"]["tool_calls"]
                and tool_call_count < MAX_TOOL_CALLS_PER_CONVERSATION
            ):
                tool_call_count += 1
                print(f"Tool call #{tool_call_count}")

                # Add assistant message with tool calls
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "content": res["message"].get("content", ""),
                        "tool_calls": res["message"]["tool_calls"],
                    }
                )

                # Execute all tool calls in this turn
                tool_results = []
                for tool_call in res["message"]["tool_calls"]:
                    try:
                        function_name = tool_call["function"]["name"]
                        function_args = tool_call["function"]["arguments"]

                        print(
                            f"Executing: {function_name} with params: {function_args}"
                        )

                        chosen_fn = getattr(ChatFn, function_name)

                        # Parse arguments if they're a string
                        if isinstance(function_args, str):
                            params = json.loads(function_args)
                        else:
                            params = function_args

                        fn_res = chosen_fn(**params)
                        print(f"Function response: {fn_res}")

                        # Add tool result message
                        formatted_messages.append(
                            {
                                "role": "tool",
                                "content": str(fn_res),
                                "name": function_name,
                            }
                        )
                        tool_results.append(fn_res)

                    except Exception as e:
                        error_msg = (
                            f"Tool execution error for {function_name}: {str(e)}"
                        )
                        print(error_msg)
                        formatted_messages.append(
                            {
                                "role": "tool",
                                "content": f"Error: {error_msg}",
                                "name": function_name,
                            }
                        )

                # Get response after tool execution
                try:
                    res = ollama.chat(
                        model="qwen3:8b",
                        messages=formatted_messages,
                        tools=tools,
                        options={
                            # "temperature": 0.1,
                            "top_p": 0.9,
                            # "num_predict": 1000,
                        },
                        think=False,
                    )

                    if (
                        "message" in res
                        and "content" in res["message"]
                        and res["message"]["content"]
                    ):
                        response = res["message"]["content"]
                        break

                except Exception as e:
                    print(f"Error in follow-up completion: {e}")
                    response = "I encountered an error processing your request. Please try again."
                    break

            # Handle case where max tool calls exceeded
            if tool_call_count >= MAX_TOOL_CALLS_PER_CONVERSATION:
                if not response:
                    response = "I've reached the maximum number of tool calls for this conversation. Please start a new conversation or rephrase your request."

        return response

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Complete error in get_response: {error_msg}")
        return error_msg


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
