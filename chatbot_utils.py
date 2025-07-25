import litellm
from litellm import completion
import re
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Generator
from ChatbotFunctions import ChatbotFunctions as ChatFn
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

# Configure LiteLLM for Ollama
litellm.set_verbose = False

# === ADVISOR SYSTEM PROMPT ===
ADVISOR_SYSTEM_PROMPT = """
/no_think
You are a Tool Call Advisor for DineMate, a restaurant assistant for FoodieSpot chain in Bengaluru.

Your job is to analyze conversation context and recommend which tool calls the main agent should make before responding to the user.

Available Tools:
- get_matching_locations(area: str) - Check if FoodieSpot exists in an area
- get_cuisine_by_area(area: str) - Show cuisines available in confirmed area  
- get_all_cuisines() - List all cuisines across Bengaluru
- get_area_by_cuisine(cuisine: str) - Find areas with specific cuisine
- get_area_by_ambience(ambience: str) - Find areas with specific ambience
- get_ambience_by_area(area: str) - Show ambience available in confirmed area
- get_all_ambiences() - List all ambiences across Bengaluru
- recommend_restaurants(area, cuisine: str=None, ambience: str=None) - Get restaurant recommendations
- make_reservation(restaurant_name: str, name: str, phone_number: str, headcount: int, time_slot: dict[str, int]) - Make reservation
{
  "time_slot": {
    "type": "object",
    "description": "The time slot for which the user is making a booking at the restaurant. Must be 24-hour time only. If unclear, ask the user to specify AM or PM.",
    "properties": {
      "hour": {
        "type": "number",
        "description": "The hour at which the user will be arriving at the restaurant. Use 24-hour time only. If unclear, ask the user to specify AM or PM."
      },
      "minute": {
        "type": "number",
        "description": "The minute at which the user will be arriving at the restaurant. Use 24-hour time only. If unclear, ask the user to specify AM or PM."
      }
    },
    "required": [
      "hour",
      "minute"
    ]
  }
}

Analysis Rules:
1. Location is REQUIRED for most operations - if user mentions area, verify it exists
2. If user asks about cuisine availability, need area first
3. If user wants specific cuisine, find areas that serve it
4. For recommendations, area is mandatory, cuisine/ambience optional
5. To make a reservation the following details are required: name, phone number, headcount and time slot. If you think the user wants to make a reservation but required data is missing or even partly missing, DO NOT suggest tool call. Instead advise collecting relevant data first before making a reservation.
6. For reservations, need all required parameters and the user must have confirmed a restaurant first
7. If user just greets or thanks, no tools needed

Context Analysis:
- Look at last 3-5 messages for context
- Identify: locations mentioned, cuisines requested, ambience preferences
- Track conversation progression (greeting → location → cuisine → recommendation → reservation)

Response Format:
Provide recommendations in this exact format:

RECOMMENDED_TOOL_CALLS:
1. tool_name(param1="value1", param2="value2") - Reason for this call
2. tool_name(param="value") - Reason for this call

If no tools needed:
NO_TOOL_CALLS_NEEDED: Brief reason why

Be specific with parameters and provide clear reasoning for each recommendation.
"""

# === COMBINED SYSTEM PROMPT ===
SYSTEM_PROMPT = """
/no_think
You are DineMate, a restaurant assistant for FoodieSpot chain in Bengaluru.

Core Function
Help users find FoodieSpot restaurants and make reservations.

Essential Rules
1. **Location is required** - You cannot recommend a restaurant without first having known the location in Bengaluru. The user might not always have a preferred location so guide him by showing locations he may be interested by using tools accordingly
2. **Verify with tools** - Use `get_matching_locations` to confirm FoodieSpot exists there
3. **Never guess** - Only use tool results, never make up information
4. **Empty results = inform user** - If no results, say so and suggest alternatives
5. **Never indicate ongoing process** - DO NOT respond like: "Let me check the cuisines available at FoodieSpot locations in XYZ. One moment!" This is incorrect. It means you are meant to make a tool call and you skipped it. This will irritate the user. Always make tool calls to get the relevant information before making the final response.
6. **Find alternatives** - If a cuisine that a user is interested in is not served at a location, ALWAYS try to find locations that DO serve that cuisine and let the user know about it. DO NOT leave the user asking for more information, try to obtain it automatically.
7. **Do not skip tool calls** - DO NOT SKIP tool calls as the user progresses with the chat. You must use tool calls to fetch latest data when user changes his mind about the cuisine, location or even ambience.

Available Tools
- `get_matching_locations` - Check if FoodieSpot is in their area
- `get_cuisine_by_area` - Show cuisines available in confirmed area
- `get_all_cuisines` - List all cuisines across Bengaluru
- `get_area_by_cuisine` - Find areas with specific cuisine
- `get_area_by_ambience` - Find areas with specific ambience
- `get_ambience_by_area` - Show ambience available in confirmed area
- `get_all_ambiences` - List all ambiences across Bengaluru
- `recommend_restaurants` - Get restaurant recommendations (area is required for this but ambience and cuisine are optional)

Feel free to make sequential tool calls (one tool call after another) to obtain next piece of information so as to better help the user without the user asking explicitly. That makes you a better restaurant assistant.

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
- Guide the user towards picking a restaurant, ask if they want to dine at a location or if they are in the mood for having a specific type of cuisine
- The user can also look up ambiences in the city and pick a reservation according to that

Reservation making process
- Always show the full restaurant data by using the `recommend_restaurant` tool before asking the user whether they want to make a reservation there. 
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
ADVISOR_CONTEXT_LIMIT = 6  # Last 6 messages for advisor analysis

# Directory for storing threads
THREADS_DIR = "threads"

client = Client(
    host=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
)


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


def get_advisor_context(messages: List[Dict]) -> List[Dict]:
    """Get recent messages for advisor analysis"""
    if len(messages) <= ADVISOR_CONTEXT_LIMIT:
        return messages
    return messages[-ADVISOR_CONTEXT_LIMIT:]


def get_tool_call_recommendations(messages: List[Dict]) -> str:
    """
    Separate advisor agent that analyzes conversation and recommends tool calls
    Returns: String with recommendations to inject into system prompt
    """
    try:
        # Get recent messages for context analysis
        advisor_context = get_advisor_context(messages)

        # Create context summary for advisor
        context_summary = "Recent conversation context:\n"
        for i, msg in enumerate(advisor_context):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Limit content length
            context_summary += f"{i+1}. {role.upper()}: {content}\n"

        # Advisor prompt
        advisor_prompt = f"""
        {context_summary}
        
        Based on the above conversation context, analyze what tool calls the main DineMate agent should make to properly respond to the user's latest message.
        
        Consider:
        - What information is the user seeking?
        - What data is needed to provide a complete response?
        - What locations, cuisines, or ambiences have been mentioned?
        - What stage of the conversation are we in (greeting, location finding, cuisine selection, recommendation, reservation)?
        
        Provide specific tool call recommendations with exact parameters.
        """

        # Call advisor model
        advisor_response = client.chat(
            model="qwen3:8b-fp16",
            messages=[
                {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": advisor_prompt},
            ],
            options={
                "temperature": 0.3,  # Lower temperature for consistent recommendations
                "top_p": 0.8,
            },
            think=False,
        )

        if "message" in advisor_response and "content" in advisor_response["message"]:
            recommendations = advisor_response["message"]["content"].strip()
            print(f"[ADVISOR] Recommendations: {recommendations}")
            return recommendations
        else:
            return "NO_TOOL_CALLS_NEEDED: Advisor response was empty"

    except Exception as e:
        print(f"[ADVISOR] Error getting recommendations: {e}")
        return "NO_TOOL_CALLS_NEEDED: Advisor error occurred"


def inject_advisor_after_user_message(
    messages: List[Dict], recommendations: str
) -> List[Dict]:
    """
    Inject advisor recommendations as a system message after the latest user message
    """
    if not messages:
        return messages

    # Create the advisor injection message
    advisor_message = {
        "role": "system",
        "content": f"""=== TOOL CALL ADVISOR RECOMMENDATIONS ===
{recommendations}

Important: These are suggestions based on conversation analysis. Use your judgment to decide which tools to call and when. The recommendations are meant to help you provide complete and helpful responses without missing important tool calls.
===""",
    }

    # Insert the advisor message after the last user message
    modified_messages = messages.copy()
    modified_messages.append(advisor_message)

    return modified_messages


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
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
            model="ollama_chat/qwen3:8b-fp16",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise, descriptive titles for conversations. Respond with only the title, no additional text.",
                },
                {"role": "user", "content": title_prompt},
            ],
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
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
            model="ollama_chat/qwen3:8b-fp16",
            messages=[{"role": "user", "content": "Hello"}],
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


def get_response_stream(messages) -> Generator[str, None, None]:
    """Get streaming response from the model with advisor recommendations"""
    try:
        # Get advisor recommendations
        yield "🤖 Analyzing conversation context...\n\n"
        recommendations = get_tool_call_recommendations(messages)

        yield "✨ Processing your request...\n\n"

        # Manage context window to prevent token overflow
        managed_messages = manage_context_window(messages)

        # Prepare messages with system prompt
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_messages.extend(managed_messages)

        # Inject advisor recommendations after the latest user message
        formatted_messages = inject_advisor_after_user_message(
            formatted_messages, recommendations
        )

        # Format tools for Ollama
        tools = [
            {"type": "function", "function": description}
            for description in ChatFn.get_descriptions()
        ]

        tool_call_count = 0
        accumulated_response = ""

        # Initial completion call with streaming
        stream = client.chat(
            model="qwen3:8b-fp16",
            messages=formatted_messages,
            tools=tools,
            options={
                "top_p": 0.9,
            },
            think=False,
            stream=True,
        )

        # Check if we need to handle tool calls or can stream directly
        first_chunk = next(stream, None)
        if not first_chunk:
            yield "Error: No response from model"
            return

        # If the first chunk has tool calls, we need to handle them first
        if (
            "message" in first_chunk
            and "tool_calls" in first_chunk["message"]
            and first_chunk["message"]["tool_calls"]
        ):
            # Tool calls detected - handle them non-streaming first
            # Reconstruct the full response for tool handling
            full_response = first_chunk

            # Continue reading the stream to get complete tool call info
            for chunk in stream:
                # print(f"Chunk: {chunk}")
                if "message" in chunk:
                    if (
                        "tool_calls" in chunk["message"]
                        and chunk["message"]["tool_calls"]
                    ):
                        # Merge tool calls
                        if "tool_calls" not in full_response["message"]:
                            full_response["message"]["tool_calls"] = []
                        full_response["message"]["tool_calls"].extend(
                            chunk["message"]["tool_calls"]
                        )

                    if "content" in chunk["message"] and chunk["message"]["content"]:
                        if "content" not in full_response["message"]:
                            full_response["message"]["content"] = ""
                        full_response["message"]["content"] += chunk["message"][
                            "content"
                        ]

            # Handle tool calls
            while (
                "message" in full_response
                and "tool_calls" in full_response["message"]
                and full_response["message"]["tool_calls"]
                and tool_call_count < MAX_TOOL_CALLS_PER_CONVERSATION
            ):
                tool_call_count += 1
                yield f"🔧 Executing tools... (Call #{tool_call_count})\n\n"

                # Add assistant message with tool calls
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "content": full_response["message"].get("content", ""),
                        "tool_calls": full_response["message"]["tool_calls"],
                    }
                )

                # Execute all tool calls in this turn
                for tool_call in full_response["message"]["tool_calls"]:
                    try:
                        function_name = tool_call["function"]["name"]
                        function_args = tool_call["function"]["arguments"]

                        yield f"📊 Using {function_name}...\n\n"

                        chosen_fn = getattr(ChatFn, function_name)

                        # Parse arguments if they're a string
                        if isinstance(function_args, str):
                            params = json.loads(function_args)
                        else:
                            params = function_args

                        fn_res = chosen_fn(**params)
                        print(
                            f"🔧 Executing {function_name} with {function_args} gave:\n{fn_res}"
                        )

                        # Add tool result message
                        formatted_messages.append(
                            {
                                "role": "tool",
                                "content": str(fn_res),
                                "name": function_name,
                            }
                        )

                    except Exception as e:
                        error_msg = (
                            f"Tool execution error for {function_name}: {str(e)}"
                        )
                        print(error_msg)
                        yield f"❌ {error_msg}\n\n"
                        formatted_messages.append(
                            {
                                "role": "tool",
                                "content": f"Error: {error_msg}",
                                "name": function_name,
                            }
                        )

                # Get streaming response after tool execution
                try:
                    stream = client.chat(
                        model="qwen3:8b-fp16",
                        messages=formatted_messages,
                        tools=tools,
                        options={
                            "top_p": 0.9,
                        },
                        think=False,
                        stream=True,
                    )

                    # Stream the response after tool execution
                    tool_response_started = False
                    for chunk in stream:
                        # print(f"Chunk: {chunk}")
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            if content:
                                if not tool_response_started:
                                    yield f"✨ **Response:**\n\n"
                                    tool_response_started = True
                                accumulated_response += content
                                yield content

                        # Check if there are more tool calls
                        if (
                            "message" in chunk
                            and "tool_calls" in chunk["message"]
                            and chunk["message"]["tool_calls"]
                        ):
                            full_response = chunk
                            break
                    else:
                        # No more tool calls, we're done
                        break

                except Exception as e:
                    yield f"❌ Error in follow-up completion: {str(e)}"
                    break

            # Handle case where max tool calls exceeded
            if tool_call_count >= MAX_TOOL_CALLS_PER_CONVERSATION:
                yield "\n\n⚠️ Maximum tool calls reached. Please start a new conversation."

        else:
            # No tool calls, stream the response directly
            if "message" in first_chunk and "content" in first_chunk["message"]:
                content = first_chunk["message"]["content"]
                if content:
                    accumulated_response += content
                    yield content

            # Continue streaming the rest
            for chunk in stream:
                # print(f"Chunk: {chunk}")
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:
                        accumulated_response += content
                        yield content

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        yield error_msg


def get_response(messages):
    """Get complete response from the model with advisor recommendations"""
    try:
        # Get advisor recommendations
        print("[ADVISOR] Getting tool call recommendations...")
        recommendations = get_tool_call_recommendations(messages)

        # Manage context window to prevent token overflow
        managed_messages = manage_context_window(messages)

        # Prepare messages with system prompt
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_messages.extend(managed_messages)

        # Inject advisor recommendations after the latest user message
        formatted_messages = inject_advisor_after_user_message(
            formatted_messages, recommendations
        )

        # Format tools for Ollama
        tools = [
            {"type": "function", "function": description}
            for description in ChatFn.get_descriptions()
        ]

        response = ""
        tool_call_count = 0

        # Initial completion call
        res = client.chat(
            model="qwen3:8b-fp16",
            messages=formatted_messages,
            tools=tools,
            options={
                "top_p": 0.9,
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
                    res = client.chat(
                        model="qwen3:8b-fp16",
                        messages=formatted_messages,
                        tools=tools,
                        options={
                            "top_p": 0.9,
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
