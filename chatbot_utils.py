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

# === NEW ADVISOR SYSTEM PROMPT ===
ADVISOR_SYSTEM_PROMPT = """
/no_think
You are a Tool Execution Advisor for DineMate, a restaurant assistant for FoodieSpot chain in Bengaluru.

Your ONLY job is to analyze the conversation and execute the necessary tool calls to gather information. You do NOT provide final responses to users.

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

Tool Execution Rules:
1. You can make up to 5 sequential tool calls
2. Make tool calls based on what information is needed to answer the user's query
3. Stop when you have sufficient information or reach the 5-call limit
4. If a tool call fails, stop and return results collected so far
5. Always prioritize location verification first if area is mentioned
6. For recommendations, ensure you have area confirmed before calling recommend_restaurants
7. For reservations, ensure all required data is present before calling make_reservation. DO NOT assume any data for making reservations.
8. Here is what you need to make a reservation: restaurant, name, phone number, headcount and time slot. If any of this data is missing from the user, inform the responder to collect the data from user. 

Decision Logic:
- If user has finalized a location, proceed with showing the restaurant in that area.
- If user mentions area: verify with get_matching_locations first
- If user wants cuisine info: get_cuisine_by_area or get_area_by_cuisine
- If user wants ambience info: get_ambience_by_area or get_area_by_ambience  
- If user wants recommendations: ensure area is known, then call recommend_restaurants
- If user wants reservation: ensure all required data, then call make_reservation
- If just greeting/thanks: no tools needed

Response Format:
If tool calls are needed: Execute the tools and let the tool results speak
If no tools needed: Provide a brief reasoning (2-3 sentences max)

Remember: You execute tools, you don't provide final answers to users.
"""
# === NEW FINAL RESPONDER SYSTEM PROMPT ===
FINAL_RESPONDER_SYSTEM_PROMPT = """
/no_think
You are DineMate, a restaurant assistant for FoodieSpot chain in Bengaluru.

CRITICAL RULE: You can ONLY use information that is explicitly provided in the tool execution results. You must NEVER make up, assume, or infer any information that is not directly stated in the tool results.

Your job is to provide the final response to users based STRICTLY on:
1. The conversation context
2. Tool execution results provided to you
3. The user's latest message

Response Guidelines:
1. **NEVER HALLUCINATE**: Use ONLY the exact information from tool results - never make up data, restaurant names, addresses, phone numbers, or any other details
2. **Be explicit about limitations**: If tool results are empty, incomplete, or don't contain specific information, clearly state this
3. **Quote directly**: When referencing tool results, use the exact wording and data provided
- If user has finalized a location, proceed with showing the restaurant in that area.
4. **No assumptions**: Don't assume availability, pricing, menu items, or any details not explicitly provided
5. **Admit gaps**: Say "I don't have that information" rather than guessing
6. **For empty results**: State exactly what was searched and that no results were found
7. **For partial results**: Only present the information that was actually returned

Forbidden Actions:
- Creating restaurant names, addresses, or contact details
- Assuming cuisines, ambiences, or features not explicitly listed
- Making up availability or timing information
- Inferring location details not provided in tool results
- Adding descriptive details about restaurants not in the tool results

Required Phrases for Common Situations:
- When no results: "I searched for [specific query] but found no FoodieSpot locations matching your criteria."
- When partial results: "Based on the information I found, here's what I can tell you..."
- When missing details: "I don't have information about [specific detail] in my current results."
- When uncertain: "The available information shows..."

You will receive tool results in this format:
- tool_name: result_data
- execution_summary: brief summary of what was executed

Use ONLY this information to craft your response. When in doubt, be conservative and admit limitations rather than fill gaps with assumptions.
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


def execute_tools_with_advisor(messages: List[Dict]) -> Dict:
    """
    New advisor agent that only executes tools and returns structured results
    """
    try:
        # Get recent context for advisor
        advisor_context = get_advisor_context(messages)

        # Format tools for Ollama
        tools = [
            {"type": "function", "function": description}
            for description in ChatFn.get_descriptions()
        ]

        # Prepare advisor messages
        advisor_messages = [{"role": "system", "content": ADVISOR_SYSTEM_PROMPT}]
        advisor_messages.extend(advisor_context)

        tool_results = {
            "tools_executed": [],
            "execution_summary": "",
            "total_calls": 0,
            "status": "success",
        }

        max_tool_calls = 5
        current_calls = 0

        # Initial call to advisor
        res = client.chat(
            model="qwen3:8b",
            messages=advisor_messages,
            tools=tools,
            options={"temperature": 0.3, "top_p": 0.8},
            think=False,
        )

        # Execute tool calls if present
        while (
            "message" in res
            and "tool_calls" in res["message"]
            and res["message"]["tool_calls"]
            and current_calls < max_tool_calls
        ):
            # Add assistant message with tool calls
            advisor_messages.append(
                {
                    "role": "assistant",
                    "content": res["message"].get("content", ""),
                    "tool_calls": res["message"]["tool_calls"],
                }
            )

            # Execute all tool calls in this turn
            for tool_call in res["message"]["tool_calls"]:
                if current_calls >= max_tool_calls:
                    break

                try:
                    current_calls += 1
                    function_name = tool_call["function"]["name"]
                    function_args = tool_call["function"]["arguments"]

                    print(
                        f"[ADVISOR] Executing {function_name} (call #{current_calls})"
                    )

                    chosen_fn = getattr(ChatFn, function_name)

                    # Parse arguments
                    if isinstance(function_args, str):
                        params = json.loads(function_args)
                    else:
                        params = function_args

                    # Execute function
                    fn_result = chosen_fn(**params)
                    print(f"[ADVISOR] The tool call gave results:\n{fn_result}")

                    # Store result
                    tool_results["tools_executed"].append(
                        {
                            "tool_name": function_name,
                            "parameters": params,
                            "result": fn_result,
                            "call_number": current_calls,
                        }
                    )

                    # Add tool result to conversation
                    advisor_messages.append(
                        {
                            "role": "tool",
                            "content": str(fn_result),
                            "name": function_name,
                        }
                    )

                except Exception as e:
                    error_msg = f"Tool execution error for {function_name}: {str(e)}"
                    print(f"[ADVISOR] {error_msg}")

                    tool_results["tools_executed"].append(
                        {
                            "tool_name": function_name,
                            "parameters": function_args,
                            "result": f"ERROR: {error_msg}",
                            "call_number": current_calls,
                        }
                    )

                    # Stop on error
                    tool_results["status"] = "error"
                    tool_results["total_calls"] = current_calls
                    tool_results["execution_summary"] = (
                        f"Executed {current_calls} tool calls, stopped due to error in {function_name}"
                    )
                    return tool_results

            # If we've hit the limit, stop
            if current_calls >= max_tool_calls:
                break

            # Get next response to see if more tools needed
            try:
                res = client.chat(
                    model="qwen3:8b",
                    messages=advisor_messages,
                    tools=tools,
                    options={"temperature": 0.3, "top_p": 0.8},
                    think=False,
                )
            except Exception as e:
                print(f"[ADVISOR] Error in follow-up call: {e}")
                break

        # Set final status
        tool_results["total_calls"] = current_calls

        if current_calls == 0:
            # No tools were executed, get reasoning from advisor
            advisor_reasoning = res["message"].get(
                "content", "No tools needed for this query."
            )
            tool_results["execution_summary"] = (
                f"No tools executed. Reasoning: {advisor_reasoning}"
            )
        else:
            tool_results["execution_summary"] = (
                f"Successfully executed {current_calls} tool calls to gather required information."
            )

        print(f"[ADVISOR] Completed with {current_calls} tool calls")
        return tool_results

    except Exception as e:
        print(f"[ADVISOR] Error in tool execution: {e}")
        return {
            "tools_executed": [],
            "execution_summary": f"Error in tool execution: {str(e)}",
            "total_calls": 0,
            "status": "error",
        }


def get_final_response_stream(
    messages: List[Dict], tool_results: Dict
) -> Generator[str, None, None]:
    """
    Final responder agent that generates streaming response using tool results
    """
    try:
        # Prepare context for final responder
        recent_messages = messages[-3:] if len(messages) > 3 else messages

        # Create tool results summary for the final responder
        tool_summary = ""
        if tool_results["tools_executed"]:
            tool_summary = "\n=== TOOL EXECUTION RESULTS ===\n"
            for tool_result in tool_results["tools_executed"]:
                tool_summary += (
                    f"\n{tool_result['tool_name']}:\n{tool_result['result']}\n"
                )
            tool_summary += (
                f"\nExecution Summary: {tool_results['execution_summary']}\n"
            )
            tool_summary += "===============================\n"
        else:
            tool_summary = f"\n=== NO TOOLS EXECUTED ===\n{tool_results['execution_summary']}\n========================\n"

        # Prepare final responder messages
        final_messages = [{"role": "system", "content": FINAL_RESPONDER_SYSTEM_PROMPT}]
        final_messages.extend(recent_messages)

        # Add tool results as system message
        final_messages.append({"role": "system", "content": tool_summary})

        # Stream the final response
        stream = client.chat(
            model="qwen3:8b",
            messages=final_messages,
            options={"top_p": 0.9, "temperature": 0.7},
            think=False,
            stream=True,
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                if content:
                    yield content

    except Exception as e:
        yield f"Error generating response: {str(e)}"


def get_final_response(messages: List[Dict], tool_results: Dict) -> str:
    """
    Final responder agent that generates complete response using tool results
    """
    try:
        # Prepare context for final responder
        recent_messages = messages[-3:] if len(messages) > 3 else messages

        # Create tool results summary
        tool_summary = ""
        if tool_results["tools_executed"]:
            tool_summary = "\n=== TOOL EXECUTION RESULTS ===\n"
            for tool_result in tool_results["tools_executed"]:
                tool_summary += (
                    f"\n{tool_result['tool_name']}:\n{tool_result['result']}\n"
                )
            tool_summary += (
                f"\nExecution Summary: {tool_results['execution_summary']}\n"
            )
            tool_summary += "===============================\n"
        else:
            tool_summary = f"\n=== NO TOOLS EXECUTED ===\n{tool_results['execution_summary']}\n========================\n"

        # Prepare final responder messages
        final_messages = [{"role": "system", "content": FINAL_RESPONDER_SYSTEM_PROMPT}]
        final_messages.extend(recent_messages)

        # Add tool results as system message
        final_messages.append({"role": "system", "content": tool_summary})

        # Get the final response
        res = client.chat(
            model="qwen3:8b",
            messages=final_messages,
            options={"top_p": 0.9, "temperature": 0.7},
            think=False,
        )

        return res["message"].get(
            "content", "I apologize, but I couldn't generate a proper response."
        )

    except Exception as e:
        return f"Error generating response: {str(e)}"


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
            model="qwen3:8b",
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
            model="ollama_chat/qwen3:8b",
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
            model="ollama_chat/qwen3:8b",
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
    """Modified streaming response using new two-agent architecture"""
    try:
        # Step 1: Execute tools with advisor (silently)
        yield "🤖 Thinking...\n\n"
        tool_results = execute_tools_with_advisor(messages)

        # Step 2: Stream final response
        yield from get_final_response_stream(messages, tool_results)

    except Exception as e:
        yield f"❌ Error: {str(e)}"


def get_response(messages):
    """Modified complete response using new two-agent architecture"""
    try:
        # Step 1: Execute tools with advisor
        tool_results = execute_tools_with_advisor(messages)

        # Step 2: Generate final response
        response = get_final_response(messages, tool_results)

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
