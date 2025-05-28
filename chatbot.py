import streamlit as st
import json
import os
import uuid
from datetime import datetime
from chatbot_utils import (
    get_response,  # Changed from get_response_stream
    test_model_connection,
    parse_thinking_response,
    generate_thread_title,
    save_thread,
    load_thread,
    get_all_threads,
    delete_thread,
    create_threads_directory,
)

# Page configuration
st.set_page_config(
    page_title="FoodieSpot Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Create threads directory if it doesn't exist
create_threads_directory()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "current_thread_title" not in st.session_state:
    st.session_state.current_thread_title = "New Chat"
if "threads_list" not in st.session_state:
    st.session_state.threads_list = get_all_threads()


def create_new_thread():
    """Create a new thread"""
    # Save current thread if it has messages
    if st.session_state.messages and st.session_state.current_thread_id:
        save_thread(
            st.session_state.current_thread_id,
            st.session_state.current_thread_title,
            st.session_state.messages,
        )

    # Create new thread
    new_thread_id = str(uuid.uuid4())
    st.session_state.current_thread_id = new_thread_id
    st.session_state.current_thread_title = "New Chat"
    st.session_state.messages = []
    st.session_state.threads_list = get_all_threads()


def switch_to_thread(thread_id, thread_title):
    """Switch to an existing thread"""
    # Save current thread if it has messages
    if st.session_state.messages and st.session_state.current_thread_id:
        save_thread(
            st.session_state.current_thread_id,
            st.session_state.current_thread_title,
            st.session_state.messages,
        )

    # Load selected thread
    thread_data = load_thread(thread_id)
    if thread_data:
        st.session_state.current_thread_id = thread_id
        st.session_state.current_thread_title = thread_data.get(
            "title", "Untitled Chat"
        )
        st.session_state.messages = thread_data.get("messages", [])
    else:
        st.error("Failed to load thread")


def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.messages:
        chat_data = {
            "thread_id": st.session_state.current_thread_id,
            "title": st.session_state.current_thread_title,
            "export_time": datetime.now().isoformat(),
            "messages": st.session_state.messages,
        }
        return json.dumps(chat_data, indent=2)
    return None


def display_thinking_section(thinking_content):
    """Display thinking section in an expander"""
    if thinking_content:
        with st.expander("🧠 AI thoughts (click to see)", expanded=False):
            st.markdown(f"**Thought process:**\n\n{thinking_content}")


# Sidebar
with st.sidebar:
    st.header("🤖 FoodieSpot Chatbot")

    # Thread management
    st.subheader("💬 Threads")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("➕ New Thread", type="primary", use_container_width=True):
            create_new_thread()
            st.rerun()

    with col2:
        if st.button("🔄", help="Refresh threads list"):
            st.session_state.threads_list = get_all_threads()
            st.rerun()

    # Current thread info
    if st.session_state.current_thread_id:
        st.markdown(f"**Current:** {st.session_state.current_thread_title}")
        st.caption(f"ID: {st.session_state.current_thread_id[:8]}...")
    else:
        st.markdown("**Current:** No active thread")
        st.caption("Click 'New Thread' to start")

    # Thread list
    if st.session_state.threads_list:
        st.markdown("**Recent Threads:**")
        for thread in st.session_state.threads_list[:10]:  # Show last 10 threads
            thread_id = thread["id"]
            thread_title = thread["title"]
            created_at = thread.get("created_at", "Unknown")

            # Create a container for each thread
            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Thread button
                    if st.button(
                        f"📝 {thread_title}",
                        key=f"thread_{thread_id}",
                        help=f"Created: {created_at}",
                        use_container_width=True,
                        type=(
                            "secondary"
                            if thread_id != st.session_state.current_thread_id
                            else "primary"
                        ),
                    ):
                        switch_to_thread(thread_id, thread_title)
                        st.rerun()

                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"delete_{thread_id}", help="Delete thread"):
                        if delete_thread(thread_id):
                            if thread_id == st.session_state.current_thread_id:
                                create_new_thread()
                            st.session_state.threads_list = get_all_threads()
                            st.success("Thread deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete thread")

    st.divider()

    # Model status check
    st.subheader("Model Status")
    if st.button("Check Connection", type="secondary"):
        with st.spinner("Testing connection..."):
            is_available, status_msg = test_model_connection()
            if is_available:
                st.success(status_msg)
            else:
                st.error(status_msg)

    st.divider()

    # Chat controls
    st.subheader("Chat Controls")

    if st.button("Clear Current Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.current_thread_title = "New Chat"
        st.rerun()

    # Export functionality
    if st.session_state.messages:
        st.subheader("Export Chat")
        chat_json = export_chat_history()
        if chat_json:
            # Safe filename generation
            thread_id_short = (
                st.session_state.current_thread_id[:8]
                if st.session_state.current_thread_id
                else "unknown"
            )
            filename = f"chat_{thread_id_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="Download Chat History",
                data=chat_json,
                file_name=filename,
                mime="application/json",
            )

    st.divider()

    # Statistics
    st.subheader("Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Total Threads", len(st.session_state.threads_list))
    if st.session_state.messages:
        user_messages = len(
            [m for m in st.session_state.messages if m["role"] == "user"]
        )
        assistant_messages = len(
            [m for m in st.session_state.messages if m["role"] == "assistant"]
        )
        st.metric("User Messages", user_messages)
        st.metric("Assistant Messages", assistant_messages)

    st.divider()

    # Info section
    st.subheader("ℹ️ About")
    st.markdown(
        """
    This chatbot uses Llama3 Groq Tool Use 8B model running locally via Ollama.
    
    **Features:**
    - Thread-based conversations
    - Automatic title generation
    - Complete responses (non-streaming)
    - Thought process visibility
    - Chat history export
    - Local AI processing
    """
    )

# Main chat interface
st.title("💬 FoodieSpot Chat")

# Create new thread if none exists
if not st.session_state.current_thread_id:
    create_new_thread()

st.caption(f"Thread: {st.session_state.current_thread_title}")

# Quick start instructions in expander
with st.expander("📋 Quick Start Guide"):
    st.markdown(
        """
    **Prerequisites:**
    1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
    2. Start Ollama: `ollama serve`
    3. Pull Deepseek model: `ollama pull llama3-groq-tool-use:8b`
    4. Install dependencies: `pip install streamlit litellm`
    
    **Features:**
    - **Threads**: Each conversation is saved as a separate thread
    - **Auto Titles**: Thread titles are automatically generated from the first message
    - **Persistent Storage**: All threads are saved locally in the `threads/` directory
    - **Thread Management**: Create, switch between, and delete threads easily
    - **Non-streaming**: Complete responses are loaded at once
    
    **Usage:**
    - Click "➕ New Thread" to start a fresh conversation
    - Click on any thread in the sidebar to switch to it
    - Thread titles are generated automatically after the first exchange
    - Use 🗑️ to delete unwanted threads
    - Export individual thread conversations as JSON
    
    **Files needed:**
    - `chatbot.py` (main application)
    - `chatbot_utils.py` (utility functions)
    - `threads/` directory (created automatically)
    """
    )

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display thinking section if it exists
            if "thinking" in message and message["thinking"]:
                display_thinking_section(message["thinking"])
        # Display the main content
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            # Show loading indicator
            with st.spinner("AI is thinking..."):
                # Get complete response (non-streaming)
                full_response = get_response(st.session_state.messages)

            if full_response.startswith("Error:"):
                st.error(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                # Parse the complete response
                thinking_content, main_response = parse_thinking_response(full_response)

                final_content = main_response if main_response else full_response

                # Show thinking section if exists
                if thinking_content:
                    display_thinking_section(thinking_content)

                # Display final response
                st.markdown(final_content)

                # Add to chat history
                message_data = {"role": "assistant", "content": final_content}
                if thinking_content:
                    message_data["thinking"] = thinking_content

                st.session_state.messages.append(message_data)

                # Generate thread title if this is the first exchange and title is still "New Chat"
                if (
                    st.session_state.current_thread_title == "New Chat"
                    and len(
                        [m for m in st.session_state.messages if m["role"] == "user"]
                    )
                    == 1
                ):
                    # Generate title in background
                    new_title = generate_thread_title(prompt, final_content)
                    if new_title and new_title != "New Chat":
                        st.session_state.current_thread_title = new_title

                # Save the thread
                save_thread(
                    st.session_state.current_thread_id,
                    st.session_state.current_thread_title,
                    st.session_state.messages,
                )

                # Update threads list
                st.session_state.threads_list = get_all_threads()

                # Force a rerun to refresh the display
                st.rerun()

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
            st.rerun()

# Footer
st.divider()
st.caption(
    "🔧 Make sure Ollama is running with `ollama serve` and the Deepseek model is available with `ollama pull llama3-groq-tool-use:8b`"
)
