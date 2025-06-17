import streamlit as st
import json
import uuid
from datetime import datetime
from chatbot_utils import (
    get_response_stream,  # New streaming function
    get_response,  # Keep non-streaming for compatibility
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

# Shimmer CSS
shimmer_css = """
<style>
.shimmer-text {
    background: linear-gradient(90deg, #666 25%, #888 50%, #666 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    color: transparent;
    background-clip: text;
    -webkit-background-clip: text;
    font-size: 16px;
    font-weight: 500;
    padding: 8px 0;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
</style>
"""

st.markdown(shimmer_css, unsafe_allow_html=True)

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
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True


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

    # Streaming toggle
    st.subheader("⚙️ Settings")
    streaming_enabled = st.toggle(
        "Enable Streaming",
        value=st.session_state.streaming_enabled,
        help="Stream responses in real-time vs complete responses",
    )
    if streaming_enabled != st.session_state.streaming_enabled:
        st.session_state.streaming_enabled = streaming_enabled
        st.rerun()

    st.divider()

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
        f"""
    This chatbot uses Qwen3 8B model running locally via Ollama.
    
    **Features:**
    - {'🔥 **Streaming responses**' if st.session_state.streaming_enabled else '📄 Complete responses'}
    - Thread-based conversations
    - Automatic title generation
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

# Display streaming status
streaming_status = (
    "🔥 Streaming Mode" if st.session_state.streaming_enabled else "📄 Complete Mode"
)
st.caption(f"Thread: {st.session_state.current_thread_title} | {streaming_status}")

# Quick start instructions in expander
with st.expander("📋 Quick Start Guide"):
    st.markdown(
        """
    **Prerequisites:**
    1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
    2. Start Ollama: `ollama serve`
    3. Pull Qwen3 model: `ollama pull qwen3:8b-fp16`
    4. Install dependencies: `pip install streamlit litellm ollama python-dotenv`
    
    **Features:**
    - **Streaming Responses**: Toggle between streaming and complete responses
    - **Threads**: Each conversation is saved as a separate thread
    - **Auto Titles**: Thread titles are automatically generated from the first message
    - **Persistent Storage**: All threads are saved locally in the `threads/` directory
    - **Thread Management**: Create, switch between, and delete threads
    """
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Parse thinking content if it exists
            content = message["content"]
            thinking_content, main_response = parse_thinking_response(content)

            # Display thinking section if available
            display_thinking_section(thinking_content)

            # Display main response
            st.markdown(main_response)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        if st.session_state.streaming_enabled:
            # Streaming response
            response_placeholder = st.empty()
            thinking_placeholder = st.empty()

            full_response = ""
            tool_output_buffer = ""
            current_section = "response"
            shimmer_active = False

            try:
                for chunk in get_response_stream(st.session_state.messages):
                    if chunk:
                        # Check for shimmer status messages
                        if chunk in [
                            "🤖 Thinking...\n\n",
                        ]:
                            shimmer_active = True
                            # Extract message without emoji and extra newlines
                            shimmer_text = (
                                chunk.replace("🤖", "").replace("✨", "").strip()
                            )

                            with response_placeholder.container():
                                st.markdown(
                                    f'<div class="shimmer-text">{shimmer_text}</div>',
                                    unsafe_allow_html=True,
                                )
                            continue

                        # Check if actual response is starting
                        elif chunk.startswith("✨ **Response:**"):
                            shimmer_active = False
                            current_section = "response"
                            full_response = ""
                            # Clear the shimmer and don't add this marker to response
                            response_placeholder.empty()
                            continue

                        # Check if this is a tool execution indicator
                        elif (
                            chunk.startswith("🔧")
                            or chunk.startswith("📊")
                            or chunk.startswith("❌")
                        ):
                            # If shimmer was active, clear it
                            if shimmer_active:
                                shimmer_active = False
                                response_placeholder.empty()

                            # Tool execution output
                            tool_output_buffer += chunk
                            current_section = "tools"

                            # Update thinking section with tool output
                            with thinking_placeholder.container():
                                with st.expander("🛠️ Tool Execution", expanded=True):
                                    st.markdown(tool_output_buffer)

                        else:
                            # Regular response content
                            if current_section == "response":
                                # Clear shimmer if it was active
                                if shimmer_active:
                                    shimmer_active = False
                                    response_placeholder.empty()

                                full_response += chunk

                                # Parse and display response with thinking
                                thinking_content, main_response = (
                                    parse_thinking_response(full_response)
                                )

                                # Update the response display
                                with response_placeholder.container():
                                    if thinking_content:
                                        display_thinking_section(thinking_content)
                                    st.markdown(main_response)

            except Exception as e:
                # Clear shimmer on error
                if shimmer_active:
                    response_placeholder.empty()

                st.error(f"Streaming error: {str(e)}")
                # Fallback to non-streaming
                full_response = get_response(st.session_state.messages)
                thinking_content, main_response = parse_thinking_response(full_response)

                if thinking_content:
                    display_thinking_section(thinking_content)
                st.markdown(main_response)

        else:
            # Non-streaming response
            with st.spinner("Generating response..."):
                full_response = get_response(st.session_state.messages)

            # Parse thinking content
            thinking_content, main_response = parse_thinking_response(full_response)

            # Display thinking section if available
            display_thinking_section(thinking_content)

            # Display main response
            st.markdown(main_response)

    # Add assistant response to chat history
    final_response = (
        full_response if "full_response" in locals() else "Response generated"
    )
    st.session_state.messages.append({"role": "assistant", "content": final_response})

    # Generate thread title if this is the first exchange
    if (
        len(st.session_state.messages) == 2
    ):  # First user message + first assistant response
        try:
            user_msg = st.session_state.messages[0]["content"]
            assistant_msg = st.session_state.messages[1]["content"]

            # Generate title in the background
            new_title = generate_thread_title(user_msg, assistant_msg)
            if new_title and new_title != "New Chat":
                st.session_state.current_thread_title = new_title
                # Save immediately after title generation
                save_thread(
                    st.session_state.current_thread_id,
                    st.session_state.current_thread_title,
                    st.session_state.messages,
                )
                # Refresh threads list
                st.session_state.threads_list = get_all_threads()
                # Rerun to update the display
                st.rerun()

        except Exception as e:
            print(f"Error generating title: {e}")
            # Keep default title if generation fails

    # Always save thread after each exchange
    if st.session_state.current_thread_id:
        save_thread(
            st.session_state.current_thread_id,
            st.session_state.current_thread_title,
            st.session_state.messages,
        )
        # Refresh threads list to show updated info
        st.session_state.threads_list = get_all_threads()

# Auto-scroll behavior note
st.caption("💡 The chat will automatically scroll to show new messages")
