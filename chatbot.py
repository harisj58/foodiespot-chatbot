import streamlit as st
import json
from datetime import datetime
from chatbot_utils import (
    get_response_stream,
    test_model_connection,
    parse_thinking_response,
)

# Page configuration
st.set_page_config(
    page_title="Deepseek Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.messages:
        chat_data = {
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
    st.header("🤖 Deepseek Chatbot")

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

    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # Export functionality
    if st.session_state.messages:
        st.subheader("Export Chat")
        chat_json = export_chat_history()
        if chat_json:
            st.download_button(
                label="Download Chat History",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    st.divider()

    # Statistics
    st.subheader("Session Stats")
    st.metric("Messages", len(st.session_state.messages))
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
    This chatbot uses Deepseek R1 8B model running locally via Ollama.
    
    **Features:**
    - Real-time streaming responses
    - Thought process visibility
    - Chat history export
    - Local AI processing
    """
    )

# Main chat interface
st.title("💬 Deepseek Chat")
st.caption("Powered by Deepseek R1 8B via Ollama")

# Quick start instructions in expander
with st.expander("📋 Quick Start Guide"):
    st.markdown(
        """
    **Prerequisites:**
    1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
    2. Start Ollama: `ollama serve`
    3. Pull Deepseek model: `ollama pull deepseek-r1:8b`
    4. Install dependencies: `pip install streamlit litellm`
    
    **Usage:**
    - Type your message in the chat input below
    - Responses are streamed in real-time
    - When the AI thinks through problems, you'll see a "🧠 AI is thinking..." indicator
    - Click on the thinking expander to see the AI's thought process
    - Export your chat history as JSON when needed
    
    **Files needed:**
    - `chatbot.py` (main application)
    - `chatbot_utils.py` (utility functions)
    
    **Troubleshooting:**
    - If you get connection errors, make sure Ollama is running on port 11434
    - Use "Check Connection" button to test model availability
    - Clear chat history if you encounter persistent issues
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
            # Create placeholders for thinking indicator and response
            thinking_indicator = st.empty()
            response_placeholder = st.empty()

            full_response = ""
            thinking_detected = False
            thinking_complete = False

            # Stream the response
            for chunk_content, current_full_response in get_response_stream(
                st.session_state.messages
            ):
                if chunk_content.startswith("Error:"):
                    response_placeholder.error(chunk_content)
                    break

                full_response = current_full_response

                # Check if we're in thinking mode
                if "<think>" in full_response and not thinking_detected:
                    thinking_detected = True
                    thinking_indicator.markdown("🧠 *AI is thinking...*")

                # Check if thinking is complete
                if (
                    thinking_detected
                    and "</think>" in full_response
                    and not thinking_complete
                ):
                    thinking_complete = True
                    thinking_indicator.empty()

                # Display the response content
                if thinking_complete:
                    # Parse and show only the main response part
                    _, main_response = parse_thinking_response(full_response)
                    if main_response:
                        response_placeholder.markdown(main_response + "▌")
                elif not thinking_detected:
                    # Normal streaming without thinking
                    response_placeholder.markdown(full_response + "▌")

            # Final processing after streaming is complete
            if not full_response.startswith("Error:") and full_response:
                # Parse the complete response
                thinking_content, main_response = parse_thinking_response(full_response)

                # Clear the thinking indicator
                thinking_indicator.empty()

                final_content = main_response if main_response else full_response

                # Show final thinking section if exists
                if thinking_content:
                    display_thinking_section(thinking_content)

                # Display final response without streaming cursor
                response_placeholder.markdown(final_content)

                # Add to chat history (this will be displayed next time)
                message_data = {"role": "assistant", "content": final_content}
                if thinking_content:
                    message_data["thinking"] = thinking_content

                st.session_state.messages.append(message_data)

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
    "🔧 Make sure Ollama is running with `ollama serve` and the Deepseek model is available with `ollama pull deepseek-r1:8b`"
)
