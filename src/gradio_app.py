"""
Banking Database Analysis Tool - Professional read-only database analysis with Excel reporting.
Converts natural language queries into SQL and generates comprehensive Excel reports.
"""

import os
import time
import threading
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
import pandas as pd
from loguru import logger

# Handle both module and direct execution
try:
    from .config import settings
    from .database import db_manager
    from .excel_export import excel_exporter
    from .llm_service import llm_service
except ImportError:
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from src.config import settings
    from src.database import db_manager
    from src.excel_export import excel_exporter
    from src.llm_service import llm_service

# Global processing state management
processing_lock = threading.Lock()
stop_requested = threading.Event()


def get_sample_queries():
    """Get expert banking analysis query suggestions showcasing 2023-2025 capabilities."""
    from src.llm_service import llm_service
    return llm_service.suggest_sample_queries()


def check_ollama_service():
    """Check if Ollama service is running and available."""
    try:
        import requests
        from src.config import settings

        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            tags_data = response.json()
            models = [model["name"] for model in tags_data.get("models", [])]
            return True, models
        else:
            return False, []
    except Exception as e:
        logger.warning(f"Ollama service check failed: {e}")
        return False, []


def process_query(user_input: str, chat_messages: list) -> Tuple[list, str, Optional[str], bool]:
    """Process user query and return updated chat messages, empty input, download file, and show_suggestions."""

    if not user_input.strip():
        return chat_messages, "", None, len(chat_messages) == 0

    logger.info(f"Processing query: {user_input}")

    # Add user message
    chat_messages.append({"role": "user", "content": user_input, "type": "text"})

    try:
        # Initial status message - will be updated as we progress
        chat_messages.append({"role": "assistant", "content": "Analyzing banking query...", "type": "status"})
        yield chat_messages, "", None, False
        time.sleep(0.3)

        # Check if stop was requested
        if stop_requested.is_set():
            chat_messages[-1] = {"role": "assistant", "content": "Processing stopped by user.", "type": "text"}
            yield chat_messages, "", None, False
            return

        # Generate SQL with better error handling
        chat_messages[-1]["content"] = "Generating SQL query..."
        yield chat_messages, "", None, False

        logger.info("Calling LLM service to generate SQL")
        try:
            llm_result = llm_service.generate_sql(user_input)
            logger.debug(f"LLM result: {llm_result}")

            # Check if stop was requested after LLM call
            if stop_requested.is_set():
                chat_messages[-1] = {"role": "assistant", "content": "Processing stopped by user.", "type": "text"}
                yield chat_messages, "", None, False
                return

        except Exception as llm_error:
            logger.error(f"LLM service error: {llm_error}", exc_info=True)
            chat_messages[-1] = {"role": "assistant", "content": f"Error connecting to Ollama: {str(llm_error)}. Please ensure Ollama is running with the qwen2.5:14b model.", "type": "text"}
            yield chat_messages, "", None, False
            return

        if not llm_result.get('success', False):
            error_msg = llm_result.get('error', 'Failed to generate SQL query')
            chat_messages[-1] = {"role": "assistant", "content": f"SQL Generation Failed: {error_msg}", "type": "text"}
            logger.error(f"LLM SQL generation failed: {error_msg}")
            yield chat_messages, "", None, False
            return

        sql_query = llm_result.get('sql_query', '')

        # Show generated SQL
        chat_messages.append({
            "role": "assistant",
            "content": f"**SQL Generated**\n\n```sql\n{sql_query}\n```",
            "type": "text"
        })
        yield chat_messages, "", None, False

        logger.info(f"Generated safe SQL: {sql_query}")

        # Execute query with better error handling
        chat_messages.append({"role": "assistant", "content": "Executing query...", "type": "status"})
        yield chat_messages, "", None, False

        # Check if stop was requested before database execution
        if stop_requested.is_set():
            chat_messages[-1] = {"role": "assistant", "content": "Processing stopped by user.", "type": "text"}
            yield chat_messages, "", None, False
            return

        logger.info("Executing database query")
        try:
            results_df = db_manager.execute_query(sql_query)
            logger.debug(f"Query execution result: {type(results_df)}, empty: {results_df is None or (hasattr(results_df, 'empty') and results_df.empty)}")

            # Check if stop was requested after database execution
            if stop_requested.is_set():
                chat_messages[-1] = {"role": "assistant", "content": "Processing stopped by user.", "type": "text"}
                yield chat_messages, "", None, False
                return

        except Exception as db_error:
            logger.error(f"Database query error: {db_error}", exc_info=True)
            chat_messages[-1] = {"role": "assistant", "content": f"Database error: {str(db_error)}", "type": "text"}
            yield chat_messages, "", None, False
            return

        if results_df is None or len(results_df) == 0:
            chat_messages[-1] = {"role": "assistant", "content": "Query executed successfully, but no results found. Try a different query or check if data exists.", "type": "text"}
            logger.warning("Query returned no results")
            yield chat_messages, "", None, False
            return

        logger.info(f"Query returned {len(results_df)} rows")

        # Create Excel report with better error handling
        chat_messages[-1]["content"] = "Creating Excel report..."
        yield chat_messages, "", None, False

        # Check if stop was requested before Excel export
        if stop_requested.is_set():
            chat_messages[-1] = {"role": "assistant", "content": "Processing stopped by user.", "type": "text"}
            yield chat_messages, "", None, False
            return

        logger.info("Generating Excel export")
        try:
            # Use export_query_results which handles list data and None filename
            filename = excel_exporter.export_query_results(
                results_df,
                {
                    'sql_query': sql_query,
                    'query_description': user_input
                },
                filename=None  # Let the smart filename generator create a descriptive name
            )
            logger.info(f"Excel file created: {filename}")
        except Exception as excel_error:
            logger.error(f"Excel export error: {excel_error}", exc_info=True)
            chat_messages[-1] = {"role": "assistant", "content": f"Error creating Excel report: {str(excel_error)}", "type": "text"}
            yield chat_messages, "", None, False
            return

        # Final message with download
        chat_messages[-1] = {
            "role": "assistant",
            "content": f"**Banking Analysis Complete!**\n\nYour Excel report is ready with:\n• **{len(results_df)}** rows of data\n• Professional charts and formatting\n• Query details and results",
            "type": "download",
            "filename": filename
        }

        yield chat_messages, "", filename, False

    except Exception as e:
        error_msg = f"Unexpected error occurred: {str(e)}"
        logger.error(f"Query processing error: {e}", exc_info=True)

        # Ensure we have a status message to update
        if not chat_messages or chat_messages[-1].get("role") != "assistant":
            chat_messages.append({"role": "assistant", "content": "", "type": "status"})

        chat_messages[-1] = {"role": "assistant", "content": f"{error_msg}", "type": "text"}
        yield chat_messages, "", None, False




def render_chat_message(message):
    """Render a single chat message with proper styling."""

    # Handle status messages
    if message.get("type") == "status":
        html = f'''
        <div class="status-message">
            {message["content"]}
        </div>
        '''
        return html

    # Handle download messages as styled buttons
    if message.get("type") == "download" and message.get("filename"):
        filename = message["filename"]
        if os.path.exists(filename):
            html = f'''
            <div class="assistant-message" style="
                background: #0f0f0f !important;
                border: 1px solid #2a2a2a !important;
                color: #e5e5e5 !important;
                padding: 1rem 1.25rem !important;
                border-radius: 14px !important;
                margin: 0.5rem auto 0.5rem 0 !important;
                width: fit-content !important;
                max-width: 70% !important;
                font-size: 16px !important;
                font-weight: 400 !important;
                line-height: 1.4 !important;
                text-align: left !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            ">
                <div style="color: #4ade80; font-size: 16px; font-weight: 500; margin-bottom: 0.5rem;">
                    Excel Report Generated Successfully
                </div>
                <div style="color: #a0a0a0; font-size: 14px;">
                    File: {os.path.basename(filename)}<br>
                    Download link available below
                </div>
            </div>'''
            return html

    # Regular chat messages
    role_class = "user-message" if message["role"] == "user" else "assistant-message"

    html = f'''
    <div class="{role_class}">
        {message["content"]}
    </div>
    '''

    return html


def render_chat_history(messages):
    """Render only the last user message and assistant response (no chat history)."""
    if not messages:
        return ""

    html = ""
    # Show the last user message and last assistant message
    user_message = None
    assistant_message = None

    # Find the last user and assistant messages
    for message in reversed(messages):
        if message["role"] == "user" and user_message is None:
            user_message = message
        elif message["role"] == "assistant" and assistant_message is None:
            assistant_message = message

        # Stop once we have both
        if user_message and assistant_message:
            break

    # Render user message first, then assistant
    if user_message:
        html += render_chat_message(user_message)
    if assistant_message:
        html += render_chat_message(assistant_message)

    return html


def create_interface():
    """Create and configure the Gradio interface."""

    # Custom CSS for clean dark chat interface
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Remove all Gradio branding and unnecessary elements */
    .gradio-container .footer,
    .gradio-container footer,
    .gr-button[data-testid="use-via-api"],
    .gr-button[data-testid="built-with-gradio"],
    .built-with-gradio,
    .use-via-api,
    footer,
    .footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* Full dark theme */
    html, body {
        background-color: #0a0a0a !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .gradio-container, .gr-box, .gr-form, .gr-panel, .contain, .main {
        background-color: #0a0a0a !important;
        color: #e5e5e5 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        align-items: stretch !important;
        justify-content: flex-start !important;
    }

    /* Content area - simple layout starting from top */
    .content-area {
        max-width: 900px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 2rem 3rem !important;
        box-sizing: border-box !important;
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }

    /* Remove all display overrides - let Gradio handle visibility completely */
    .content-area > div {
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }

    /* Force hidden elements to stay hidden */
    .content-area [style*="display: none"],
    .content-area > div[style*="display: none"],
    .content-area .gr-column[style*="display: none"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* COMPLETELY REDESIGN Gradio file component to look like chat bubble */
    .gradio-container .gr-file,
    .gradio-container .gr-file-container,
    .gradio-container [data-testid="file"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0.5rem auto 0.5rem 0 !important;
        width: fit-content !important;
        max-width: 70% !important;
        min-width: 300px !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Main container styling - exactly like assistant message */
    .gradio-container .gr-file > div,
    .gradio-container .gr-file .file-wrap,
    .gradio-container .gr-file .wrap {
        background: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide ALL upload elements completely */
    .gradio-container .gr-file .upload-container,
    .gradio-container .gr-file input[type="file"],
    .gradio-container .gr-file .file-preview,
    .gradio-container .gr-file .upload-text,
    .gradio-container .gr-file .or-text,
    .gradio-container .gr-file .upload-button,
    .gradio-container .gr-file .drag-text,
    .gradio-container .gr-file label[for],
    .gradio-container .gr-file .file-name {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Style download links to look like inline buttons */
    .gradio-container .gr-file a,
    .gradio-container .gr-file .download-link {
        background: #1a1a1a !important;
        color: #4ade80 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        text-decoration: none !important;
        border: 1px solid #2a2a2a !important;
        display: inline-block !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
        margin-top: 0.5rem !important;
    }

    .gradio-container .gr-file a:hover,
    .gradio-container .gr-file .download-link:hover {
        background: #262626 !important;
        border-color: #404040 !important;
    }

    /* Force proper sizing constraints */
    .gradio-container .gr-file * {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Headers - much larger text */
    h1, h2, h3 {
        color: #e5e5e5 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500 !important;
        font-size: 28px !important;
        margin-bottom: 1.25rem !important;
        letter-spacing: -0.02em !important;
    }

    /* Chat container - simple layout, no scrolling needed */
    .chat-container {
        padding: 1rem 0 !important;
        margin-bottom: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 0.5rem !important;
    }

    /* Input area - fixed at bottom with same width as content */
    .input-area {
        position: fixed !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 100% !important;
        max-width: 900px !important;
        z-index: 1001 !important;
        background-color: #0a0a0a !important;
        padding: 1rem 3rem !important;
        box-sizing: border-box !important;
    }

    /* Make textbox container relative for button positioning */
    .input-area .gr-textbox {
        position: relative !important;
    }

    /* Chat content has proper spacing without extra padding */

    /* Textarea styling - darker with proper borders, non-resizable */
    .gr-textbox {
        position: relative !important;
    }

    .gr-textbox > div,
    .gr-textbox > div > div,
    .gr-textbox > div > div > div {
        background: transparent !important;
        border: none !important;
    }

    .gr-textbox textarea,
    .gr-textbox > div textarea,
    .gr-textbox > div > div textarea,
    textarea[data-testid="textbox"] {
        background-color: #0f0f0f !important;
        border: 2px solid #2a2a2a !important;
        border-radius: 12px !important;
        color: #e5e5e5 !important;
        font-size: 20px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 400 !important;
        padding: 0.75rem 6rem 0.75rem 1.5rem !important;
        height: 88px !important;
        min-height: 88px !important;
        max-height: 128px !important;
        resize: none !important;
        -webkit-resize: none !important;
        -moz-resize: none !important;
        line-height: 1.4 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        letter-spacing: -0.01em !important;
        -webkit-user-select: text !important;
        -moz-user-select: text !important;
        user-select: text !important;
        overflow: auto !important;
        transition: border-color 0.2s ease !important;
    }

    .gr-textbox textarea:hover:not(:focus),
    .gr-textbox > div textarea:hover:not(:focus),
    textarea[data-testid="textbox"]:hover:not(:focus) {
        border-color: #2563eb !important;
    }

    .gr-textbox textarea:focus,
    .gr-textbox > div textarea:focus,
    textarea[data-testid="textbox"]:focus {
        border-color: #2a2a2a !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .gr-textbox textarea::placeholder,
    .gr-textbox > div textarea::placeholder,
    textarea[data-testid="textbox"]::placeholder {
        color: #666666 !important;
        font-size: 20px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Send button - darker colors, positioned bottom right inside input */
    #send_btn,
    .gr-button[elem_id="send_btn"] {
        position: absolute !important;
        right: 58px !important;
        top: 61% !important;
        transform: translateY(-50%) !important;
        z-index: 1000 !important;
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
        width: auto !important;
        height: 36px !important;
        min-width: 80px !important;
        max-width: 80px !important;
        padding: 0 8px !important;
        margin: 0 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #cccccc !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        box-sizing: border-box !important;
    }

    #send_btn:hover:not([disabled]) {
        background-color: #262626 !important;
        border-color: #555555 !important;
    }

    #send_btn[disabled],
    #send_btn:disabled {
        background-color: #0a0a0a !important;
        border-color: #1a1a1a !important;
        color: #666666 !important;
        cursor: not-allowed !important;
        opacity: 0.5 !important;
    }

    /* Suggestion buttons - aligned to bottom like chat */
    .suggestion-btn {
        background-color: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #aaaaaa !important;
        font-size: 18px !important;
        font-weight: 400 !important;
        text-align: left !important;
        margin-bottom: 0.25rem !important;
        width: fit-content !important;
        min-width: auto !important;
        padding: 1rem 1.25rem !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.01em !important;
        line-height: 1.3 !important;
        display: block !important;
    }

    /* Let flexbox handle the positioning - no display overrides */
    .content-area > div {
        margin-bottom: 0 !important;
    }

    /* Very aggressive hiding for invisible sections */
    .gradio-container [style*="display: none"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    .suggestion-btn:hover {
        border-color: #2563eb !important;
        background-color: #1a1a1a !important;
    }

    /* Chat message styling - much larger text */
    .user-message {
        background: #2563eb !important;
        color: #f5f5f5 !important;
        padding: 1rem 1.25rem !important;
        border-radius: 14px !important;
        margin: 0.5rem 0 0.5rem auto !important;
        width: fit-content !important;
        max-width: 70% !important;
        font-size: 20px !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
        text-align: left !important;
        word-wrap: break-word !important;
        letter-spacing: -0.01em !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .assistant-message {
        background: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        color: #e5e5e5 !important;
        padding: 1rem 1.25rem !important;
        border-radius: 14px !important;
        margin: 0.5rem auto 0.5rem 0 !important;
        width: fit-content !important;
        max-width: 70% !important;
        font-size: 20px !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
        text-align: left !important;
        word-wrap: break-word !important;
        letter-spacing: -0.01em !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Status message bubble - changes content */
    .status-message {
        background: #1a1a1a !important;
        border: 1px solid #333333 !important;
        color: #999999 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 12px !important;
        margin: 0.5rem auto 0.5rem 0 !important;
        width: fit-content !important;
        max-width: 60% !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        font-style: italic !important;
        line-height: 1.4 !important;
        text-align: left !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Remove all outlines */
    * {
        outline: none !important;
        -webkit-tap-highlight-color: transparent !important;
    }

    /* File download styling - much larger text */
    .gr-file {
        background-color: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 12px !important;
        color: #e5e5e5 !important;
        font-size: 20px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Clean up layout */
    .gr-column {
        gap: 0 !important;
    }

    .gr-group {
        background: transparent !important;
        border: none !important;
    }

    /* Make the interface feel more like a chat app */
    .gr-row {
        gap: 0 !important;
    }

    .gr-form {
        background: transparent !important;
    }
    """

    with gr.Blocks(css=css, title="Banking Database Analysis Tool", theme=gr.themes.Base()) as demo:
        # State variables
        chat_messages = gr.State([])

        # Main content area - combines chat history and suggestions
        with gr.Column(elem_classes=["content-area"]):
            # Chat history display
            with gr.Column(visible=False) as chat_section:
                chat_display = gr.HTML("", elem_classes=["chat-container"])

            # Suggestions section (visible when no messages)
            with gr.Column(visible=True) as suggestions_section:
                suggestions = get_sample_queries()
                suggestion_buttons = []

                for suggestion in suggestions:
                    btn = gr.Button(
                        suggestion,
                        elem_classes=["suggestion-btn"],
                        size="sm"
                    )
                    suggestion_buttons.append(btn)

        # Input area - fixed at bottom
        with gr.Column(elem_classes=["input-area"]):
            msg = gr.Textbox(
                placeholder="Enter your banking analysis query... (Read-only database analysis)",
                show_label=False,
                container=False,
                lines=2,
                max_lines=4,
                autofocus=True,
                elem_id="main_input"
            )
            submit_btn = gr.Button("Analyze", size="sm", elem_id="send_btn", interactive=True)

        # Download area
        download_file = gr.File(label="Download Excel Report", visible=False)

        # Event handlers
        def handle_message(message, messages):
            """Handle message submission and update interface."""
            if not message.strip():
                show_suggestions = len(messages) == 0
                return (
                    gr.update(visible=not show_suggestions),  # chat_section
                    gr.update(visible=show_suggestions),      # suggestions_section
                    render_chat_history(messages),           # chat_display
                    "",                                       # msg
                    gr.update(visible=False),                # download_file
                    messages,                                # chat_messages
                    gr.update(value="Send", interactive=True)  # submit_btn
                )

            # Process the query - get the final result from generator
            final_result = None
            try:
                for result in process_query(message, messages):
                    final_result = result
                    # Yield intermediate results for progress updates
                    updated_messages, empty_input, file_path, show_sugg = result
                    yield (
                        gr.update(visible=True),                    # chat_section
                        gr.update(visible=False),                   # suggestions_section
                        render_chat_history(updated_messages),     # chat_display
                        empty_input,                               # msg
                        gr.update(visible=bool(file_path), value=file_path if file_path else None),  # download_file
                        updated_messages,                          # chat_messages
                        gr.update(value="Stop", interactive=True)    # submit_btn - show Stop during processing
                    )

                # Return final result
                if final_result:
                    updated_messages, empty_input, file_path, show_sugg = final_result
                    return (
                        gr.update(visible=True),                    # chat_section
                        gr.update(visible=False),                   # suggestions_section
                        render_chat_history(updated_messages),     # chat_display
                        empty_input,                               # msg
                        gr.update(visible=bool(file_path), value=file_path if file_path else None),  # download_file
                        updated_messages,                          # chat_messages
                        gr.update(value="Send", interactive=True)    # submit_btn - back to Send when complete
                    )
            except Exception as e:
                logger.error(f"Error in handle_message: {e}", exc_info=True)
                error_messages = messages.copy()
                error_messages.append({"role": "assistant", "content": f"Error processing request: {str(e)}", "type": "text"})
                return (
                    gr.update(visible=True),                    # chat_section
                    gr.update(visible=False),                   # suggestions_section
                    render_chat_history(error_messages),       # chat_display
                    "",                                         # msg
                    gr.update(visible=False),                   # download_file
                    error_messages,                             # chat_messages
                    gr.update(value="Send", interactive=True)   # submit_btn
                )

        def use_suggestion(suggestion, messages):
            """Use a suggestion as input with progressive updates."""
            if not suggestion.strip():
                show_suggestions = len(messages) == 0
                return (
                    gr.update(visible=not show_suggestions),     # chat_section
                    gr.update(visible=show_suggestions),         # suggestions_section
                    render_chat_history(messages),              # chat_display
                    "",                                          # msg
                    gr.update(visible=False),                    # download_file
                    messages,                                    # chat_messages
                    gr.update(value="Send", interactive=True) # submit_btn
                )

            # Clear any previous stop request
            stop_requested.clear()

            # Immediately show user message and switch to chat mode
            updated_messages = messages.copy()
            updated_messages.append({"role": "user", "content": suggestion, "type": "text"})

            # First yield: immediate UI update showing user message
            yield (
                gr.update(visible=True),                    # chat_section
                gr.update(visible=False),                   # suggestions_section
                render_chat_history(updated_messages),     # chat_display
                "",                                         # msg
                gr.update(visible=False),                   # download_file
                updated_messages,                          # chat_messages
                gr.update(value="Stop", interactive=True)   # submit_btn - show Stop during processing
            )

            # Process the query - call process_query directly to avoid duplication
            final_result = None
            try:
                for result in process_query(suggestion, messages):
                    final_result = result
                    # Yield intermediate results with Stop button
                    updated_messages, empty_input, file_path, show_sugg = result
                    yield (
                        gr.update(visible=True),                    # chat_section
                        gr.update(visible=False),                   # suggestions_section
                        render_chat_history(updated_messages),     # chat_display
                        "",                                         # msg
                        gr.update(visible=bool(file_path), value=file_path if file_path else None),  # download_file
                        updated_messages,                          # chat_messages
                        gr.update(value="Stop", interactive=True)    # submit_btn - show Stop during processing
                    )

                # Return final result with Send button
                if final_result:
                    updated_messages, empty_input, file_path, show_sugg = final_result
                    return (
                        gr.update(visible=True),                    # chat_section
                        gr.update(visible=False),                   # suggestions_section
                        render_chat_history(updated_messages),     # chat_display
                        "",                                         # msg
                        gr.update(visible=bool(file_path), value=file_path if file_path else None),  # download_file
                        updated_messages,                          # chat_messages
                        gr.update(value="Send", interactive=True)    # submit_btn - back to Send when complete
                    )
            except Exception as e:
                logger.error(f"Error in use_suggestion: {e}", exc_info=True)
                error_messages = updated_messages.copy()
                error_messages.append({"role": "assistant", "content": f"Error: {str(e)}", "type": "text"})
                return (
                    gr.update(visible=True),                    # chat_section
                    gr.update(visible=False),                   # suggestions_section
                    render_chat_history(error_messages),       # chat_display
                    "",                                         # msg
                    gr.update(visible=False),                   # download_file
                    error_messages,                             # chat_messages
                    gr.update(value="Send", interactive=True)   # submit_btn
                )

        def handle_stop(messages):
            """Handle stop button click - cancel processing and reset UI."""
            # Signal to stop any ongoing processing
            stop_requested.set()
            logger.info("Stop requested by user")

            # Determine if we should show suggestions or stay in chat mode
            show_suggestions = len(messages) == 0

            # Add stop message to chat if there are messages
            if not show_suggestions:
                updated_messages = messages.copy()
                updated_messages.append({
                    "role": "assistant",
                    "content": "Processing stopped by user.",
                    "type": "text"
                })
            else:
                updated_messages = messages

            yield (
                gr.update(visible=not show_suggestions),        # chat_section
                gr.update(visible=show_suggestions),            # suggestions_section
                render_chat_history(updated_messages),         # chat_display
                gr.update(value="", interactive=True),          # msg - ensure input is re-enabled
                gr.update(visible=False),                       # download_file
                updated_messages,                               # chat_messages
                gr.update(value="Analyze" if show_suggestions else "Send", interactive=True)  # submit_btn
            )

        def handle_button_click(current_button_value, message, messages):
            """Handle button click - either start processing or stop."""
            if current_button_value == "Stop":
                # Stop button clicked - cancel processing (generator)
                yield from handle_stop(messages)
            else:
                # Clear any previous stop request and start processing
                stop_requested.clear()
                yield from handle_message(message, messages)

        def update_button_state(text, is_processing=False):
            """Update button state based on input text and processing state."""
            if is_processing:
                return gr.update(value="Stop", interactive=True)
            else:
                is_empty = not text.strip()
                return gr.update(value="Analyze" if is_empty else "Send", interactive=True)

        # Wire up events
        submit_outputs = [chat_section, suggestions_section, chat_display, msg, download_file, chat_messages, submit_btn]

        def handle_msg_submit(message, messages):
            """Handle message submission - clear stop request and process."""
            stop_requested.clear()
            yield from handle_message(message, messages)

        msg.submit(
            handle_msg_submit,
            inputs=[msg, chat_messages],
            outputs=submit_outputs
        )

        submit_btn.click(
            handle_button_click,
            inputs=[submit_btn, msg, chat_messages],
            outputs=submit_outputs
        )

        # Button is always enabled - no state checking needed

        # Wire up suggestion buttons
        for i, btn in enumerate(suggestion_buttons):
            btn.click(
                use_suggestion,
                inputs=[gr.State(suggestions[i]), chat_messages],
                outputs=submit_outputs
            )

    return demo


def main():
    """Main application entry point."""
    demo = create_interface()

    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections for Docker
        server_port=8505,
        share=False,
        show_error=True,
        quiet=False,
        show_api=False,  # Hide API documentation
        favicon_path=None  # Remove favicon
    )


if __name__ == "__main__":
    main()