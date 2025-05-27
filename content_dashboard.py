"""
Content Creation Dashboard for LangGraph 101 project.

This module provides the Streamlit UI components for the content creation dashboard.
"""
import streamlit as st
import os
import base64
from typing import Dict, Any, List, Optional
import pandas as pd
import altair as alt
import json
import tempfile
import time
from datetime import datetime
import subprocess
import logging
from dotenv import load_dotenv

# Import content creation functionality
from content_creation import get_content_creator
from config import load_config
# Import new tools
from tools import search_web, calculator, get_weather_info, search_news


# Explicitly load the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Debug log for .env loading
logger = logging.getLogger(__name__)
logger.debug(f"Loading .env file from: {dotenv_path}")
if not os.path.exists(dotenv_path):
    logger.error(".env file not found at the specified path.")
else:
    logger.debug(".env file loaded successfully.")

# Helper function moved to module level for broader accessibility
def get_api_key_from_sources(secret_name: str, *env_var_names: str) -> Optional[str]:
    """Retrieve an API key, prioritizing st.secrets, then environment variables."""
    logger.info(f"Attempting to retrieve {secret_name} from st.secrets and environment variables.")
    if hasattr(st, 'secrets'):
        try:
            # Attempt to access the specific secret
            if secret_name in st.secrets:
                key_value = st.secrets[secret_name]
                if key_value and str(key_value).strip():  # Ensure the secret is not empty or just whitespace
                    logger.info(f"{secret_name} found in st.secrets.")
                    return str(key_value).strip()
        except st.errors.StreamlitSecretNotFoundError:
            logger.warning(f"{secret_name} not found in st.secrets. Falling back to environment variables.")
        except Exception as e:
            logger.error(f"Error accessing st.secrets for {secret_name}: {e}")

    # Fallback to environment variables
    env_key = next((os.getenv(var) for var in env_var_names if os.getenv(var)), None)
    if env_key:
        logger.info(f"{secret_name} found in environment variables.")
    else:
        logger.warning(f"{secret_name} not found in environment variables.")
    return env_key


def display_standardized_error(error_result, default_message="An error occurred."):
    """Display a standardized error response in Streamlit, with technical details if available."""
    if isinstance(error_result, dict) and ("error" in error_result or "error_type" in error_result):
        error_type = error_result.get("error_type", "error").replace("_", " ").title()
        error_msg = error_result.get("error", default_message)
        st.error(f"{error_type}: {error_msg}")
        # Show context as JSON if present
        if error_result.get("context"):
            with st.expander("Show error details"):
                st.json(error_result["context"])
        # Show technical details if present
        if error_result.get("traceback"):
            st.exception(error_result["traceback"])
    elif isinstance(error_result, Exception):
        st.error(default_message)
        st.exception(error_result)
    else:
        st.error(str(error_result) if error_result else default_message)


def render_content_creation_dashboard():
    """Render the content creation dashboard in Streamlit."""
    st.title("🚀 Content Creation Dashboard")

    if "content_creator_initialized" not in st.session_state:
        try:
            config = load_config() # For non-API key settings like model_name, temperature

            api_keys_for_creator: Dict[str, Optional[str]] = {}
            missing_keys_for_init = []

            # Define keys needed by content_creator and their secret/env names
            # Format: (dict_key, display_name, secret_name, *env_vars)
            key_definitions = [
                ("api_key", "Gemini", "GEMINI_API_KEY", "API_KEY", "GEMINI_API_KEY"),
                ("openai", "OpenAI", "OPENAI_API_KEY", "OPENAI_API_KEY"),
                ("elevenlabs", "ElevenLabs", "ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"),
                ("dalle", "DALL-E", "DALLE_API_KEY", "DALLE_API_KEY"), # DALL-E might also use OpenAI key internally
                ("tavily", "Tavily", "TAVILY_API_KEY", "TAVILY_API_KEY"),
                ("stabilityai", "Stability AI", "STABILITYAI_API_KEY", "STABILITY_API_KEY"),
                ("pixabay", "Pixabay", "PIXABAY_API_KEY", "PIXABAY_API_KEY"),
                ("pexels", "Pexels", "PEXELS_API_KEY", "PEXELS_API_KEY"),
                ("deepgram", "Deepgram", "DEEPGRAM_API_KEY", "DEEPGRAM_API_KEY"),
                ("assemblyai", "AssemblyAI", "ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY"),
                ("youtube_data", "YouTube Data", "YOUTUBE_DATA_API_KEY", "YOUTUBE_API_KEY")
            ]

            essential_keys = ["api_key", "openai", "elevenlabs"] # Keys absolutely essential for basic operation

            for dict_key, display_name, secret_name, *env_vars in key_definitions:
                key_value = get_api_key_from_sources(secret_name, *env_vars)
                api_keys_for_creator[dict_key] = key_value
                if dict_key in essential_keys and not key_value:
                    missing_keys_for_init.append(f"{display_name} ({secret_name} or {', '.join(env_vars)})")

            # Special handling for DALL-E: can use OpenAI key if DALL-E specific key is missing
            if not api_keys_for_creator.get("dalle") and api_keys_for_creator.get("openai"):
                # If content_creator uses "dalle" key but can fallback to "openai", this logic might need adjustment
                # For now, we assume "dalle" key is preferred if present, otherwise it might use "openai" if configured internally
                pass # No explicit override here, content_creator should handle this

            if missing_keys_for_init:
                error_msg = f"Failed to initialize content creator. Essential API key(s) missing: {', '.join(missing_keys_for_init)}. Please set them in your Streamlit secrets (e.g., secrets.toml) or environment variables. See onboarding help for details."
                st.error(error_msg)
                st.session_state['last_dashboard_error'] = error_msg
                return

            # Add non-API key configurations
            api_keys_for_creator["model_name"] = config.get("model_name", "gemini-pro")
            api_keys_for_creator["temperature"] = config.get("temperature", 0.7)

            # Filter out None values before passing to get_content_creator if it expects all keys to be strings
            # However, it's often better if get_content_creator can handle Optional[str] for robustness
            # For now, we pass them as is, assuming get_content_creator can manage.
            # If get_content_creator strictly requires all keys to be present and be strings,
            # then more checks or different handling for optional services would be needed.

            st.session_state.content_creator = get_content_creator(api_keys_for_creator) # type: ignore
            st.session_state.content_creator_initialized = True
            st.success("Content creator initialized successfully.")

        except Exception as e:
            st.error(f"Failed to initialize content creator: {str(e)}")
            st.info("Please check your Streamlit secrets (e.g., secrets.toml) or environment variables for the necessary API keys, and ensure the application has the correct permissions.")
            st.exception(e) # Show full traceback for debugging
            return

    # Help expander with usage instructions
    with st.expander("📋 How to Use the Content Creation Dashboard"):
        st.markdown("""
        ## Quick Start

        1. **Full Workflow**: Enter your topic and requirements, and let the agent generate script, audio, images, and more automatically.
        2. **Manual Tools**: Use any tool separately (e.g., just generate a script, just TTS, just a thumbnail).

        ### Available Tools
        | Tool/API         | What it Does                                 |
        |------------------|----------------------------------------------|
        | Google Gemini    | Generate scripts, titles, descriptions       |
        | ElevenLabs TTS   | Convert script to high-quality voiceover     |
        | Pexels/Pixabay   | Fetch stock images/videos                    |
        | Stability AI/DALL-E | Generate custom images (thumbnails, art)  |
        | AssemblyAI/Deepgram | Transcribe audio/video                    |
        | YouTube Data API | Research trends, keywords, competitors       |

        ### Example Workflow
        1. Enter: "5-min video on password security, Don Corleone style, with thumbnail and narration"
        2. Agent:
           - Generates script, title, description (Gemini)
           - Converts script to audio (ElevenLabs)
           - Generates thumbnail (Stability AI/DALL-E)
           - Fetches stock clips/images (Pexels/Pixabay)
           - Transcribes audio for captions (AssemblyAI/Deepgram)
        3. Returns: script, audio, images, captions, and links
        """)

    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Full Workflow", "🛠️ Individual Tools", "🤖 AI Tools", "📊 API Analytics"])

    # Full Workflow tab
    with tab1:
        render_full_workflow()

    # Individual Tools tab
    with tab2:
        render_individual_tools()

    # AI Tools tab
    with tab3:
        render_ai_tools_tab()

    # API Analytics tab
    with tab4:
        render_api_analytics_tab()


def render_ai_tools_tab():
    """Render the AI Tools tab with UI for all new AI-powered features."""
    st.header("🤖 AI Tools & Automation")
    st.write("Access advanced AI-powered coding, automation, and analytics tools directly from your browser.")

    # 1. AI Code Generation
    with st.expander("💡 AI Code Generation (Gemini)", expanded=False):
        prompt = st.text_area("Enter your code or doc prompt:", key="ai_code_gen_prompt")
        file_out = st.text_input("Output file (optional):", key="ai_code_gen_file")
        if st.button("Generate Code", key="ai_code_gen_btn"):
            if not prompt:
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Generating code with Gemini..."):
                    try:
                        cmd = ["python", "generate_code.py", "--prompt", prompt]
                        if file_out:
                            cmd += ["--output", file_out]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        st.code(result.stdout or result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 2. AI Error Log Analysis
    with st.expander("🪲 AI Error Log Analyzer", expanded=False):
        log_path = st.text_input("Path to error log file:", key="ai_error_log_path", value="error_logs/latest.log")
        if st.button("Analyze Errors", key="ai_error_analyze_btn"):
            with st.spinner("Analyzing errors with Gemini..."):
                try:
                    cmd = ["python", "ai_error_analyzer.py", "--log", log_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    st.code(result.stdout or result.stderr)
                except Exception as e:
                    st.error(f"Error: {e}")

    # 3. Agent Automation (Natural Language)
    with st.expander("🤖 Agent Automation CLI", expanded=False):
        agent_cmd = st.text_area("Describe your coding/testing/deployment task:", key="agent_cli_cmd")
        if st.button("Run Agent Task", key="agent_cli_btn"):
            if not agent_cmd:
                st.warning("Please enter a task description.")
            else:
                with st.spinner("Running agent automation..."):
                    try:
                        cmd = ["python", "agent_cli.py", "--task", agent_cmd]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        st.code(result.stdout or result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 4. AI Performance Analysis
    with st.expander("⚡ AI Performance Analyzer", expanded=False):
        perf_log = st.text_input("Path to performance log/metrics file:", key="ai_perf_log", value="performance_cache/latest_metrics.json")
        if st.button("Analyze Performance", key="ai_perf_analyze_btn"):
            with st.spinner("Analyzing performance with Gemini..."):
                try:
                    cmd = ["python", "ai_performance_analyzer.py", "--metrics", perf_log]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    st.code(result.stdout or result.stderr)
                except Exception as e:
                    st.error(f"Error: {e}")

    # 5. AI Documentation Generation
    with st.expander("📄 AI Documentation Generator", expanded=False):
        doc_file = st.text_input("Path to code file for documentation:", key="ai_doc_file")
        if st.button("Generate/Update Docs", key="ai_doc_gen_btn"):
            if not doc_file:
                st.warning("Please enter a file path.")
            else:
                with st.spinner("Generating documentation with Gemini..."):
                    try:
                        cmd = ["python", "ai_doc_generator.py", "--file", doc_file]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                        st.code(result.stdout or result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 6. Unified AI Tools Launcher
    with st.expander("🧰 Unified AI Tools Launcher", expanded=False):
        st.write("Launch the full AI tools menu in your terminal (for advanced workflows).")
        if st.button("Open AI Tools Launcher (Terminal)", key="ai_tools_launcher_btn"):
            st.info("Please run 'python ai_tools_launcher.py' in your terminal for the full menu-driven experience.")


def render_full_workflow():
    """Render the full content creation workflow UI."""
    st.header("Full Content Creation Workflow")
    st.write("Generate all assets for your content with a single request.")

    # Check for required API keys before rendering the rest of the workflow
    # This uses the now module-level get_api_key_from_sources
    missing_display_keys = []

    # Helper get_api_key_from_sources is now at module level

    # Check for "api_key" (Gemini)
    if not get_api_key_from_sources("GEMINI_API_KEY", "API_KEY", "GEMINI_API_KEY"):
        missing_display_keys.append("api_key (Gemini - GEMINI_API_KEY or API_KEY)")

    # Check for "tavily_api_key"
    if not get_api_key_from_sources("TAVILY_API_KEY", "TAVILY_API_KEY"):
        missing_display_keys.append("tavily_api_key (TAVILY_API_KEY)")

    # Check for "openai"
    if not get_api_key_from_sources("OPENAI_API_KEY", "OPENAI_API_KEY"):
        missing_display_keys.append("openai (OPENAI_API_KEY)")

    # Check for "elevenlabs"
    if not get_api_key_from_sources("ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"):
        missing_display_keys.append("elevenlabs (ELEVENLABS_API_KEY)")

    # Check for "dalle" (OpenAI key is primary, DALL-E specific key is fallback)
    if not (get_api_key_from_sources("OPENAI_API_KEY", "OPENAI_API_KEY") or get_api_key_from_sources("DALLE_API_KEY", "DALLE_API_KEY")):
        missing_display_keys.append("dalle (DALLE_API_KEY or OPENAI_API_KEY)")

    if missing_display_keys:
        unique_keys_to_check = sorted(list(set(missing_display_keys)))
        msg = f"Action required: Please ensure the following API keys are set in your Streamlit secrets (e.g., secrets.toml) or environment variables: {', '.join(unique_keys_to_check)}. See onboarding help for details."
        st.error(msg)
        st.session_state['last_dashboard_error'] = msg
        st.info("For local development without Streamlit Cloud, you can set these in a `.env` file in the project root (ensure `python-dotenv` is installed and used by your application to load it), or export them to your shell environment.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Topic input
        topic = st.text_area("Topic or title",
                            placeholder="Enter a topic for your video (e.g., 'Password security best practices for beginners')",
                            height=100)

        # Tone selection
        tones = ["Professional", "Casual", "Humorous", "Educational", "Dramatic", "Inspirational"]
        tone = st.selectbox("Tone", tones, index=0)

        # Duration selection
        duration = st.slider("Duration (minutes)", min_value=1, max_value=30, value=5, step=1)

    with col2:
        # Asset selection
        st.subheader("Assets to Generate")
        generate_tts = st.checkbox("Audio Narration", value=True)
        generate_thumbnail = st.checkbox("Thumbnail Image", value=True)
        search_stock = st.checkbox("Stock Media", value=True)
        transcribe = st.checkbox("Transcription/Captions", value=True)

        # Generate stock media options
        if search_stock:
            stock_provider = st.radio("Stock Media Provider", ["Pexels", "Pixabay"], horizontal=True)
            stock_type = st.radio("Stock Media Type", ["Photo", "Video"], horizontal=True)

        # Generate image options
        if generate_thumbnail:
            image_provider = st.radio("Image Generation Provider", ["DALL-E", "Stability AI"], horizontal=True)

    # Run full workflow
    if st.button("🚀 Generate All Content", type="primary", use_container_width=True):
        if not topic:
            st.error("Please enter a topic before generating content.")
            return

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize results container for assets
        if "workflow_results" not in st.session_state:
            st.session_state.workflow_results = {}

        try:
            # Update status
            status_text.text("Starting content generation...")
            progress_bar.progress(10)

            # Run the pipeline
            image_provider_param = image_provider.lower().replace(" ", "").replace("-", "") if generate_thumbnail else "dalle"
            stock_provider_param = stock_provider.lower() if search_stock else "pexels"

            # Get content creator instance
            content_creator = st.session_state.content_creator

            # Step 1: Generate script
            status_text.text("Generating script...")
            script_result = content_creator.generate_script(topic, tone.lower(), duration)
            progress_bar.progress(30)

            if "error" in script_result and script_result["error"]:
                display_standardized_error(script_result)
                return

            # Store script result
            st.session_state.workflow_results["script"] = script_result
              # Step 2: Generate TTS audio if selected
            if generate_tts:
                status_text.text("Generating audio narration...")
                script_text = script_result.get("script", "")
                # Make sure we're sending plain text to TTS
                if isinstance(script_text, dict):
                    script_text = script_text.get("script", str(script_text))
                tts_result = content_creator.generate_tts(script_text)
                progress_bar.progress(50)

                if not tts_result.get("success"):
                    display_standardized_error(tts_result)
                else:
                    st.session_state.workflow_results["audio"] = tts_result

                    # Step 3: Transcribe audio if selected
                    if transcribe and "filepath" in tts_result:
                        status_text.text("Generating transcription and captions...")
                        transcription_result = content_creator.transcribe_audio(tts_result["filepath"])
                        progress_bar.progress(70)

                        if not transcription_result.get("success"):
                            display_standardized_error(transcription_result)
                        else:
                            st.session_state.workflow_results["transcription"] = transcription_result

            # Step 4: Generate thumbnail if selected
            if generate_thumbnail:
                status_text.text("Generating thumbnail image...")
                thumbnail_prompt = f"Create a striking thumbnail for a video about {topic}. Professional quality, eye-catching, suitable for YouTube."
                thumbnail_result = content_creator.generate_image(thumbnail_prompt, provider=image_provider_param)
                progress_bar.progress(85)

                if not thumbnail_result.get("success"):
                    display_standardized_error(thumbnail_result)
                else:
                    st.session_state.workflow_results["thumbnail"] = thumbnail_result

            # Step 5: Search stock media if selected
            if search_stock:
                status_text.text("Searching for stock media...")
                stock_result = content_creator.search_stock_media(
                    topic,
                    media_type=stock_type.lower(),
                    provider=stock_provider_param
                )
                progress_bar.progress(95)

                if not stock_result.get("success"):
                    display_standardized_error(stock_result)
                else:
                    st.session_state.workflow_results["stock_media"] = stock_result

            # Complete!
            progress_bar.progress(100)
            status_text.text("Content generation complete!")

            # Display results
            display_workflow_results()

        except Exception as e:
            # Enhanced actionable error handling for dashboard testability
            import re
            error_str = str(e).lower()
            actionable_msg = None
            if (
                'api key' in error_str or 'apikey' in error_str or 'authentication' in error_str or 'no api key' in error_str
            ):
                actionable_msg = "Action required: Please check your API keys in the .env file or dashboard settings. See onboarding help for details."
            elif (
                'internet' in error_str or 'network' in error_str or 'connection' in error_str or 'timeout' in error_str
            ):
                actionable_msg = "Action required: Please check your internet connection and try again."

            if actionable_msg:
                st.error(actionable_msg)
                st.session_state['last_dashboard_error'] = actionable_msg
            else:
                st.error(f"An error occurred during content generation: {str(e)}")
                st.session_state['last_dashboard_error'] = str(e)
            progress_bar.empty()
            status_text.empty()

    # Display previous results if they exist
    if "workflow_results" in st.session_state and st.session_state.workflow_results:
        display_workflow_results()


def display_workflow_results():
    """Display the results of the content generation workflow."""
    if "workflow_results" in st.session_state or not st.session_state.workflow_results:
        return

    results = st.session_state.workflow_results

    st.divider()
    st.subheader("📊 Generated Content")

    # Display script if available
    if "script" in results:
        with st.expander("📝 Script", expanded=True):
            script = results["script"]
            st.markdown(f"### {script.get('title', 'Generated Script')}")
            st.markdown(f"**Description:** {script.get('description', '')}")

            # Show script text in a code block for better visibility
            st.code(script.get("script", ""), language="")

            # Download button for script
            if "filepath" in script:
                with open(script["filepath"], "r") as f:
                    script_content = f.read()

                st.download_button(
                    label="📥 Download Script",
                    data=script_content,
                    file_name=os.path.basename(script["filepath"]),
                    mime="text/plain"
                )

    # Display audio if available
    if "audio" in results:
        with st.expander("🔊 Audio Narration", expanded=True):
            audio = results["audio"]
            if audio.get("success") and "filepath" in audio:
                try:
                    # Display audio player
                    with open(audio["filepath"], "rb") as f:
                        audio_bytes = f.read()

                    st.audio(audio_bytes, format="audio/mp3")

                    # Show metadata
                    st.markdown(f"**Duration:** ~{int(audio.get('duration_seconds', 0))} seconds")

                    # Download button
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_bytes,
                        file_name=os.path.basename(audio["filepath"]),
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"Error displaying audio: {str(e)}")
            else:
                st.warning("Audio generation was not successful.")

    # Display transcription if available
    if "transcription" in results:
        with st.expander("📋 Transcription", expanded=True):
            transcription = results["transcription"]
            if transcription.get("success"):
                st.markdown("### Transcript")

                # Display transcript
                st.markdown(transcription.get("text", ""))

                # Display subtitle information if available
                subtitles = transcription.get("subtitles", {})
                if subtitles and subtitles.get("filepath"):
                    st.markdown("### Subtitles/Captions (SRT)")

                    # Show preview of subtitles
                    subtitle_lines = subtitles.get("lines", [])
                    if subtitle_lines:
                        st.markdown("**Preview:**")
                        for i, line in enumerate(subtitle_lines[:5]):
                            st.code(f"{line['line']}\n{line['start']} --> {line['end']}\n{line['text']}", language="")
                            if i >= 4:
                                st.markdown("*... (more lines) ...*")
                                break

                    # Download buttons
                    with open(transcription["filepath"], "r") as f:
                        transcript_content = f.read()

                    st.download_button(
                        label="📥 Download Transcript",
                        data=transcript_content,
                        file_name=os.path.basename(transcription["filepath"]),
                        mime="text/plain"
                    )

                    if subtitles.get("filepath"):
                        with open(subtitles["filepath"], "r") as f:
                            subtitles_content = f.read()

                        st.download_button(
                            label="📥 Download SRT Subtitles",
                            data=subtitles_content,
                            file_name=os.path.basename(subtitles["filepath"]),
                            mime="text/plain"
                        )
            else:
                st.warning("Transcription was not successful.")

    # Display thumbnail if available
    if "thumbnail" in results:
        with st.expander("🖼️ Thumbnail", expanded=True):
            thumbnail = results["thumbnail"]
            if thumbnail.get("success") and "filepath" in thumbnail:
                try:
                    # Display image
                    image_path = thumbnail["filepath"]
                    st.image(image_path, caption=f"Generated thumbnail ({thumbnail.get('provider', 'AI')})")

                    # Show prompt used
                    st.markdown(f"**Prompt:** {thumbnail.get('prompt', 'No prompt available')}")

                    # Download button
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()

                    st.download_button(
                        label="📥 Download Thumbnail",
                        data=image_bytes,
                        file_name=os.path.basename(image_path),
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error displaying thumbnail: {str(e)}")
            else:
                st.warning("Thumbnail generation was not successful.")

    # Display stock media if available
    media_results = []  # Default value to avoid undefined variable
    if "stock_media" in results:
        with st.expander("🎬 Stock Media", expanded=True):
            stock_media = results["stock_media"]
            if stock_media.get("success") and "results" in stock_media:
                media_results = stock_media["results"]
                provider = stock_media.get("provider", "").capitalize()
                media_type = stock_media.get("media_type", "media").capitalize()

                st.markdown(f"### {provider} {media_type} Results")
                st.markdown(f"**Search Query:** {stock_media.get('query', 'Unknown')}" )

                # Display results in a grid
                cols = st.columns(3)
                for i, media in enumerate(media_results[:9]):  # Limit to 9 items for UI clarity
                    with cols[i % 3]:
                        # For photos
                        if media.get("type") == "photo":
                            st.image(
                                media["src"],
                                caption=f"Photo by {media.get('photographer', media.get('user', 'Unknown'))}",
                                use_column_width=True
                            )
                        # For videos (show thumbnail and link)
                        else:
                            st.markdown(f"**Video {i+1}**")
                            st.markdown(f"By: {media.get('user', 'Unknown')}")
                            if media.get("src"):
                                st.video(media["src"])
                            else:
                                st.markdown(f"[View video]({media['url']})")

                        # Link to source
                        st.markdown(f"[View on {provider}]({media['url']})")
            else:
                st.warning("Stock media search was not successful.")

    # Clear results button
    st.button("🗑️ Clear Results", on_click=clear_workflow_results)


def clear_workflow_results():
    """Clear the workflow results from session state."""
    if "workflow_results" in st.session_state:
        st.session_state.workflow_results = {}


def render_individual_tools():
    """Render UI for individual content creation and utility tools."""
    st.header("🛠️ Individual Tools")
    st.write("Access specific tools for targeted tasks.")

    # Existing tools
    render_script_generator()
    st.divider()
    render_text_to_speech()
    st.divider()
    render_image_generator()
    st.divider()
    render_stock_media_search()
    st.divider()
    render_audio_transcription()
    st.divider()
    render_youtube_research()
    st.divider()

    # New utility tools
    render_web_search_tool()
    st.divider()
    render_calculator_tool()
    st.divider()
    render_weather_tool()
    st.divider()
    render_news_tool()


def render_script_generator():
    """Render the script generator tool UI."""
    st.subheader("📝 Script Generator")
    st.write("Generate professional scripts for your videos using Google Gemini.")

    if "content_creator" not in st.session_state or not st.session_state.get("content_creator_initialized"):
        st.warning("Content creator is not initialized. Please check essential API key configurations.")
        return

    # Form for script generation
    with st.form("script_form"):
        topic = st.text_area("Topic or title", placeholder="Enter a topic for your video")

        col1, col2 = st.columns(2)

        with col1:
            tones = ["Professional", "Casual", "Humorous", "Educational", "Dramatic", "Inspirational"]
            tone = st.selectbox("Tone", tones)

        with col2:
            duration = st.slider("Duration (minutes)", min_value=1, max_value=30, value=5)

        submit = st.form_submit_button("Generate Script", type="primary", use_container_width=True)

    # Process form submission
    if submit:
        if not topic:
            st.error("Please enter a topic before generating a script.")
            return

        with st.spinner("Generating script..."):
            try:
                # Get content creator instance
                content_creator = st.session_state.content_creator

                # Generate script
                result = content_creator.generate_script(topic, tone.lower(), duration)

                # Check for errors
                if "error" in result and result["error"]:
                    display_standardized_error(result)
                    return

                # Display results
                st.success("Script generated successfully!")

                # Display script
                st.markdown(f"### {result.get('title', 'Generated Script')}")
                st.markdown(f"**Description:** {result.get('description', '')}")

                # Show script text in a code block for better visibility
                st.code(result.get("script", ""), language="")

                # Download button for script
                if "filepath" in result:
                    with open(result["filepath"], "r") as f:
                        script_content = f.read()

                    st.download_button(
                        label="📥 Download Script",
                        data=script_content,
                        file_name=os.path.basename(result["filepath"]),
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def render_text_to_speech():
    """Render the text-to-speech tool UI."""
    st.subheader("🔊 Text-to-Speech")
    st.write("Convert text to natural-sounding speech using ElevenLabs.")

    if "content_creator" not in st.session_state or not st.session_state.get("content_creator_initialized"):
        st.warning("Content creator is not initialized. Please check essential API key configurations.")
        return

    # Form for TTS
    with st.form("tts_form"):
        text = st.text_area("Text to convert",
                          placeholder="Enter the text you want to convert to speech",
                          height=200)

        # Voice selection (simplified for now)
        # In a production app, you could fetch available voices from ElevenLabs API
        voices = {
            "Rachel (Female)": "21m00Tcm4TlvDq8ikWAM",
            "Domi (Female)": "AZnzlk1XvdvUeBnXmlld",
            "Bella (Female)": "EXAVITQu4vr4xnSDxMaL",
            "Antoni (Male)": "ErXwobaYiN019PkySvjV",
            "Josh (Male)": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold (Male)": "VR6AewLTigWG4xSOukaG"
        }

        voice_name = st.selectbox("Voice", list(voices.keys()))
        voice_id = voices[voice_name]

        submit = st.form_submit_button("Generate Speech", type="primary", use_container_width=True)

    # Process form submission
    if submit:
        if not text:
            st.error("Please enter text before generating speech.")
            return

        with st.spinner("Generating speech..."):
            try:
                # Get content creator instance
                content_creator = st.session_state.content_creator

                # Generate TTS
                result = content_creator.generate_tts(text, voice_id)

                # Check for errors
                if not result.get("success"):
                    display_standardized_error(result)
                    return

                # Display results
                st.success("Speech generated successfully!")

                # Display audio player
                with open(result["filepath"], "rb") as f:
                    audio_bytes = f.read()

                st.audio(audio_bytes, format="audio/mp3")

                # Show metadata
                st.markdown(f"**Duration:** ~{int(result.get('duration_seconds', 0))} seconds")
                st.markdown(f"**Voice:** {voice_name}")

                # Download button
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name=os.path.basename(result["filepath"]),
                    mime="audio/mp3"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def render_image_generator():
    """Render the image generator tool UI."""
    st.subheader("🖼️ Image Generator")
    st.write("Generate custom images for thumbnails, graphics, and more.")

    if "content_creator" not in st.session_state or not st.session_state.get("content_creator_initialized"):
        st.warning("Content creator is not initialized. Please check essential API key configurations.")
        return

    content_creator = st.session_state.content_creator
    dalle_key_present = bool(content_creator.api_keys.get("dalle") or content_creator.api_keys.get("openai"))
    stabilityai_key_present = bool(content_creator.api_keys.get("stabilityai"))

    if not dalle_key_present and not stabilityai_key_present:
        st.error("No image generation API keys (DALL-E/OpenAI or Stability AI) are configured. Please set the required keys.")
        return

    # Form for image generation
    with st.form("image_form"):
        prompt = st.text_area("Image description",
                            placeholder="Describe the image you want to generate in detail")

        available_providers = []
        if dalle_key_present:
            available_providers.append("DALL-E")
        if stabilityai_key_present:
            available_providers.append("Stability AI")

        if not available_providers: # Should be caught by the check above
            st.error("No image generation providers available due to missing API keys.")
            provider = None
        elif len(available_providers) == 1:
            provider = st.radio("Provider", available_providers, index=0, disabled=True)
            st.info(f"Only {available_providers[0]} is available as its API key is configured.")
        else:
            provider = st.radio("Provider", available_providers, horizontal=True)

        submit = st.form_submit_button("Generate Image", type="primary", use_container_width=True)

    # Process form submission
    if submit:
        if not prompt:
            st.error("Please enter an image description before generating.")
            return
        if not provider:
            st.error("No image generation provider selected or available.")
            return

        # Check if the selected provider's key is actually available (should be, due to UI logic)
        current_provider_key_ok = False
        if provider == "DALL-E" and dalle_key_present:
            current_provider_key_ok = True
        elif provider == "Stability AI" and stabilityai_key_present:
            current_provider_key_ok = True

        if not current_provider_key_ok:
            st.error(f"{provider} API key is not configured. Cannot generate image.")
            return

        with st.spinner("Generating image..."):
            try:
                # Get content creator instance
                content_creator = st.session_state.content_creator

                # Generate image
                provider_param = provider.lower().replace(" ", "").replace("-", "")
                result = content_creator.generate_image(prompt, provider=provider_param)

                # Check for errors
                if not result.get("success"):
                    display_standardized_error(result)
                    return

                # Display results
                st.success("Image generated successfully!")

                # Display image
                st.image(result["filepath"], caption=f"Generated by {provider}")

                # Show prompt
                st.markdown(f"**Prompt:** {prompt}")

                # Download button
                with open(result["filepath"], "rb") as f:
                    image_bytes = f.read()

                st.download_button(
                    label="📥 Download Image",
                    data=image_bytes,
                    file_name=os.path.basename(result["filepath"]),
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def render_stock_media_search():
    """Render the stock media search tool UI."""
    st.subheader("🎬 Stock Media Search")
    st.write("Search for royalty-free stock photos and videos.")

    if "content_creator" not in st.session_state or not st.session_state.get("content_creator_initialized"):
        st.warning("Content creator is not initialized. Please check essential API key configurations.")
        return

    content_creator = st.session_state.content_creator
    pexels_key_present = bool(content_creator.api_keys.get("pexels"))
    pixabay_key_present = bool(content_creator.api_keys.get("pixabay"))

    if not pexels_key_present and not pixabay_key_present:
        st.error("Neither Pexels nor Pixabay API key is configured. Please set PEXELS_API_KEY or PIXABAY_API_KEY.")
        return

    # Form for stock media search
    with st.form("stock_form"):
        query = st.text_input("Search query", placeholder="Enter keywords to search for")

        col1, col2 = st.columns(2)

        available_providers = []
        if pexels_key_present:
            available_providers.append("Pexels")
        if pixabay_key_present:
            available_providers.append("Pixabay")

        with col1:
            if not available_providers: # Should be caught by the check above
                st.error("No stock media providers available due to missing API keys.")
                provider = None
            elif len(available_providers) == 1:
                provider = st.radio("Provider", available_providers, index=0, disabled=True)
                st.info(f"Only {available_providers[0]} is available as its API key is configured.")
            else:
                provider = st.radio("Provider", available_providers, horizontal=True)

        with col2:
            media_type = st.radio("Media type", ["Photo", "Video"], horizontal=True)

        submit = st.form_submit_button("Search Media", type="primary", use_container_width=True)

    # Process form submission
    if submit:
        if not query:
            st.error("Please enter a search query.")
            return
        if not provider:
            st.error("No stock media provider selected or available.")
            return

        selected_provider_key_name = provider.lower()
        if not content_creator.api_keys.get(selected_provider_key_name):
            st.error(f"{provider} API key is not configured. Cannot perform search.")
            return

        with st.spinner("Searching for media..."):
            try:
                # Get content creator instance
                content_creator = st.session_state.content_creator

                # Search stock media
                result = content_creator.search_stock_media(
                    query,
                    media_type=media_type.lower(),
                    provider=provider.lower()
                )

                # Check for errors
                if not result.get("success"):
                    display_standardized_error(result)
                    return

                # Display results
                media_results = result.get("results", [])
                if not media_results:
                    st.warning(f"No {media_type.lower()} results found for '{query}'.")
                    return

                st.success(f"Found {len(media_results)} {media_type.lower()} results.")

                # Display results in a grid
                cols = st.columns(3)
                for i, media in enumerate(media_results[:9]):  # Limit to 9 items for UI clarity
                    with cols[i % 3]:
                        # For photos
                        if media.get("type") == "photo":
                            st.image(
                                media["src"],
                                caption=f"Photo by {media.get('photographer', media.get('user', 'Unknown'))}",
                                use_column_width=True
                            )
                        # For videos (show thumbnail and link)
                        else:
                            st.markdown(f"**Video {i+1}**")
                            st.markdown(f"By: {media.get('user', 'Unknown')}")
                            if media.get("src"):
                                st.video(media["src"])
                            else:
                                st.markdown(f"[View video]({media['url']})")

                        # Link to source
                        st.markdown(f"[View on {provider}]({media['url']})")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def render_audio_transcription():
    """Render the audio transcription tool UI."""
    st.subheader("📋 Audio Transcription")
    st.write("Transcribe audio files to text and generate subtitles.")

    if "content_creator" not in st.session_state or not st.session_state.get("content_creator_initialized"):
        st.warning("Content creator is not initialized. Please check essential API key configurations.")
        return

    content_creator = st.session_state.content_creator
    assemblyai_key_present = bool(content_creator.api_keys.get("assemblyai"))
    deepgram_key_present = bool(content_creator.api_keys.get("deepgram"))

    if not assemblyai_key_present and not deepgram_key_present:
        st.error("Neither AssemblyAI nor Deepgram API key is configured. Please set ASSEMBLYAI_API_KEY or DEEPGRAM_API_KEY.")
        return

    # Form for audio transcription
    with st.form("transcription_form"):
        uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a", "ogg"])

        available_providers = []
        if assemblyai_key_present:
            available_providers.append("AssemblyAI")
        if deepgram_key_present:
            available_providers.append("Deepgram")

        if not available_providers: # Should be caught by the check above
            st.error("No transcription providers available due to missing API keys.")
            provider = None
        elif len(available_providers) == 1:
            provider = st.radio("Transcription provider", available_providers, index=0, disabled=True)
            st.info(f"Only {available_providers[0]} is available as its API key is configured.")
        else:
            provider = st.radio("Transcription provider", available_providers, horizontal=True)

        submit = st.form_submit_button("Transcribe Audio", type="primary", use_container_width=True)

    # Process form submission
    if submit:
        if not uploaded_file:
            st.error("Please upload an audio file.")
            return
        if not provider:
            st.error("No transcription provider selected or available.")
            return

        selected_provider_key_name = provider.lower().replace(" ", "") # "assemblyai" or "deepgram"
        if not content_creator.api_keys.get(selected_provider_key_name):
            st.error(f"{provider} API key is not configured. Cannot transcribe audio.")
            return

        with st.spinner("Transcribing audio..."):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    audio_path = tmp.name

                # Get content creator instance
                content_creator = st.session_state.content_creator

                # Transcribe audio
                result = content_creator.transcribe_audio(audio_path, provider=provider.lower())

                # Delete temporary file
                try:
                    os.unlink(audio_path)
                except:
                    pass

                # Check for errors
                if not result.get("success"):
                    display_standardized_error(result)
                    return

                # Display results
                st.success("Audio transcribed successfully!")

                # Display transcript
                st.markdown("### Transcript")
                st.markdown(result.get("text", ""))

                # Display subtitle information if available
                subtitles = result.get("subtitles", {})
                if subtitles and subtitles.get("filepath"):
                    st.markdown("### Subtitles/Captions (SRT)")

                    # Show preview of subtitles
                    subtitle_lines = subtitles.get("lines", [])
                    if subtitle_lines:
                        st.markdown("**Preview:**")
                        for i, line in enumerate(subtitle_lines[:5]):
                            st.code(f"{line['line']}\n{line['start']} --> {line['end']}\n{line['text']}", language="")
                            if i >= 4:
                                st.markdown("*... (more lines) ...*")
                                break

                    # Download buttons
                    with open(result["filepath"], "r") as f:
                        transcript_content = f.read()

                    st.download_button(
                        label="📥 Download Transcript",
                        data=transcript_content,
                        file_name="transcript.txt",
                        mime="text/plain"
                    )

                    if subtitles.get("filepath"):
                        with open(subtitles["filepath"], "r") as f:
                            subtitles_content = f.read()

                        st.download_button(
                            label="📥 Download SRT Subtitles",
                            data=subtitles_content,
                            file_name="subtitles.srt",
                            mime="text/plain"
                        )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def render_youtube_research():
    """Render the YouTube research tool UI."""
    st.subheader("📺 YouTube Research")
    st.write("Search YouTube for relevant videos and insights.")

    if "content_creator" not in st.session_state or not st.session_state.get("content_creator_initialized"):
        st.warning("Content creator is not initialized. Please check essential API key configurations.")
        return

    content_creator = st.session_state.content_creator
    if not content_creator.api_keys.get("youtube_data"):
        st.error("YouTube Data API key not found. Please set YOUTUBE_DATA_API_KEY (or YOUTUBE_API_KEY) in your Streamlit secrets or environment variables to use this tool.")
        return

    # Form for YouTube research
    with st.form("youtube_form"):
        query = st.text_input("Research topic", placeholder="Enter a topic or keyword to research on YouTube")

        submit = st.form_submit_button("Research YouTube", type="primary", use_container_width=True)

    # Process form submission
    if submit:
        if not query:
            st.error("Please enter a research topic.")
            return

        with st.spinner("Researching YouTube..."):
            try:
                # Get content creator instance
                content_creator = st.session_state.content_creator

                # Research YouTube
                result = content_creator.youtube_research(query)

                # Check for errors
                if not result.get("success"):
                    display_standardized_error(result)
                    return

                # Display results
                videos = result.get("videos", [])
                if not videos:
                    st.warning(f"No YouTube results found for '{query}'.")
                    return

                st.success(f"Found {len(videos)} YouTube videos related to '{query}'.")

                # Display videos in a table
                video_data = []
                for video in videos:
                    video_data.append({
                        "Title": video["title"],
                        "Channel": video["channelTitle"],
                        "Views": f"{video['viewCount']:,}",
                        "Likes": f"{video['likeCount']:,}",
                        "Comments": f"{video['commentCount']:,}",
                        "URL": f"https://www.youtube.com/watch?v={video['id']}"
                    })

                df = pd.DataFrame(video_data)

                # Custom clickable links in dataframe
                def make_clickable(val):
                    return f'<a href="{val}" target="_blank">Watch</a>'

                df_html = df.style.format({"URL": make_clickable}).to_html()
                st.markdown(df_html, unsafe_allow_html=True)

                # Download button for research report
                with open(result["filepath"], "r") as f:
                    report_content = f.read()

                st.download_button(
                    label="📥 Download Research Report",
                    data=report_content,
                    file_name=os.path.basename(result["filepath"]),
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def render_web_search_tool():
    """Render the web search tool UI (Tavily)."""
    st.subheader("🌐 Web Search (Tavily)")
    st.write("Perform web searches using Tavily API.")

    tavily_api_key = get_api_key_from_sources("TAVILY_API_KEY", "TAVILY_API_KEY")
    if not tavily_api_key:
        st.error("Tavily API key not found. Please set TAVILY_API_KEY in your Streamlit secrets or environment variables to use this tool.")
        return

    # Form for web search
    query = st.text_input("Enter search query:", key="web_search_query", placeholder="e.g., latest AI advancements")

    if st.button("Search Web", key="web_search_btn"):
        if query:
            with st.spinner("Searching the web..."):
                try:
                    result = search_web.invoke(query)
                    st.session_state.web_search_result = result
                except Exception as e:
                    st.session_state.web_search_result = f"Error during web search: {str(e)}"
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a search query.")

    if "web_search_result" in st.session_state:
        st.markdown("#### Search Result:")
        st.markdown(st.session_state.web_search_result)


def render_calculator_tool():
    """Render the calculator tool UI."""
    st.subheader("🧮 Calculator")
    st.write("Perform basic calculations.")

    expression = st.text_input("Enter mathematical expression:", key="calculator_expression", placeholder="e.g., (2 + 3) * 5 / 2")

    if st.button("Calculate", key="calculator_btn"):
        if expression:
            with st.spinner("Calculating..."):
                try:
                    result = calculator.invoke(expression)
                    st.session_state.calculator_result = result
                except Exception as e:
                    st.session_state.calculator_result = f"Error during calculation: {str(e)}"
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a mathematical expression.")

    if "calculator_result" in st.session_state:
        st.markdown("#### Calculation Result:")
        st.code(st.session_state.calculator_result, language="")


def render_weather_tool():
    """Render the weather information tool UI."""
    st.subheader("🌦️ Weather Info")
    st.write("Get current weather information for a location.")

    # Assuming OPENWEATHERMAP_API_KEY is used by get_weather_info in tools.py
    # Secret name: OPENWEATHERMAP_API_KEY, Env Var: OPENWEATHERMAP_API_KEY
    weather_api_key = get_api_key_from_sources("OPENWEATHERMAP_API_KEY", "OPENWEATHERMAP_API_KEY")
    if not weather_api_key:
        st.error("Weather API key (OPENWEATHERMAP_API_KEY) not found. Please set it in your Streamlit secrets or environment variables to use this tool.")
        return

    # Form for weather info
    location = st.text_input("Enter location (city name):", key="weather_location", placeholder="e.g., London")

    if st.button("Get Weather", key="weather_btn"):
        if location:
            with st.spinner(f"Fetching weather for {location}..."):
                try:
                    result = get_weather_info.invoke(location)
                    st.session_state.weather_result = result
                except Exception as e:
                    st.session_state.weather_result = f"Error fetching weather: {str(e)}"
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a location.")

    if "weather_result" in st.session_state:
        st.markdown("#### Weather Report:")
        st.markdown(st.session_state.weather_result)


def render_news_tool():
    """Render the news search tool UI."""
    st.subheader("📰 News Search")
    st.write("Search for recent news articles on a topic.")

    # Assuming NEWSAPI_KEY is used by search_news in tools.py
    # Secret name: NEWSAPI_KEY, Env Var: NEWSAPI_KEY
    news_api_key = get_api_key_from_sources("NEWSAPI_KEY", "NEWSAPI_KEY")
    if not news_api_key:
        st.error("News API key (NEWSAPI_KEY) not found. Please set it in your Streamlit secrets or environment variables to use this tool.")
        return

    # Form for news search
    query = st.text_input("Enter news topic:", key="news_query", placeholder="e.g., technology trends")
    col1, col2 = st.columns(2)
    with col1:
        language = st.text_input("Language (e.g., en, pt):", value="pt", key="news_language")
    with col2:
        max_results = st.number_input("Max results:", min_value=1, max_value=20, value=5, key="news_max_results")

    if st.button("Search News", key="news_btn"):
        if query:
            with st.spinner(f"Searching news for '{query}'..."):
                try:
                    result = search_news.invoke({"query": query, "language": language, "max_results": int(max_results)})
                    st.session_state.news_result = result
                except Exception as e:
                    st.session_state.news_result = f"Error searching news: {str(e)}"
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a news topic.")

    if "news_result" in st.session_state:
        st.markdown("#### News Articles:")
        st.markdown(st.session_state.news_result)


def render_api_analytics_tab():
    """Render the API Analytics tab with usage charts."""
    st.header("📊 API Usage Analytics")

    try:
        from api_analytics import get_usage_summary # Assuming this is where it comes from
        usage_summary = get_usage_summary()
    except ImportError:
        st.error("Could not import `api_analytics`. Please ensure the module exists and is configured.")
        return
    except Exception as e:
        st.error(f"Error fetching API usage summary: {e}")
        return

    if "error" in usage_summary:
        st.error(f"Failed to load usage data: {usage_summary['error']}")
        if usage_summary.get("details"):
            st.warning(usage_summary["details"])
        return # Changed from continue as it's not in a loop

    # Display overall summary metrics
    col1_metrics, col2_metrics, col3_metrics = st.columns(3)
    col1_metrics.metric("Total API Calls", usage_summary.get("total_calls", 0))
    col2_metrics.metric("Total Cost", f"${usage_summary.get('total_cost', 0):.2f}")
    col3_metrics.metric("Monitored APIs", len(usage_summary.get("model_usage", [])))

    st.markdown("---")

    daily_df = pd.DataFrame(usage_summary.get("daily_usage", []))
    model_df = pd.DataFrame(usage_summary.get("model_usage", []))

    # Process and chart daily usage
    if not daily_df.empty:
        # Ensure 'Date' is datetime and valid
        if "Date" in daily_df.columns:
            daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors='coerce')
            daily_df.dropna(subset=["Date"], inplace=True)
        else:
            daily_df = pd.DataFrame() # Empty if 'Date' column is missing

        if not daily_df.empty:
            # Ensure numeric columns are numeric and fill NaNs with 0 for charting
            if "Cost ($)" in daily_df.columns:
                daily_df['Cost ($)'] = pd.to_numeric(daily_df['Cost ($)'], errors='coerce').fillna(0)
            else:
                daily_df['Cost ($)'] = 0

            if "Calls" in daily_df.columns:
                daily_df['Calls'] = pd.to_numeric(daily_df['Calls'], errors='coerce').fillna(0)
            else:
                daily_df['Calls'] = 0

            # Re-check if daily_df is empty after cleaning
            if not daily_df.empty:
                daily_df = daily_df.sort_values(by="Date") # Sort by date for line chart

                st.subheader("Usage Over Time")
                chart_cost_time = alt.Chart(daily_df).mark_line(point=True).encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y('Cost ($):Q', title='Cost (USD)'),
                    tooltip=[alt.Tooltip('Date:T', title='Date'), alt.Tooltip('Cost ($):Q', title='Cost', format='$.2f'), alt.Tooltip('Calls:Q', title='Calls')]
                ).properties(
                    title='Daily Cost Over Time'
                )
                st.altair_chart(chart_cost_time, use_container_width=True)

                chart_calls_time = alt.Chart(daily_df).mark_bar().encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y('Calls:Q', title='Number of Calls'),
                    tooltip=[alt.Tooltip('Date:T', title='Date'), alt.Tooltip('Calls:Q', title='Calls'), alt.Tooltip('Cost ($):Q', title='Cost', format='$.2f')]
                ).properties(
                    title='Daily API Calls Over Time'
                )
                st.altair_chart(chart_calls_time, use_container_width=True)
            else:
                st.info("No valid daily usage data available to display time-series charts after cleaning.")
        else:
            st.info("No daily usage data with valid dates available.")
    else:
        st.info("No daily usage data available to display charts.")

    st.markdown("---")

    # Process and chart model usage
    if not model_df.empty:
        if "Model" not in model_df.columns:
            model_df = pd.DataFrame() # Empty if 'Model' column is missing        else:
            model_df.dropna(subset=["Model"], inplace=True)


        if not model_df.empty:
            if "Cost ($)" in model_df.columns:
                model_df['Cost ($)'] = pd.to_numeric(model_df['Cost ($)'], errors='coerce').fillna(0)
            else:
                model_df['Cost ($)'] = 0

            if "Calls" in model_df.columns:
                model_df['Calls'] = pd.to_numeric(model_df['Calls'], errors='coerce').fillna(0)
            else:
                model_df['Calls'] = 0

            # Re-check if model_df is empty after cleaning
            if not model_df.empty:
                st.subheader("Usage by Model")
                chart_cost_model = alt.Chart(model_df).mark_bar().encode(
                    x=alt.X('Cost ($):Q', title='Total Cost (USD)'),
                    y=alt.Y('Model:N', sort='-x', title='Model'),
                    tooltip=[alt.Tooltip('Model:N', title='Model'), alt.Tooltip('Cost ($):Q', title='Cost', format='$.2f'), alt.Tooltip('Calls:Q', title='Calls')]
                ).properties(
                    title='Total Cost by Model'
                )
                st.altair_chart(chart_cost_model, use_container_width=True)

                chart_calls_model = alt.Chart(model_df).mark_bar().encode(
                    x=alt.X('Calls:Q', title='Total Calls'),
                    y=alt.Y('Model:N', sort='-x', title='Model'),
                    tooltip=[alt.Tooltip('Model:N', title='Model'), alt.Tooltip('Calls:Q', title='Calls'), alt.Tooltip('Cost ($):Q', title='Cost', format='$.2f')]
                ).properties(
                    title='Total API Calls by Model'
                )
                st.altair_chart(chart_calls_model, use_container_width=True)
            else:
                st.info("No valid model usage data available to display model-specific charts after cleaning.")
        else:
            st.info("No model usage data with valid model names available.")
    else:
        st.info("No model usage data available to display charts.")

# Make sure to call the main dashboard rendering function

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Safe debugging API key retrieval with fallback
try:
    stability_key_secrets = st.secrets.get('STABILITY_API_KEY', 'Not Found')
except Exception:
    stability_key_secrets = 'Secrets not configured'

try:
    youtube_key_secrets = st.secrets.get('YOUTUBE_API_KEY', 'Not Found')
except Exception:
    youtube_key_secrets = 'Secrets not configured'

logger.info(f"STABILITY_API_KEY from st.secrets: {stability_key_secrets}")
logger.info(f"YOUTUBE_API_KEY from st.secrets: {youtube_key_secrets}")
logger.info(f"STABILITY_API_KEY from os.getenv: {os.getenv('STABILITY_API_KEY', 'Not Found')}")
logger.info(f"YOUTUBE_API_KEY from os.getenv: {os.getenv('YOUTUBE_API_KEY', 'Not Found')}")
