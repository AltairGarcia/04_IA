"""
Content Creation module for LangGraph 101 project.

This module provides the content creation functionality for the LangGraph agent,
including script generation, text-to-speech, image generation, and more.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import logging
import tempfile
import requests
from datetime import datetime
import time
import json
from functools import wraps
import traceback
import openai  # Moved import to the top level
import re  # Import re for regular expressions

# Import API analytics
from api_analytics import track_api_usage

# Import LangChain community tools for Arxiv and Wikipedia
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.documents import Document  # For type hinting

# Import error handling utilities
from error_handling import (
    ErrorHandler,
    ErrorCategory,
    graceful_degradation,
    safe_request
)

from api_resilience import resilient_api_call

# Initialize logger
logger = logging.getLogger(__name__)

class ContentCreator:
    """Content Creation manager for the LangGraph project.

    This class provides methods for generating various content assets:
    - Scripts (via Gemini)
    - Audio (via ElevenLabs)
    - Images (via DALL-E, Stability AI)
    - Blog Posts (via Gemini)
    - Twitter Threads (via Gemini)
    - Stock media (via Pexels, Pixabay)
    - Transcriptions (via AssemblyAI, Deepgram)
    - YouTube research (via YouTube Data API)
    - Arxiv research (via LangChain community tools)
    - Wikipedia research (via LangChain community tools)
    """

    def __init__(self, api_keys: Dict[str, Optional[str]]):  # Updated type hint
        """Initialize the ContentCreator with API keys.

        Args:
            api_keys: Dictionary of API keys for various services, where keys can be None if not provided.
        """
        self.api_keys = api_keys
        self.output_dir = os.path.join(os.path.dirname(__file__), "content_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def _mask_key(self, key: str) -> str:
        if not key or len(key) < 8:
            return "***"
        return key[:4] + "..." + key[-4:]

    @track_api_usage("gemini")
    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=3, delay=2.0, backoff_factor=2.0)
    def generate_script(self, topic: str, tone: str, duration_minutes: int = 5) -> Dict[str, str]:
        """Generate a video script using Gemini.

        Args:
            topic: The topic of the script
            tone: The desired tone (casual, professional, etc.)
            duration_minutes: Target duration in minutes

        Returns:
            Dictionary with script, title, and description
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Validate inputs
        if not topic:
            logger.warning("generate_script called with empty topic.")
            return {"title": "", "description": "", "script": "Error: Topic cannot be empty."}

        if not tone:
            logger.warning("generate_script called with empty tone.")
            return {"title": "", "description": "", "script": "Error: Tone cannot be empty."}

        duration_minutes = max(1, min(30, duration_minutes))  # Limit duration to reasonable range

        # Estimated words per minute for speech
        words_per_minute = 150
        target_word_count = duration_minutes * words_per_minute

        # Create prompt
        prompt = f"""Create a video script about {topic}.
        Tone: {tone}
        Target length: {duration_minutes} minutes (approximately {target_word_count} words)

        Format your response as a JSON with the following structure:
        {{
            "title": "Catchy title for the video",
            "description": "SEO-friendly description for the video (2-3 sentences)",
            "script": "The full script with natural speaking style"
        }}

        Respond ONLY with the JSON, no other text.
        """

        # Retrieve API key, model name, and temperature from configuration
        gemini_api_key = self.api_keys.get("api_key")
        model_name = self.api_keys.get("model_name", "gemini-pro")
        temperature = float(self.api_keys.get("temperature", 0.7))

        if not gemini_api_key:
            logger.error("Missing Gemini API key in ContentCreator's api_keys.")
            raise ValueError("Missing Gemini API key")

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=temperature,
                client_options={'api_endpoint': os.getenv("GOOGLE_API_ENDPOINT")}, # Corrected: Added client_options for potential endpoint override and prepared for timeout
                # Add timeout directly if supported or via client_options if that's the way
            )
            # If timeout needs to be passed differently, adjust here. Example for some Google clients:
            # from google.api_core.client_options import ClientOptions
            # client_options = ClientOptions(api_endpoint=os.getenv("GOOGLE_API_ENDPOINT"), timeout=60)
            # llm = ChatGoogleGenerativeAI(..., client_options=client_options)

            # For ChatGoogleGenerativeAI, timeout is often part of a broader request_options or client_options
            # Let's assume for now it might be within a general options dict or specific param.
            # Re-checking documentation, it seems `request_options` was an attempt, but `client_options` is suggested by the warning.
            # However, `timeout` is usually a direct parameter to `invoke` or part of `generate`.
            # Let's try setting it via `client_options` if that's the intended way for overall client config,
            # or ensure it's applied at the `invoke` call if more specific.
            # The warning suggests 'client_options'. Let's assume it's for client-wide settings.
            # Actual timeout for a specific call is often in `llm.invoke(prompt, config={'request_timeout': 60})`

            # Correcting based on the warning and typical usage for client-level settings:
            current_client_options = {}
            api_endpoint = os.getenv("GOOGLE_API_ENDPOINT")
            if api_endpoint:
                current_client_options['api_endpoint'] = api_endpoint
            # The timeout for the specific call is better handled in `invoke` or `generate` methods if available.
            # If `ChatGoogleGenerativeAI` itself takes a default timeout in `client_options`, it would be specified in its constructor.
            # The previous `request_options={'timeout': 60}` was incorrect.
            # Let's remove it from constructor and rely on `invoke` if needed, or assume default for now if not directly supported in constructor this way.

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=temperature,
                client_options=current_client_options if current_client_options else None
            )

            logger.info(f"Generating script with Gemini. Topic: {topic}, Tone: {tone}, Duration: {duration_minutes} mins")
            response = llm.invoke(prompt, config={"request_timeout": 60}) # Pass timeout in invoke config
            # Assuming response.content contains the JSON string
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                logger.warning("Gemini response object does not have 'content' attribute, attempting to cast to string.")
                response_content = str(response)

            # Clean the response content to ensure it's valid JSON
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()
            elif response_content.strip().startswith("```"):
                response_content = response_content.strip()[3:-3].strip()

            script_data = json.loads(response_content)

            if not all(k in script_data for k in ["title", "description", "script"]):
                logger.error(f"Gemini response missing required keys. Response: {response_content}")
                raise ValueError("Invalid JSON structure from Gemini response.")
            logger.info(f"Successfully generated script for topic: {topic}")

            # Save script to file for download capability
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"script_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {script_data.get('title', 'Generated Script')}\n\n")
                    f.write(f"## Description\n{script_data.get('description', '')}\n\n")
                    f.write(f"## Script\n{script_data.get('script', '')}")

                # Add filepath to the return data
                script_data["filepath"] = filepath
            except IOError as e:
                logger.error(f"Failed to write script file: {str(e)}")

            return script_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}. Response: {response_content}")
            return {"title": "", "description": "", "script": f"Error: Could not parse script from AI response. Details: {str(e)}"}
        except Exception as e:
            logger.error(f"Error during Gemini script generation: {str(e)}")
            raise

    @track_api_usage("elevenlabs")
    @graceful_degradation
    @resilient_api_call(api_name="elevenlabs", max_retries=3, initial_backoff=2.0)
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    def generate_tts(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Dict[str, Any]:
        """Generate text-to-speech audio using ElevenLabs.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID

        Returns:
            Dictionary with audio file path and metadata
        """
        # Input validation
        if not text:
            raise ValueError("Text cannot be empty")

        # Check if input is a structured script (JSON or dict) and extract plain text
        if isinstance(text, dict):
            # If the input is a dictionary, extract the script part
            text = text.get("script", str(text))
        elif isinstance(text, str) and text.startswith("{") and text.endswith("}"):
            # Try to parse as JSON if it looks like a JSON string
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "script" in parsed:
                    text = parsed["script"]
            except json.JSONDecodeError:
                # Not valid JSON, keep as is
                pass

        # Check if input is a structured script (JSON or dict) and extract plain text
        if isinstance(text, dict):
            # If the input is a dictionary, extract the script part
            text = text.get("script", str(text))
        elif text.startswith("{") and text.endswith("}"):
            # Try to parse as JSON if it looks like a JSON string
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "script" in parsed:
                    text = parsed["script"]
            except json.JSONDecodeError:
                # Not valid JSON, keep as is
                pass

        # Limit text length to avoid excessive API usage and timeouts
        if len(text) > 5000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 5000 chars")
            text = text[:5000] + "..."

        # Get API key with validation
        api_key = self.api_keys.get("elevenlabs")
        logger.debug(f"[TTS] Using ElevenLabs API key: {self._mask_key(api_key)}")
        masked_key = (api_key[:6] + "..." + api_key[-4:]) if api_key and len(api_key) > 10 else str(api_key)
        if not api_key or not isinstance(api_key, str) or len(api_key) < 20 or "xxxx" in api_key:
            logger.error(f"Missing or invalid ElevenLabs API key: {masked_key}")
            raise ValueError("Missing or invalid ElevenLabs API key. Please check your .env and dashboard settings.")
        logger.info(f"Using ElevenLabs API key: {masked_key}")

        # The default voice_id is for ElevenLabs' "Rachel" voice
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        logger.debug(f"[TTS] Request URL: {url}")
        logger.debug(f"[TTS] Request headers: {{'xi-api-key': '{self._mask_key(api_key)}'}}")

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }

        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        # Use a timeout for the request
        start_time = time.time()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            request_time = time.time() - start_time
            logger.info(f"TTS API request completed in {request_time:.2f} seconds")

            # Handle the response
            if response.status_code == 200:
                # Create directory if it doesn't exist
                os.makedirs(self.output_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"audio_{timestamp}.mp3"
                filepath = os.path.join(self.output_dir, filename)

                try:
                    with open(filepath, "wb") as f:
                        f.write(response.content)

                    # Calculate more accurate duration estimate based on word count
                    word_count = len(text.split())
                    # Average English speaker: ~150 words per minute = 2.5 words per second
                    estimated_duration = word_count / 2.5

                    return {
                        "success": True,
                        "filepath": filepath,
                        "duration_seconds": estimated_duration,
                        "word_count": word_count,
                        "file_size_bytes": len(response.content)
                    }
                except IOError as e:
                    logger.error(f"Failed to write audio file: {str(e)}")
                    raise IOError(f"Failed to save audio file: {str(e)}")
            elif response.status_code == 400:
                # Bad request - might be due to text being too long or invalid
                error_data = response.json() if response.text else {"detail": "Unknown bad request error"}
                logger.error(f"ElevenLabs API validation error: {error_data}")
                raise ValueError(f"TTS API validation error: {error_data.get('detail', 'Unknown error')}")
            elif response.status_code in (401, 403):
                # Authentication issues
                logger.error(f"ElevenLabs API authentication error: {response.status_code}")
                raise ValueError(f"TTS API authentication failed: Invalid or expired API key")
            elif response.status_code == 429:
                # Rate limiting
                logger.error("ElevenLabs API rate limit exceeded")
                raise ValueError("TTS API rate limit exceeded. Please try again later.")
            else:
                # Other API errors
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"TTS API error: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)} | text_len={len(text)}, voice_id={voice_id}", exc_info=True)
            return ErrorHandler.format_error_response(e, context={"text_len": len(text), "voice_id": voice_id})

    @track_api_usage("dalle")  # Will be overridden to "stabilityai" in the method if needed
    @graceful_degradation
    @resilient_api_call(api_name="dalle", max_retries=3, initial_backoff=2.0)
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    def generate_image(self, prompt: str, provider: str = "dalle") -> Dict[str, Any]:
        """Generate an image using DALL-E or Stability AI.

        Args:
            prompt: Description of the image to generate
            provider: 'dalle' or 'stability'

        Returns:
            Dictionary with image file path and metadata
        """
        # Validate inputs
        if not prompt:
            logger.error("generate_image called with empty prompt.")
            # Raise ValueError for invalid input, consistent with other methods
            raise ValueError("Prompt cannot be empty for image generation.")

        if len(prompt) > 1000: # DALL-E 2 prompt limit, DALL-E 3 is 4000. Stability varies.
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to 1000 chars for DALL-E compatibility.")
            prompt = prompt[:1000] # Truncate, or handle based on provider

        # Placeholder for actual implementation
        logger.warning(f"Image generation for provider '{provider}' is not fully implemented yet.")
        # Return an error structure consistent with other methods
        return {
            "error": f"Image generation with {provider} is not implemented yet.",
            "prompt_received": prompt,
            "filepath": None # Ensure consistent error return structure
        }

    @track_api_usage("gemini")
    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=2, delay=3.0, backoff_factor=1.5) # Adjusted retry for potentially large content
    def generate_blog_post(self, topic: str, tone: str, target_word_count: int = 800, keywords: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate a blog post using Gemini.

        Args:
            topic: The topic of the blog post.
            tone: The desired tone (e.g., informative, casual, professional).
            target_word_count: Approximate desired word count for the blog post.
            keywords: Optional list of keywords for SEO.

        Returns:
            Dictionary with blog post title, meta_description, content (Markdown), and filepath.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not topic:
            logger.warning("generate_blog_post called with empty topic.")
            return {"title": "", "meta_description": "", "content": "Error: Topic cannot be empty.", "filepath": ""}
        if not tone:
            logger.warning("generate_blog_post called with empty tone.")
            return {"title": "", "meta_description": "", "content": "Error: Tone cannot be empty.", "filepath": ""}

        target_word_count = max(200, min(5000, target_word_count)) # Reasonable limits

        keyword_string = ""
        if keywords:
            keyword_string = f"Please incorporate the following keywords naturally: {', '.join(keywords)}."

        prompt = f"""Create a blog post about '{topic}'.
        Tone: {tone}
        Target word count: Approximately {target_word_count} words.
        {keyword_string}

        Format your response as a SINGLE, VALID JSON object with the following keys:
        "title": "A catchy and SEO-friendly title for the blog post",
        "meta_description": "A concise meta description (150-160 characters) for SEO",
        "content": "The full blog post content in Markdown format. Ensure it is well-structured with headings, paragraphs, and lists where appropriate. The content should be engaging and informative. Ensure all strings within the JSON are properly escaped."

        Respond ONLY with the JSON object. Do not include any other text, explanations, or markdown formatting like ```json before or after the JSON.
        """

        gemini_api_key = self.api_keys.get("api_key")
        model_name = self.api_keys.get("model_name", "gemini-1.5-flash-latest") # Use a potentially more capable model for JSON
        temperature = float(self.api_keys.get("temperature", 0.7))

        if not gemini_api_key:
            logger.error("Missing Gemini API key in ContentCreator's api_keys for blog post generation.")
            return {"error": "Missing Gemini API key for blog post generation."}

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=temperature,
                client_options={'api_endpoint': os.getenv("GOOGLE_API_ENDPOINT")}, # Corrected: Added client_options for potential endpoint override and prepared for timeout
                # Add timeout directly if supported or via client_options if that's the way
            )
            # If timeout needs to be passed differently, adjust here. Example for some Google clients:
            # from google.api_core.client_options import ClientOptions
            # client_options = ClientOptions(api_endpoint=os.getenv("GOOGLE_API_ENDPOINT"), timeout=60)
            # llm = ChatGoogleGenerativeAI(..., client_options=client_options)

            # For ChatGoogleGenerativeAI, timeout is often part of a broader request_options or client_options
            # Let's assume for now it might be within a general options dict or specific param.
            # Re-checking documentation, it seems `request_options` was an attempt, but `client_options` is suggested by the warning.
            # However, `timeout` is usually a direct parameter to `invoke` or part of `generate`.
            # Let's try setting it via `client_options` if that's the intended way for overall client config,
            # or ensure it's applied at the `invoke` call if more specific.
            # The warning suggests 'client_options'. Let's assume it's for client-wide settings.
            # Actual timeout for a specific call is often in `llm.invoke(prompt, config={'request_timeout': 60})`

            # Correcting based on the warning and typical usage for client-level settings:
            current_client_options = {}
            api_endpoint = os.getenv("GOOGLE_API_ENDPOINT")
            if api_endpoint:
                current_client_options['api_endpoint'] = api_endpoint
            # The timeout for the specific call is better handled in `invoke` or `generate` methods if available.
            # If `ChatGoogleGenerativeAI` itself takes a default timeout in `client_options`, it would be specified in its constructor.
            # The previous `request_options={'timeout': 60}` was incorrect.
            # Let's remove it from constructor and rely on `invoke` if needed, or assume default for now if not directly supported in constructor this way.

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=temperature,
                client_options=current_client_options if current_client_options else None
            )

            logger.info(f"Generating blog post with Gemini. Topic: {topic}, Tone: {tone}, Word Count: {target_word_count}")
            
            response = llm.invoke(prompt, config={"request_timeout": 60}) # Pass timeout in invoke config
            response_content = response.content if hasattr(response, 'content') else str(response)

            # Attempt to extract JSON object if wrapped or has extraneous text
            # Standard cleaning for backticks
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()
            elif response_content.strip().startswith("```"):
                response_content = response_content.strip()[3:-3].strip()
            
            # More robust extraction of the JSON object itself
            # This helps if there's any leading/trailing non-JSON text not caught by strip()
            # or if the LLM doesn't perfectly adhere to "ONLY JSON"
            json_match = re.search(r'\{[\s\S]*\}', response_content)
            if json_match:
                json_to_parse = json_match.group(0)
            else:
                # If no clear JSON object is found, this will likely fail parsing, which is handled below.
                json_to_parse = response_content 

            blog_data = json.loads(json_to_parse)

            if not all(k in blog_data for k in ["title", "meta_description", "content"]):
                logger.error(f"Gemini response for blog post missing required keys. Parsed data: {blog_data}")
                return {"error": "Gemini response for blog post was valid JSON but missed required keys (title, meta_description, content)."}

            filename = f"blog_{topic.replace(' ', '_').lower()[:50]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            file_path = os.path.join(self.output_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {blog_data['title']}\\n\\n")
                f.write(f"**Meta Description:** {blog_data['meta_description']}\\n\\n")
                f.write(blog_data['content'])
            
            logger.info(f"Blog post '{blog_data['title']}' successfully generated and saved to {file_path}")
            return {"title": blog_data['title'], "filepath": file_path, "content_preview": blog_data['content'][:200] + "..."} # Return preview, not full content

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response for blog post: {e}. Response snippet: {response_content[:1000]}...") # Log more for debugging
            # Return a clear error structure that the agent can check
            return {"error": f"Gemini response was not valid JSON: {e}", "raw_response_on_error": response_content}
        except Exception as e:
            logger.error(f"Error generating blog post: {str(e)}", exc_info=True)
            return {"error": f"An unexpected error occurred during blog post generation: {str(e)}"}

    @track_api_usage("gemini")
    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=3, delay=2.0, backoff_factor=2.0)
    def generate_twitter_thread(self, topic: str, base_content: Optional[str] = None, num_tweets: int = 5, tone: str = "engaging") -> Dict[str, Any]:
        """Generate a Twitter thread using Gemini.

        Args:
            topic: The main topic of the Twitter thread.
            base_content: Optional base content (e.g., a summary of a blog post or video script) to expand into a thread.
            num_tweets: The desired number of tweets in the thread (approximate).
            tone: The desired tone for the tweets (e.g., informative, witty, engaging).

        Returns:
            Dictionary with the thread title, a list of tweets, and a filepath to the saved thread.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not topic and not base_content:
            logger.warning("generate_twitter_thread called with no topic or base_content.")
            return {"title": "", "tweets": [], "filepath": "", "error": "Topic or base content must be provided."}

        num_tweets = max(2, min(15, num_tweets)) # Keep thread length reasonable

        content_prompt_part = f"The main topic is: '{topic}'."
        if base_content:
            # Ensure base_content is properly escaped if it contains quotes or special characters for the f-string
            escaped_base_content = base_content.replace('"' , '\\"').replace('\n', '\\n')
            content_prompt_part = f"The thread should be based on the following content: \n\"{escaped_base_content}\"\nIt should expand on the topic: '{topic}'."

        prompt = f"""Create a Twitter thread with approximately {num_tweets} tweets.
{content_prompt_part}
Tone: {tone}
Each tweet should be engaging and concise (under 280 characters).
The first tweet should be a strong hook.
The thread should have a clear narrative flow.
Use relevant hashtags where appropriate.

Format your response as a JSON object with the following structure:
{{
    "title": "A concise title for the Twitter thread (e.g., for internal reference)",
    "tweets": [
        "Tweet 1 text... #hashtag1",
        "Tweet 2 text... #hashtag2",
        "Tweet 3 text... (1/3)",
        "Tweet 4 text... (2/3)",
        "Tweet 5 text... (3/3) #FinalThoughts"
    ]
}}

Ensure each tweet in the 'tweets' list is a string.
Respond ONLY with the JSON object, no other text before or after it.
"""

        gemini_api_key = self.api_keys.get("api_key")
        model_name = self.api_keys.get("model_name", "gemini-pro")
        temperature = float(self.api_keys.get("temperature", 0.75)) # Slightly higher for creative tweets

        if not gemini_api_key:
            logger.error("Missing Gemini API key for Twitter thread generation.")
            raise ValueError("Missing Gemini API key for Twitter thread generation")

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=temperature,
                client_options={'api_endpoint': os.getenv("GOOGLE_API_ENDPOINT")}, # Corrected: Added client_options for potential endpoint override and prepared for timeout
                # Add timeout directly if supported or via client_options if that's the way
            )
            # If timeout needs to be passed differently, adjust here. Example for some Google clients:
            # from google.api_core.client_options import ClientOptions
            # client_options = ClientOptions(api_endpoint=os.getenv("GOOGLE_API_ENDPOINT"), timeout=60)
            # llm = ChatGoogleGenerativeAI(..., client_options=client_options)

            # For ChatGoogleGenerativeAI, timeout is often part of a broader request_options or client_options
            # Let's assume for now it might be within a general options dict or specific param.
            # Re-checking documentation, it seems `request_options` was an attempt, but `client_options` is suggested by the warning.
            # However, `timeout` is usually a direct parameter to `invoke` or part of `generate`.
            # Let's try setting it via `client_options` if that's the intended way for overall client config,
            # or ensure it's applied at the `invoke` call if more specific.
            # The warning suggests 'client_options'. Let's assume it's for client-wide settings.
            # Actual timeout for a specific call is often in `llm.invoke(prompt, config={'request_timeout': 60})`

            # Correcting based on the warning and typical usage for client-level settings:
            current_client_options = {}
            api_endpoint = os.getenv("GOOGLE_API_ENDPOINT")
            if api_endpoint:
                current_client_options['api_endpoint'] = api_endpoint

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=temperature,
                client_options=current_client_options if current_client_options else None
            )

            logger.info(f"Generating Twitter thread. Topic: {topic}, Num Tweets: {num_tweets}, Tone: {tone}")
            response = llm.invoke(prompt, config={"request_timeout": 60}) # Pass timeout in invoke config

            response_content = ""
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                logger.warning("Gemini response object does not have 'content' attribute for Twitter thread, attempting to cast to string.")
                response_content = str(response)

            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()
            elif response_content.strip().startswith("```"):
                response_content = response_content.strip()[3:-3].strip()

            thread_data = json.loads(response_content)

            if not isinstance(thread_data, dict) or not all(k in thread_data for k in ["title", "tweets"]) or not isinstance(thread_data["tweets"], list):
                logger.error(f"Gemini response for Twitter thread missing required keys or incorrect format. Response: {response_content}")
                raise ValueError("Invalid JSON structure from Gemini response for Twitter thread.")
            
            logger.info(f"Successfully generated Twitter thread for topic: {topic}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twitter_thread_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Title: {thread_data.get('title', 'Generated Twitter Thread')}\n\n")
                    for i, tweet_text in enumerate(thread_data.get("tweets", [])):
                        f.write(f"Tweet {i+1}:\n{tweet_text}\n\n")
                thread_data["filepath"] = filepath
            except IOError as e:
                logger.error(f"Failed to write Twitter thread file: {str(e)}")
                thread_data["filepath"] = ""

            return thread_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response for Twitter thread: {e}. Response: {response_content}")
            return {"title": "", "tweets": [], "filepath": "", "error": f"Could not parse Twitter thread from AI response. Details: {str(e)}"}
        except Exception as e:
            logger.error(f"Error during Gemini Twitter thread generation: {str(e)}")
            return {"title": "", "tweets": [], "filepath": "", "error": f"An unexpected error occurred. Details: {str(e)}"}

    @track_api_usage("arxiv")
    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    def search_arxiv(self, query: str, max_docs: int = 2, doc_content_chars_max: int = 2000) -> Dict[str, Any]:
        """Search Arxiv for research papers.

        Args:
            query: Search query.
            max_docs: Maximum number of documents to return.
            doc_content_chars_max: Max characters for each document summary.

        Returns:
            Dictionary with search results.
        """
        if not query:
            raise ValueError("Search query cannot be empty for Arxiv search")

        logger.info(f"Searching Arxiv for: {query}")
        try:
            arxiv_wrapper = ArxivAPIWrapper(
                top_k_results=max_docs,
                doc_content_chars_max=doc_content_chars_max,
                load_max_docs=max_docs,  # Ensure we load enough to pick from
                load_all_available_meta=False  # Avoids fetching unnecessary metadata
            )
            arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

            raw_results = arxiv_tool.run(query)

            processed_results = []
            docs: List[Document] = arxiv_wrapper.load(query)

            for doc in docs[:max_docs]:
                processed_results.append({
                    "title": doc.metadata.get("Title", "N/A"),
                    "authors": ", ".join(doc.metadata.get("Authors", [])),
                    "published_date": str(doc.metadata.get("Published", "N/A")),
                    "summary": doc.page_content,
                    "pdf_url": doc.metadata.get("entry_id", "").replace("abs", "pdf")  # Construct PDF URL
                })

            if not processed_results:
                logger.warning(f"No Arxiv papers found for query: {query}")
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No papers found for the query on Arxiv."
                }

            # Save research to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_research_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Arxiv Research: {query}\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for i, paper in enumerate(processed_results, 1):
                        f.write(f"{i}. Title: {paper.get('title', 'N/A')}\n")
                        f.write(f"   Authors: {paper.get('authors', 'N/A')}\n")
                        f.write(f"   Published: {paper.get('published_date', 'N/A')}\n")
                        f.write(f"   Summary: {paper.get('summary', 'N/A')[:500]}...\n")  # Truncate summary for brevity
                        f.write(f"   PDF URL: {paper.get('pdf_url', 'N/A')}\n\n")
            except IOError as e:
                logger.error(f"Failed to write Arxiv research file: {str(e)}")

            return {
                "success": True,
                "query": query,
                "results": processed_results,
                "count": len(processed_results),
                "filepath": filepath
            }

        except ImportError:
            logger.error("Arxiv/LangChain community libraries not installed.")
            raise ImportError("The 'arxiv' and 'langchain-community' libraries are required for Arxiv research. Please install them.")
        except Exception as e:
            logger.error(f"Arxiv research error: {str(e)}\n{traceback.format_exc()}")
            raise

    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    @track_api_usage("wikipedia")
    def search_wikipedia(self, query: str, max_docs: int = 2, doc_content_chars_max: int = 2000) -> Dict[str, Any]:
        """Search Wikipedia for articles.

        Args:
            query: Search query.
            max_docs: Maximum number of documents (summaries) to return.
            doc_content_chars_max: Max characters for each document summary.

        Returns:
            Dictionary with search results.
        """
        if not query:
            raise ValueError("Search query cannot be empty for Wikipedia search")

        logger.info(f"Searching Wikipedia for: {query}")
        try:
            wiki_wrapper = WikipediaAPIWrapper(
                top_k_results=max_docs,  # How many search results to consider
                doc_content_chars_max=doc_content_chars_max,
                load_all_available_meta=False
            )
            wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

            docs: List[Document] = wiki_wrapper.load(query)

            processed_results = []
            for doc in docs[:max_docs]:
                processed_results.append({
                    "title": doc.metadata.get("title", "N/A"),  # WikipediaAPIWrapper uses 'title'
                    "summary": doc.page_content,
                    "url": doc.metadata.get("source", "")  # WikipediaAPIWrapper uses 'source' for URL
                })

            if not processed_results:
                logger.warning(f"No Wikipedia articles found for query: {query}")
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No articles found for the query on Wikipedia."
                }

            # Save research to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wikipedia_research_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Wikipedia Research: {query}\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for i, article in enumerate(processed_results, 1):
                        f.write(f"{i}. Title: {article.get('title', 'N/A')}\n")
                        f.write(f"   Summary: {article.get('summary', 'N/A')[:500]}...\n")  # Truncate summary
                        f.write(f"   URL: {article.get('url', 'N/A')}\n\n")
            except IOError as e:
                logger.error(f"Failed to write Wikipedia research file: {str(e)}")

            return {
                "success": True,
                "query": query,
                "results": processed_results,
                "count": len(processed_results),
                "filepath": filepath
            }
        except ImportError:
            logger.error("Wikipedia/LangChain community libraries not installed.")
            raise ImportError("The 'wikipedia' and 'langchain-community' libraries are required for Wikipedia research. Please install them.")
        except Exception as e:
            logger.error(f"Wikipedia research error: {str(e)}\n{traceback.format_exc()}")
            raise

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format: HH:MM:SS,mmm.

        Args:
            seconds: Time in seconds

        Returns:
            String in SRT timestamp format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _generate_srt_subtitles(self, words_with_timestamps: List[Dict[str, Any]], chunk_size: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate SRT formatted subtitles from word timestamps.

        Args:
            words_with_timestamps: List of words with start and end timestamps
            chunk_size: Number of words per line

        Returns:
            Tuple with (SRT formatted string, list of line dictionaries)
        """
        if not words_with_timestamps:
            return "", []

        # Group words into chunks
        lines = []
        current_line = []
        current_line_text = ""
        line_counter = 1

        for word in words_with_timestamps:
            current_line.append(word)
            if len(current_line) >= chunk_size:
                # Get the start time of the first word and end time of the last word
                start_time = current_line[0]["start"]
                end_time = current_line[-1]["end"]

                # Format the text
                line_text = " ".join([w["text"] for w in current_line])

                # Add to lines
                lines.append({
                    "line": str(line_counter),
                    "start": self._format_timestamp(start_time),
                    "end": self._format_timestamp(end_time),
                    "text": line_text
                })

                line_counter += 1
                current_line = []

        # Handle any remaining words
        if current_line:
            start_time = current_line[0]["start"]
            end_time = current_line[-1]["end"]
            line_text = " ".join([w["text"] for w in current_line])

            lines.append({
                "line": str(line_counter),
                "start": self._format_timestamp(start_time),
                "end": self._format_timestamp(end_time),
                "text": line_text
            })

        # Generate SRT formatted text
        srt_content = ""
        for line in lines:
            srt_content += f"{line['line']}\n"
            srt_content += f"{line['start']} --> {line['end']}\n"
            srt_content += f"{line['text']}\n\n"

        return srt_content, lines
    @graceful_degradation
    @resilient_api_call(api_name="transcription", max_retries=3, initial_backoff=2.0)
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    @track_api_usage("assemblyai")  # This will be dynamically changed to "deepgram" in the method if needed
    def transcribe_audio(self, audio_filepath: str, provider: str = "assemblyai") -> Dict[str, Any]:
        """Transcribe audio to text using a provider like AssemblyAI or Deepgram.

        Args:
            audio_filepath: Path to the audio file.
            provider: 'assemblyai' or 'deepgram' (or others you might implement).

        Returns:
            Dictionary with transcription result, including text and subtitles.
        """
        # Override the decorator's default API name with the actual provider being used
        # This will be picked up by the track_api_usage decorator's wrapper
        api_provider = provider.lower()
        if hasattr(track_api_usage, "override_api_name"):
            track_api_usage.override_api_name = api_provider

        if not os.path.exists(audio_filepath):
            logger.error(f"Audio file not found for transcription: {audio_filepath}")
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

        logger.info(f"Starting audio transcription for {audio_filepath} using {provider}")
        transcribed_text = ""
        words_with_timestamps = []

        # Initialize timestamps for SRT generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcription_filename = f"transcription_{timestamp}.txt"
        transcription_filepath = os.path.join(self.output_dir, transcription_filename)
        subtitles_filename = f"subtitles_{timestamp}.srt"
        subtitles_filepath = os.path.join(self.output_dir, subtitles_filename)

        # Provider-specific transcription logic
        if provider.lower() == "assemblyai":
            api_key = self.api_keys.get("assemblyai")
            if not api_key:
                logger.error("AssemblyAI API key not found.")
                raise ValueError("AssemblyAI API key is missing. Please check your .env and dashboard settings.")

            # AssemblyAI API implementation
            try:
                # Set up API headers
                headers = {
                    "authorization": api_key,
                    "content-type": "application/json"
                }

                logger.info(f"Uploading audio file to AssemblyAI: {audio_filepath}")

                # Step 1: Upload the audio file
                with open(audio_filepath, "rb") as audio_file:
                    upload_url = "https://api.assemblyai.com/v2/upload"
                    response = requests.post(upload_url, headers=headers, data=audio_file, timeout=90)

                    if response.status_code != 200:
                        logger.error(f"Failed to upload audio: {response.status_code} - {response.text}")
                        raise RuntimeError(f"AssemblyAI upload failed: {response.status_code}")

                    upload_data = response.json()
                    audio_url = upload_data["upload_url"]

                # Step 2: Submit the transcription job
                transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
                job_data = {
                    "audio_url": audio_url,
                    "word_boost": ["LangGraph", "content", "AI", "transcription"],  # Boost accuracy for these terms
                    "punctuate": True,
                    "format_text": True,
                    "dual_channel": False,
                    "speaker_labels": False,
                    "word_timestamps": True  # This is crucial for subtitle generation
                }

                response = requests.post(transcript_endpoint, json=job_data, headers=headers, timeout=30)

                if response.status_code != 200:
                    logger.error(f"Failed to submit transcription job: {response.status_code} - {response.text}")
                    raise RuntimeError(f"AssemblyAI transcription job submission failed: {response.status_code}")

                job_data = response.json()
                job_id = job_data["id"]
                logger.info(f"Transcription job submitted with ID: {job_id}")

                # Step 3: Poll for results
                polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{job_id}"
                max_polling_attempts = 60  # Maximum 5 minutes with 5-second intervals
                polling_interval = 5

                for attempt in range(max_polling_attempts):
                    logger.debug(f"Polling for transcription results, attempt {attempt + 1}")
                    response = requests.get(polling_endpoint, headers=headers, timeout=30)

                    if response.status_code != 200:
                        logger.error(f"Failed to poll transcription job: {response.status_code} - {response.text}")
                        raise RuntimeError(f"AssemblyAI polling failed: {response.status_code}")

                    job_status = response.json()

                    if job_status["status"] == "completed":
                        transcribed_text = job_status["text"]

                        # Extract word timestamps for subtitle generation
                        if "words" in job_status:
                            for word in job_status["words"]:
                                words_with_timestamps.append({
                                    "text": word["text"],
                                    "start": word["start"] / 1000.0,  # Convert ms to seconds
                                    "end": word["end"] / 1000.0  # Convert ms to seconds
                                })

                        logger.info("Transcription completed successfully")
                        break
                    elif job_status["status"] == "error":
                        error_msg = job_status.get("error", "Unknown error")
                        logger.error(f"Transcription job failed: {error_msg}")
                        raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")
                    elif job_status["status"] in ["queued", "processing"]:
                        logger.debug(f"Transcription job status: {job_status['status']}")
                        time.sleep(polling_interval)
                    else:
                        logger.warning(f"Unexpected job status: {job_status['status']}")
                        time.sleep(polling_interval)

                if not transcribed_text:
                    logger.error("Transcription timed out")
                    raise RuntimeError("AssemblyAI transcription timed out after maximum polling attempts")

            except Exception as e:
                logger.error(f"AssemblyAI transcription error: {str(e)}", exc_info=True)
                raise

        elif provider.lower() == "deepgram":
            api_key = self.api_keys.get("deepgram")
            if not api_key:
                logger.error("Deepgram API key not found.")
                raise ValueError("Deepgram API key is missing. Please check your .env and dashboard settings.")

            # Deepgram API implementation
            try:
                from deepgram import Deepgram as DG

                # Initialize the Deepgram client
                deepgram = DG(api_key)

                # Open the audio file
                logger.info(f"Opening audio file for Deepgram transcription: {audio_filepath}")
                with open(audio_filepath, "rb") as audio_file:
                    audio_data = audio_file.read()

                # Set transcription options
                options = {
                    "punctuate": True,
                    "model": "general",
                    "language": "en-US",
                    "diarize": False,
                    "smart_format": True,
                    "utterances": False,
                    "numerals": True,
                    "tier": "enhanced"  # Use the enhanced model for better accuracy
                }

                # Submit for transcription
                logger.info("Submitting audio to Deepgram for transcription")
                response = deepgram.transcription.sync_prerecorded(
                    {"buffer": audio_data, "mimetype": f"audio/{os.path.splitext(audio_filepath)[1][1:]}"},
                    options
                )

                # Extract transcription
                if response and "results" in response:
                    # Get full transcript
                    transcribed_text = response["results"]["channels"][0]["alternatives"][0]["transcript"]

                    # Get words with timestamps
                    for word in response["results"]["channels"][0]["alternatives"][0]["words"]:
                        words_with_timestamps.append({
                            "text": word["word"],
                            "start": word["start"],
                            "end": word["end"]
                        })

                    logger.info("Deepgram transcription completed successfully")
                else:
                    logger.error(f"Unexpected Deepgram response structure: {response}")
                    raise RuntimeError(f"Deepgram transcription failed: unexpected response structure")

            except ImportError:
                logger.error("Deepgram Python SDK not installed. Please install it using: pip install deepgram-sdk")
                # Optionally, re-raise the error or return a specific error message
                # For now, let's allow graceful degradation if other parts of the app can function without it.
                # raise ImportError("The 'deepgram' Python module is required for Deepgram transcription. Please install it using: pip install deepgram-sdk")
                # Fallback or error reporting:
                return {
                    "success": False,
                    "provider": provider,
                    "text": "Error: Deepgram SDK not installed.",
                    "filepath": None,
                    "original_audio_path": audio_filepath,
                    "duration_seconds": 0,
                    "subtitles": {"filepath": None, "content": "", "lines": []},
                    "words": []
                }
            except Exception as e:
                logger.error(f"Deepgram transcription error: {str(e)}", exc_info=True)
                raise
        else:
            logger.error(f"Unsupported transcription provider: {provider}")
            raise ValueError(f"Unsupported transcription provider: {provider}")

        # Save transcription to file
        try:
            with open(transcription_filepath, "w", encoding="utf-8") as f:
                f.write(transcribed_text)
            logger.info(f"Transcription saved to {transcription_filepath}")
        except IOError as e:
            logger.error(f"Failed to write transcription file: {str(e)}")

        # Generate and save subtitles if we have word timestamps
        subtitles_content = ""
        subtitle_lines = []
        if words_with_timestamps:
            try:
                subtitles_content, subtitle_lines = self._generate_srt_subtitles(words_with_timestamps)
                with open(subtitles_filepath, "w", encoding="utf-8") as f:
                    f.write(subtitles_content)
                logger.info(f"Subtitles saved to {subtitles_filepath}")
            except IOError as e:
                logger.error(f"Failed to write subtitles file: {str(e)}")
        # Calculate audio duration for analytics
        duration_seconds = 0
        if words_with_timestamps and len(words_with_timestamps) > 0:
            # Use the end time of the last word as the duration
            duration_seconds = words_with_timestamps[-1]["end"]

        return {
            "success": True,
            "provider": provider,
            "text": transcribed_text,
            "filepath": transcription_filepath,
            "original_audio_path": audio_filepath,
            "duration_seconds": duration_seconds,  # Add duration for analytics tracking
            "subtitles": {
                "filepath": subtitles_filepath if subtitles_content else None,
                "content": subtitles_content,
                "lines": subtitle_lines
            },
            "words": words_with_timestamps
        }

    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    @track_api_usage("youtube_data")
    def youtube_research(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Research YouTube for videos related to a query.

        Args:
            query: Search query.
            max_results: Maximum number of videos to return.

        Returns:
            Dictionary with search results and metadata.
        """
        if not query:
            raise ValueError("Search query cannot be empty for YouTube research")

        logger.info(f"Researching YouTube for: {query}")
        try:
            # Get API key
            api_key = self.api_keys.get("youtube_data")
            if not api_key:
                logger.error("YouTube Data API key not found.")
                raise ValueError("YouTube Data API key is missing. Please check your .env file.")

            # YouTube Data API v3 endpoints
            search_url = "https://www.googleapis.com/youtube/v3/search"
            video_url = "https://www.googleapis.com/youtube/v3/videos"

            # Step 1: Search for videos
            search_params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": api_key
            }

            logger.debug(f"YouTube search params: {search_params}")
            search_response = requests.get(search_url, params=search_params, timeout=30)

            if search_response.status_code != 200:
                logger.error(f"YouTube API search error: {search_response.status_code} - {search_response.text}")
                raise RuntimeError(f"YouTube API search error: {search_response.status_code}")

            search_data = search_response.json()
            video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]

            if not video_ids:
                logger.warning(f"No YouTube videos found for query: {query}")
                return {
                    "success": True,
                    "query": query,
                    "videos": [],
                    "message": "No videos found for the query on YouTube."
                }

            # Step 2: Get video details (including statistics)
            video_params = {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(video_ids),
                "key": api_key
            }

            logger.debug(f"YouTube video details params: {video_params}")
            video_response = requests.get(video_url, params=video_params, timeout=30)

            if video_response.status_code != 200:
                logger.error(f"YouTube API video details error: {video_response.status_code} - {video_response.text}")
                raise RuntimeError(f"YouTube API video details error: {video_response.status_code}")

            video_data = video_response.json()
            videos = []

            for item in video_data.get("items", []):
                # Process and transform video data
                video_info = {
                    "id": item["id"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "channelId": item["snippet"]["channelId"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "thumbnailUrl": item["snippet"]["thumbnails"]["high"]["url"],
                    "viewCount": int(item["statistics"].get("viewCount", 0)),
                    "likeCount": int(item["statistics"].get("likeCount", 0)),
                    "commentCount": int(item["statistics"].get("commentCount", 0)),
                    "duration": item["contentDetails"]["duration"]  # ISO 8601 format
                }
                videos.append(video_info)

            # Save research to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_research_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"YouTube Research: {query}\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    for i, video in enumerate(videos, 1):
                        f.write(f"{i}. {video['title']}\n")
                        f.write(f"   Channel: {video['channelTitle']}\n")
                        f.write(f"   Views: {video['viewCount']:,} | Likes: {video['likeCount']:,} | Comments: {video['commentCount']:,}\n")
                        f.write(f"   URL: https://www.youtube.com/watch?v={video['id']}\n")
                        f.write(f"   Description: {video['description'][:200]}...\n\n")  # Truncate description
            except IOError as e:
                logger.error(f"Failed to write YouTube research file: {str(e)}")

            return {
                "success": True,
                "query": query,
                "videos": videos,
                "count": len(videos),
                "filepath": filepath
            }

        except Exception as e:
            logger.error(f"YouTube research error: {str(e)}", exc_info=True)
            return ErrorHandler.format_error_response(e, context={"query": query})    @track_api_usage("pexels")  # Will be overridden based on actual provider used
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    def search_images(self, query: str, source: str = "pexels", count: int = 5) -> List[Dict[str, Any]]:
        """Search for stock images using Pexels or Pixabay APIs.

        Args:
            query: Search query for images
            source: Image provider ('pexels' or 'pixabay')
            count: Number of images to return (default: 5)

        Returns:
            List of image dictionaries with metadata
        """
        if not query:
            raise ValueError("Search query cannot be empty for image search")

        logger.info(f"Searching for images: query='{query}', source='{source}', count={count}")

        try:
            # Import API classes from tools
            from tools import PexelsAPI, PixabayAPI

            source = source.lower()

            if source == "pexels":
                api_key = self.api_keys.get("pexels")
                if not api_key:
                    raise ValueError("Pexels API key not found. Please check your .env file.")

                pexels_api = PexelsAPI(api_key)
                raw_images = pexels_api.search_images(query=query, per_page=count)

                # Transform Pexels response to standardized format
                images = []
                for img in raw_images:
                    images.append({
                        "id": img.get("id"),
                        "url": img.get("url", ""),
                        "src": img.get("src", {}).get("large", img.get("src", {}).get("original", "")),
                        "photographer": img.get("photographer", "Unknown"),
                        "photographer_url": img.get("photographer_url", ""),
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                        "type": "photo",
                        "provider": "pexels"
                    })

            elif source == "pixabay":
                api_key = self.api_keys.get("pixabay")
                if not api_key:
                    raise ValueError("Pixabay API key not found. Please check your .env file.")

                pixabay_api = PixabayAPI(api_key)
                raw_images = pixabay_api.search_images(query=query, per_page=count)

                # Transform Pixabay response to standardized format
                images = []
                for img in raw_images:
                    images.append({
                        "id": img.get("id"),
                        "url": f"https://pixabay.com/photos/{img.get('id', '')}/",
                        "src": img.get("largeImageURL", img.get("webformatURL", "")),
                        "photographer": img.get("user", "Unknown"),
                        "photographer_url": f"https://pixabay.com/users/{img.get('user', '')}/",
                        "width": img.get("imageWidth", 0),
                        "height": img.get("imageHeight", 0),
                        "type": "photo",
                        "provider": "pixabay"
                    })

            else:
                raise ValueError(f"Unsupported image source: {source}. Use 'pexels' or 'pixabay'.")

            logger.info(f"Found {len(images)} images from {source}")
            return images

        except Exception as e:
            logger.error(f"Image search error: {str(e)}", exc_info=True)
            raise

    @track_api_usage("pexels")  # Will be overridden based on actual provider used
    @graceful_degradation
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    def search_stock_media(self, query: str, media_type: str = "photo", provider: str = "pexels", count: int = 5) -> Dict[str, Any]:
        """Search for stock media (photos or videos) using Pexels or Pixabay APIs.

        Args:
            query: Search query for media
            media_type: Type of media ('photo' or 'video')
            provider: Media provider ('pexels' or 'pixabay')
            count: Number of media items to return (default: 5)

        Returns:
            Dictionary with search results and metadata
        """
        if not query:
            raise ValueError("Search query cannot be empty for stock media search")

        logger.info(f"Searching for stock media: query='{query}', type='{media_type}', provider='{provider}', count={count}")

        try:
            # Import API classes from tools
            from tools import PexelsAPI, PixabayAPI

            provider = provider.lower()
            media_type = media_type.lower()

            results = []

            if provider == "pexels":
                api_key = self.api_keys.get("pexels")
                if not api_key:
                    raise ValueError("Pexels API key not found. Please check your .env file.")

                pexels_api = PexelsAPI(api_key)

                if media_type == "photo":
                    raw_media = pexels_api.search_images(query=query, per_page=count)

                    # Transform Pexels photos response
                    for media in raw_media:
                        results.append({
                            "id": media.get("id"),
                            "url": media.get("url", ""),
                            "src": media.get("src", {}).get("large", media.get("src", {}).get("original", "")),
                            "photographer": media.get("photographer", "Unknown"),
                            "photographer_url": media.get("photographer_url", ""),
                            "width": media.get("width", 0),
                            "height": media.get("height", 0),
                            "type": "photo",
                            "provider": "pexels"
                        })

                elif media_type == "video":
                    raw_media = pexels_api.search_videos(query=query, per_page=count)

                    # Transform Pexels videos response
                    for media in raw_media:
                        video_files = media.get("video_files", [])
                        video_src = video_files[0].get("link", "") if video_files else ""

                        results.append({
                            "id": media.get("id"),
                            "url": media.get("url", ""),
                            "src": video_src,
                            "user": media.get("user", {}).get("name", "Unknown"),
                            "duration": media.get("duration", 0),
                            "width": media.get("width", 0),
                            "height": media.get("height", 0),
                            "type": "video",
                            "provider": "pexels"
                        })
                else:
                    raise ValueError(f"Unsupported media type: {media_type}. Use 'photo' or 'video'.")

            elif provider == "pixabay":
                api_key = self.api_keys.get("pixabay")
                if not api_key:
                    raise ValueError("Pixabay API key not found. Please check your .env file.")

                pixabay_api = PixabayAPI(api_key)

                if media_type == "photo":
                    raw_media = pixabay_api.search_images(query=query, per_page=count)

                    # Transform Pixabay photos response
                    for media in raw_media:
                        results.append({
                            "id": media.get("id"),
                            "url": f"https://pixabay.com/photos/{media.get('id', '')}/",
                            "src": media.get("largeImageURL", media.get("webformatURL", "")),
                            "photographer": media.get("user", "Unknown"),
                            "photographer_url": f"https://pixabay.com/users/{media.get('user', '')}/",
                            "width": media.get("imageWidth", 0),
                            "height": media.get("imageHeight", 0),
                            "type": "photo",
                            "provider": "pixabay"
                        })

                elif media_type == "video":
                    raw_media = pixabay_api.search_videos(query=query, per_page=count)

                    # Transform Pixabay videos response
                    for media in raw_media:
                        videos = media.get("videos", {})
                        video_src = videos.get("small", {}).get("url", "") if videos else ""

                        results.append({
                            "id": media.get("id"),
                            "url": f"https://pixabay.com/videos/{media.get('id', '')}/",
                            "src": video_src,
                            "user": media.get("user", "Unknown"),
                            "duration": media.get("duration", 0),
                            "width": media.get("imageWidth", 0),
                            "height": media.get("imageHeight", 0),
                            "type": "video",
                            "provider": "pixabay"
                        })
                else:
                    raise ValueError(f"Unsupported media type: {media_type}. Use 'photo' or 'video'.")

            else:
                raise ValueError(f"Unsupported provider: {provider}. Use 'pexels' or 'pixabay'.")

            logger.info(f"Found {len(results)} {media_type} items from {provider}")

            return {
                "success": True,
                "query": query,
                "media_type": media_type,                "provider": provider,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Stock media search error: {str(e)}", exc_info=True)
            return ErrorHandler.format_error_response(e, context={"query": query, "media_type": media_type, "provider": provider})

    @track_api_usage("tavily")
    @ErrorHandler.with_retry(max_retries=2, delay=1.5)
    def search_web(self, query: str) -> str:
        """Search the web using Tavily API.

        Args:
            query: Search query for web search

        Returns:
            Formatted web search results as string
        """
        if not query:
            raise ValueError("Search query cannot be empty for web search")

        logger.info(f"Searching web for: {query}")

        try:
            # Import the search_web tool from tools module
            from tools import search_web

            # Use the existing search_web tool which handles Tavily API integration
            result = search_web.invoke(query)

            # Handle different response formats
            if isinstance(result, dict):
                # If the result is a dictionary with analytics data
                search_response = result.get("response", str(result))
            else:
                # If the result is a string
                search_response = str(result)

            logger.info(f"Web search completed for query: {query}")
            return search_response

        except Exception as e:
            logger.error(f"Web search error: {str(e)}", exc_info=True)
            raise

def get_content_creator(api_keys: Dict[str, str]) -> ContentCreator:
    """Get a ContentCreator instance.

    Args:
        api_keys: Dictionary of API keys

    Returns:
        ContentCreator instance
    """
    return ContentCreator(api_keys)
