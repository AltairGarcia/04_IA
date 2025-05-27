\
import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from content_creation import ContentCreator

# Test data
SAMPLE_TOPIC = "Test Topic"
SAMPLE_TONE = "Informative"
EXPECTED_TITLE = "Test Blog Post Title"
EXPECTED_META_DESC = "This is a test meta description."
EXPECTED_CONTENT_MARKDOWN = "## Test Heading\\n\\nThis is test paragraph content."
EXPECTED_KEYWORDS = ["test", "blog"]

MOCK_GEMINI_RESPONSE_SUCCESS = {
    "title": EXPECTED_TITLE,
    "meta_description": EXPECTED_META_DESC,
    "content": EXPECTED_CONTENT_MARKDOWN
}

MOCK_API_KEYS = {
    "api_key": "test_gemini_key",
    "model_name": "gemini-test-model",
    "temperature": 0.5
}

@pytest.fixture
def content_creator_instance(tmp_path):
    """Fixture to create a ContentCreator instance with a temporary output directory."""
    creator = ContentCreator(api_keys=MOCK_API_KEYS)
    # Override the default output_dir to use the pytest tmp_path fixture
    creator.output_dir = str(tmp_path) 
    # Ensure the mocked output_dir exists, as the original __init__ would do
    os.makedirs(creator.output_dir, exist_ok=True)
    return creator

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_blog_post_success(mock_chat_google_genai, content_creator_instance, tmp_path):
    """Test successful blog post generation."""
    # Configure the mock LLM and its response
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content=json.dumps(MOCK_GEMINI_RESPONSE_SUCCESS))
    mock_chat_google_genai.return_value = mock_llm_instance

    # Call the method to test
    result = content_creator_instance.generate_blog_post(
        topic=SAMPLE_TOPIC,
        tone=SAMPLE_TONE,
        target_word_count=100,
        keywords=EXPECTED_KEYWORDS
    )

    # Assertions
    assert result is not None
    assert "error" not in result, f"Expected no error, but got: {result.get('error')}"
    assert result.get("title") == EXPECTED_TITLE
    assert "filepath" in result
    assert result.get("content_preview") == EXPECTED_CONTENT_MARKDOWN[:200] + "..."

    # Check if the file was created in the temp directory
    expected_filename_start = f"blog_{SAMPLE_TOPIC.replace(' ', '_').lower()}"
    output_files = os.listdir(tmp_path)
    assert any(f.startswith(expected_filename_start) and f.endswith(".md") for f in output_files)
    
    if output_files:
        filepath = os.path.join(tmp_path, output_files[0]) # Assuming one file is created
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        assert f"# {EXPECTED_TITLE}" in file_content
        assert f"**Meta Description:** {EXPECTED_META_DESC}" in file_content
        assert EXPECTED_CONTENT_MARKDOWN in file_content

    # Verify that the LLM was called with a prompt containing the topic, tone, word count, and keywords
    mock_llm_instance.invoke.assert_called_once()
    called_prompt = mock_llm_instance.invoke.call_args[0][0]
    assert SAMPLE_TOPIC in called_prompt
    assert SAMPLE_TONE in called_prompt
    assert "100 words" in called_prompt # target_word_count
    assert ", ".join(EXPECTED_KEYWORDS) in called_prompt
    assert '\"title\":' in called_prompt # Check for JSON structure in prompt
    assert '\"meta_description\":' in called_prompt
    assert '\"content\":' in called_prompt

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_blog_post_json_decode_error(mock_chat_google_genai, content_creator_instance):
    """Test blog post generation when Gemini returns invalid JSON."""
    mock_llm_instance = MagicMock()
    invalid_json_response = "This is not JSON { definitely not json"
    mock_llm_instance.invoke.return_value = MagicMock(content=invalid_json_response)
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_blog_post(topic="Test JSON Error", tone="any")

    assert "error" in result
    assert "Gemini response was not valid JSON" in result["error"]
    assert result.get("raw_response_on_error") == invalid_json_response
    assert "filepath" not in result # Should not attempt to create a file

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_blog_post_missing_keys_in_json(mock_chat_google_genai, content_creator_instance):
    """Test blog post generation when Gemini returns valid JSON but with missing keys."""
    mock_llm_instance = MagicMock()
    json_missing_keys = {"title": "Only Title", "some_other_key": "value"}
    mock_llm_instance.invoke.return_value = MagicMock(content=json.dumps(json_missing_keys))
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_blog_post(topic="Test Missing Keys", tone="any")

    assert "error" in result
    assert "missed required keys" in result["error"]
    assert "filepath" not in result

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_blog_post_empty_topic(mock_chat_google_genai, content_creator_instance):
    """Test blog post generation with an empty topic."""
    # LLM should not even be called if validation fails first
    
    result = content_creator_instance.generate_blog_post(topic="", tone="any")

    assert "content" in result and "Error: Topic cannot be empty." in result["content"]
    assert result.get("title") == ""
    assert result.get("meta_description") == ""
    assert result.get("filepath") == ""
    mock_chat_google_genai.assert_not_called() # Ensure LLM is not called

def test_generate_blog_post_no_api_key(tmp_path):
    """Test blog post generation when Gemini API key is missing."""
    creator_no_key = ContentCreator(api_keys={}) # No "api_key"
    creator_no_key.output_dir = str(tmp_path)
    os.makedirs(creator_no_key.output_dir, exist_ok=True)

    result = creator_no_key.generate_blog_post(topic="Test No Key", tone="any")
    
    assert "error" in result
    assert "Missing Gemini API key" in result["error"]

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_blog_post_llm_api_error(mock_chat_google_genai, content_creator_instance):
    """Test blog post generation when the LLM call raises an exception."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Simulated API Error")
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_blog_post(topic="Test API Error", tone="any")

    assert "error" in result
    assert "Error generating blog post content: Simulated API Error" in result["error"]
    assert "filepath" not in result

@patch('content_creation.os.makedirs')
@patch('content_creation.open', new_callable=mock_open)
@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_blog_post_file_write_error(mock_chat_google_genai, mock_file_open, mock_makedirs, content_creator_instance, tmp_path):
    """Test blog post generation when writing the file fails."""
    # Configure the mock LLM and its response
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content=json.dumps(MOCK_GEMINI_RESPONSE_SUCCESS))
    mock_chat_google_genai.return_value = mock_llm_instance

    # Simulate an error during file open/write
    mock_file_open.side_effect = IOError("Simulated File Write Error")
    
    # Ensure output_dir is set to tmp_path for this test instance
    content_creator_instance.output_dir = str(tmp_path)
    mock_makedirs.return_value = None # Or configure as needed if it's called again

    result = content_creator_instance.generate_blog_post(
        topic=SAMPLE_TOPIC,
        tone=SAMPLE_TONE
    )

    assert "error" in result
    assert "Failed to write blog post to file: Simulated File Write Error" in result["error"]
    assert "filepath" in result # Filepath might still be generated before the write attempt
    assert result["filepath"].startswith(str(tmp_path)) # Check it's in the correct dir

    # Verify that an attempt was made to open the file
    mock_file_open.assert_called_once()
    args, _ = mock_file_open.call_args
    assert args[0].startswith(os.path.join(str(tmp_path), f"blog_{SAMPLE_TOPIC.replace(' ', '_').lower()}"))
    assert args[1] == "w"


# --- Tests for generate_twitter_thread ---

EXPECTED_TWITTER_THREAD_MARKDOWN = "1/ Test tweet 1.\\\\n2/ Test tweet 2 with #TestHashtag."
MOCK_GEMINI_TWITTER_RESPONSE_SUCCESS = {
    "thread_content": EXPECTED_TWITTER_THREAD_MARKDOWN
}

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_twitter_thread_success(mock_chat_google_genai, content_creator_instance, tmp_path):
    """Test successful Twitter thread generation."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content=json.dumps(MOCK_GEMINI_TWITTER_RESPONSE_SUCCESS))
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_twitter_thread(
        topic="Test Twitter Topic",
        tone="Engaging",
        platform="Twitter",
        length="short"
    )

    assert result is not None
    assert "error" not in result, f"Expected no error, but got: {result.get('error')}"
    assert "filepath" in result
    assert result.get("content_preview") == EXPECTED_TWITTER_THREAD_MARKDOWN[:200] + ("..." if len(EXPECTED_TWITTER_THREAD_MARKDOWN) > 200 else "")

    expected_filename_start = "twitter_thread_Test_Twitter_Topic".replace(' ', '_').lower()
    output_files = os.listdir(tmp_path)
    # Corrected filename check for twitter threads
    assert any(f.startswith("twitter_thread_test_twitter_topic") and f.endswith(".md") for f in output_files), f"Files found: {output_files}"
    
    if output_files:
        # Find the correct file
        correct_file = next((f for f in output_files if f.startswith("twitter_thread_test_twitter_topic") and f.endswith(".md")), None)
        assert correct_file is not None, "Twitter thread file not found"
        filepath = os.path.join(tmp_path, correct_file)
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        assert EXPECTED_TWITTER_THREAD_MARKDOWN in file_content

    mock_llm_instance.invoke.assert_called_once()
    called_prompt = mock_llm_instance.invoke.call_args[0][0]
    assert "Test Twitter Topic" in called_prompt
    assert "Engaging" in called_prompt
    assert "Twitter" in called_prompt
    assert "short" in called_prompt
    assert '\\\"thread_content\\\":' in called_prompt

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_twitter_thread_json_decode_error(mock_chat_google_genai, content_creator_instance):
    """Test Twitter thread generation when Gemini returns invalid JSON."""
    mock_llm_instance = MagicMock()
    invalid_json_response = "Not a valid JSON response for Twitter"
    mock_llm_instance.invoke.return_value = MagicMock(content=invalid_json_response)
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_twitter_thread(topic="Test Twitter JSON Error", tone="any")

    assert "error" in result
    assert "Gemini response was not valid JSON" in result["error"]
    assert result.get("raw_response_on_error") == invalid_json_response
    assert "filepath" not in result

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_twitter_thread_missing_keys_in_json(mock_chat_google_genai, content_creator_instance):
    """Test Twitter thread generation with missing 'thread_content' key in JSON."""
    mock_llm_instance = MagicMock()
    json_missing_keys = {"some_other_data": "value"} # Missing 'thread_content'
    mock_llm_instance.invoke.return_value = MagicMock(content=json.dumps(json_missing_keys))
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_twitter_thread(topic="Test Twitter Missing Keys", tone="any")

    assert "error" in result
    assert "'thread_content' key is missing" in result["error"]
    assert "filepath" not in result

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_twitter_thread_empty_topic(mock_chat_google_genai, content_creator_instance):
    """Test Twitter thread generation with an empty topic."""
    result = content_creator_instance.generate_twitter_thread(topic="", tone="any")

    assert "content" in result and "Error: Topic cannot be empty." in result["content"]
    assert result.get("filepath") == ""
    mock_chat_google_genai.assert_not_called()

def test_generate_twitter_thread_no_api_key(tmp_path):
    """Test Twitter thread generation when Gemini API key is missing."""
    creator_no_key = ContentCreator(api_keys={})
    creator_no_key.output_dir = str(tmp_path)
    os.makedirs(creator_no_key.output_dir, exist_ok=True)

    result = creator_no_key.generate_twitter_thread(topic="Test Twitter No Key", tone="any")
    
    assert "error" in result
    assert "Missing Gemini API key" in result["error"]

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_twitter_thread_llm_api_error(mock_chat_google_genai, content_creator_instance):
    """Test Twitter thread generation when the LLM call raises an exception."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Simulated Twitter API Error")
    mock_chat_google_genai.return_value = mock_llm_instance

    result = content_creator_instance.generate_twitter_thread(topic="Test Twitter API Error", tone="any")

    assert "error" in result
    assert "Error generating Twitter thread content: Simulated Twitter API Error" in result["error"]
    assert "filepath" not in result

@patch('content_creation.os.makedirs')
@patch('content_creation.open', new_callable=mock_open)
@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_generate_twitter_thread_file_write_error(mock_chat_google_genai, mock_file_open, mock_makedirs, content_creator_instance, tmp_path):
    """Test Twitter thread generation when writing the file fails."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content=json.dumps(MOCK_GEMINI_TWITTER_RESPONSE_SUCCESS))
    mock_chat_google_genai.return_value = mock_llm_instance

    mock_file_open.side_effect = IOError("Simulated Twitter File Write Error")
    content_creator_instance.output_dir = str(tmp_path) # Ensure correct output dir
    mock_makedirs.return_value = None

    result = content_creator_instance.generate_twitter_thread(
        topic="Test Twitter File Write",
        tone="any"
    )

    assert "error" in result
    assert "Failed to write Twitter thread to file: Simulated Twitter File Write Error" in result["error"]
    assert "filepath" in result
    assert result["filepath"].startswith(str(tmp_path))
    
    mock_file_open.assert_called_once()
    args, _ = mock_file_open.call_args
    assert args[0].startswith(os.path.join(str(tmp_path), "twitter_thread_test_twitter_file_write"))
    assert args[1] == "w"

# Test for generate_image (placeholder function)
def test_generate_image_placeholder(content_creator_instance):
    """Test the placeholder generate_image function."""
    image_prompt = "A futuristic cityscape"
    result = content_creator_instance.generate_image(image_prompt)

    # The actual function returns an error structure, not a success structure
    assert "error" in result
    assert "Image generation with dalle is not implemented yet." in result["error"]
    assert "prompt_received" in result
    assert result["prompt_received"] == image_prompt
    assert "filepath" in result
    assert result["filepath"] is None
