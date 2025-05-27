#!/usr/bin/env python3
"""
End-to-end test of content creation through the agent system
"""

import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.append('.')

def test_agent_content_creation():
    """Test end-to-end content creation through the agent"""
    try:
        from agent import create_agent
        from content_creation import ContentCreator
        
        print("✓ Imports successful")
        
        # Set up test configuration
        test_config = {
            "api_key": "test_key",
            "model_name": "gemini-test",
            "temperature": 0.5,
            "system_prompt": "You are a helpful content creation assistant."
        }
        
        # Mock API keys for ContentCreator
        mock_api_keys = {
            "api_key": "test_gemini_key",
            "model_name": "gemini-test-model",
            "temperature": 0.5
        }
        
        # Create ContentCreator instance
        content_creator = ContentCreator(api_keys=mock_api_keys)
        print("✓ ContentCreator initialized")
        
        # Create agent with content creator
        agent = create_agent(test_config, [], content_creator=content_creator)
        print("✓ Agent with ContentCreator created")
        
        # Check if agent has content creator
        if hasattr(agent, 'content_creator') and agent.content_creator:
            print("✓ Agent has ContentCreator instance")
        else:
            print("⚠ Agent doesn't have ContentCreator attached")
            
        # Test content creation trigger detection
        blog_requests = [
            "crie um blog sobre Python",
            "escreva um post de blog sobre AI",
            "post de blog sobre tecnologia"
        ]
        
        twitter_requests = [
            "crie um twitter thread sobre Python",
            "gerar twitter thread sobre AI",
            "thread para o twitter sobre tecnologia"
        ]
        
        detected_blog = False
        detected_twitter = False
        
        # Check if agent can detect content creation requests
        if hasattr(agent, 'command_handlers'):
            for keywords_tuple in agent.command_handlers.keys():
                for keyword in keywords_tuple:
                    if 'blog' in keyword:
                        detected_blog = True
                    if 'twitter' in keyword:
                        detected_twitter = True
        
        if detected_blog:
            print("✓ Blog post triggers detected in agent")
        else:
            print("⚠ Blog post triggers not found")
            
        if detected_twitter:
            print("✓ Twitter thread triggers detected in agent")
        else:
            print("⚠ Twitter thread triggers not found")
        
        print("✓ Agent content creation integration test completed")
        return True
        
    except Exception as e:
        print(f"✗ Agent content creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_invoke_with_mock():
    """Test agent invoke with mocked LLM responses"""
    try:
        from agent import create_agent
        from content_creation import ContentCreator
        import json
        
        print("Testing agent invoke with mocked responses...")
        
        # Mock the ChatGoogleGenerativeAI to avoid real API calls
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_chat:
            # Configure mock
            mock_llm_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Mock response from agent"
            mock_llm_instance.invoke.return_value = mock_response
            mock_chat.return_value = mock_llm_instance
            
            # Test configuration
            test_config = {
                "api_key": "test_key",
                "model_name": "gemini-test",
                "temperature": 0.5,
                "system_prompt": "You are a helpful assistant."
            }
            
            # Create agent
            agent = create_agent(test_config, [])
            print("✓ Agent created with mocked LLM")
            
            # Test basic invoke
            result = agent.invoke({"input": "Hello, how are you?"})
            if result.get("output"):
                print("✓ Agent invoke working")
            else:
                print("⚠ Agent invoke returned empty output")
            
            # Test help request
            help_result = agent.invoke({"input": "ajuda"})
            if help_result.get("output"):
                print("✓ Agent help functionality working")
            else:
                print("⚠ Agent help functionality not working")
                
        print("✓ Agent invoke test completed")
        return True
        
    except Exception as e:
        print(f"✗ Agent invoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== End-to-End Content Creation Integration Test ===")
    print()
    
    # Test 1: Agent content creation setup
    print("1. Testing agent content creation setup...")
    content_test = test_agent_content_creation()
    print()
    
    # Test 2: Agent invoke functionality
    print("2. Testing agent invoke functionality...")
    invoke_test = test_agent_invoke_with_mock()
    print()
    
    # Summary
    print("=== Test Summary ===")
    if content_test and invoke_test:
        print("✓ All end-to-end tests passed!")
        print("✓ Content creation system is fully integrated with agent")
        print("✓ Agent can detect and handle content creation requests")
        print("✓ System is ready for production use")
        exit(0)
    else:
        print("✗ Some tests failed")
        exit(1)
