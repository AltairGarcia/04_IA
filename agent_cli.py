"""
AI Agent CLI for Coding, Testing, and Deployment Automation
Usage: python agent_cli.py "Your command here"
"""
import sys
import os
from langchain.chains import LLMMathChain # Added import
from langchain_google_genai import ChatGoogleGenerativeAI # Added import
from langchain.agents import Tool # Added import
from langchain_community.tools import DuckDuckGoSearchRun # Added import
from agent import create_agent
from config_manager import config # Changed from ConfigManager to the global config instance
from content_creation import ContentCreator
from weather import get_weather, format_weather_response # Added import

def main():
    if len(sys.argv) < 2:
        print("Usage: python agent_cli.py 'Your command here'")
        sys.exit(1)
    user_command = sys.argv[1]

    # The global 'config' instance from config_manager is already loaded.
    # No need to call load_config() explicitly here.

    # Get API key using the .get() method and the correct path from DEFAULT_CONFIG
    gemini_api_key_from_config = config.get("api_keys.gemini") 
    if not gemini_api_key_from_config:
        print("Error: GEMINI_API_KEY (API_KEY in .env or api_keys.gemini in config.json) not found.")
        sys.exit(1)

    # Prepare the api_keys dictionary for ContentCreator
    # ContentCreator expects a dictionary, e.g., {"api_key": "your_gemini_key", "model_name": "gemini-pro", ...}
    # The key for the Gemini API key within this dict should be "api_key" as used by generate_script.
    content_creator_api_keys_dict = {
        "api_key": gemini_api_key_from_config, # This is the Gemini API key
        "model_name": config.get("llm.model_name", default="gemini-1.5-flash-latest"),
        "temperature": config.get("llm.temperature", default=0.7)
        # Add other API keys here if ContentCreator methods expect them from self.api_keys
    }

    content_creator_instance = ContentCreator(api_keys=content_creator_api_keys_dict)

    # Agent configuration
    # The agent itself also needs the Gemini API key directly for its own LLM instance.
    agent_model_name = config.get("llm.model_name", default="gemini-1.5-flash-latest")
    agent_temperature = config.get("llm.temperature", default=0.7)

    config_data = {
        "model_name": agent_model_name,
        "temperature": float(agent_temperature),
        "api_key": gemini_api_key_from_config, # Agent's direct Gemini API key
        "system_prompt": "You are an expert AI content creation and utility assistant."
    }

    # Initialize LLM for the calculator tool
    # It can use the same Gemini key and a suitable model
    calculator_llm = ChatGoogleGenerativeAI(
        model=config.get("llm.model_name", default="gemini-1.5-flash-latest"), # Use configured model
        google_api_key=gemini_api_key_from_config,
        temperature=0 # Math should be deterministic
    )
    llm_math_chain = LLMMathChain.from_llm(llm=calculator_llm)
    calculator_tool = Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math."
    )
    
    agent_tools = [calculator_tool]
    # If you have other tools like web search, weather, they should be initialized and added here too.
    # For example:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="WebSearch", 
        func=search.run, 
        description="Useful for when you need to answer questions about current events or general knowledge."
    )
    agent_tools.append(search_tool)

    # Weather tool
    def weather_func(location: str) -> str:
        try:
            # The get_weather function will try to get OPENWEATHER_API_KEY from .env
            weather_data = get_weather(location)
            return format_weather_response(weather_data)
        except Exception as e:
            # Log the error or handle it as needed
            print(f"Error in weather_func: {e}") # Basic error printing
            return f"Desculpe, não consegui obter a previsão do tempo para {location}. Erro: {e}"

    weather_tool = Tool(
        name="Weather",
        func=weather_func, # Use the wrapper function
        description="Useful for when you need to get the current weather conditions for a specific location. Input should be the city name (e.g., London) or city, country code (e.g., London,UK)."
    )
    agent_tools.append(weather_tool)

    agent = create_agent(config_data, tools=agent_tools, content_creator=content_creator_instance)
    
    print("Agent processing your command...\n")
    result = agent.invoke({"input": user_command})
    print(result.get("output", "No output."))

if __name__ == "__main__":
    main()
