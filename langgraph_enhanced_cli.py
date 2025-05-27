#!/usr/bin/env python3
"""
LangGraph 101 - Enhanced CLI Application
========================================

This is the enhanced version of the CLI application that integrates with
the new infrastructure while maintaining full backward compatibility.

This application automatically detects and uses infrastructure components
when available, falling back gracefully to original functionality when not.

Features added:
- Infrastructure integration with fallback
- Performance monitoring
- Enhanced status commands
- Automatic service health checking
- Advanced caching and optimization

Author: GitHub Copilot
Date: 2024
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import and apply integration patches
try:
    from cli_integration_patch import patch_cli_app, enhance_help_menu
    from app_integration_wrapper import get_integration_status
    
    # Apply CLI integration patch
    cli_wrapper = patch_cli_app()
    
    if cli_wrapper:
        print("🚀 LangGraph 101 - Enhanced Edition")
        integration_status = get_integration_status()
        if integration_status.get('infrastructure_available', False):
            print("✅ Infrastructure Mode: Enhanced features enabled")
        else:
            print("⚠️  Fallback Mode: Basic features with some enhancements")
        print("-" * 50)
          # Get original main function
        try:
            # Import original main function
            exec(open('langgraph-101.py').read(), globals())
            if 'main' in globals():
                original_main = globals()['main']
                # Wrap with enhancements
                enhanced_main = cli_wrapper.wrap_main_function(original_main)
                enhanced_main()
                sys.exit(0)
        except Exception as e:
            print(f"Failed to enhance CLI: {e}")
            # Continue to fallback
    
except ImportError as e:
    print("⚠️  Integration patches not available - running original CLI")
    print(f"Details: {e}")

# Fallback to original application if integration fails
try:
    # Import the original application
    from importlib import import_module
    original_app = import_module('langgraph-101')
    
    # Execute original main if available
    if hasattr(original_app, 'main'):
        original_app.main()
    else:
        # If no main function, just import and run the module
        print("Running original LangGraph 101...")
        exec(open('langgraph-101.py').read())
        
except Exception as e:
    print(f"❌ Failed to load application: {e}")
    sys.exit(1)
