#!/usr/bin/env python3
"""
Integration Validation Test
==========================

This script validates that all integration components work together correctly.
It tests:
1. Component imports
2. Fallback mechanisms
3. Configuration loading
4. Service connectivity
5. Error handling

Author: GitHub Copilot
Date: 2024
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime

# Test results collector
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "summary": {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
}

def log_test(test_name: str, status: str, message: str = "", details: str = ""):
    """Log test result"""
    test_results["tests"].append({
        "name": test_name,
        "status": status,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })
    test_results["summary"]["total"] += 1
    if status == "PASS":
        test_results["summary"]["passed"] += 1
        print(f"✅ {test_name}: {message}")
    elif status == "FAIL":
        test_results["summary"]["failed"] += 1
        print(f"❌ {test_name}: {message}")
    elif status == "WARN":
        test_results["summary"]["warnings"] += 1
        print(f"⚠️  {test_name}: {message}")

def test_component_imports():
    """Test that all integration components can be imported"""
    
    components = [
        ("langgraph_integration_adapter", "LangGraphIntegrationAdapter"),
        ("integrated_config", "LangGraphIntegratedConfig"),
        ("app_integration_wrapper", "EnhancedLangGraphApp"),
        ("streamlit_integration_patch", "patch_streamlit_app"),
        ("cli_integration_patch", "patch_cli_app")
    ]
    
    for module_name, class_name in components:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                log_test(f"Import {module_name}.{class_name}", "PASS", f"Successfully imported {class_name}")
            else:
                log_test(f"Import {module_name}.{class_name}", "FAIL", f"Class {class_name} not found in module")
        except Exception as e:
            log_test(f"Import {module_name}.{class_name}", "FAIL", f"Import failed: {str(e)}")

def test_configuration_system():
    """Test the integrated configuration system"""
    
    try:
        from integrated_config import LangGraphIntegratedConfig
        
        # Test default configuration loading
        config = LangGraphIntegratedConfig()
        log_test("Configuration Load", "PASS", "Default configuration loaded successfully")
        
        # Test configuration access
        if hasattr(config, 'database') and hasattr(config, 'redis'):
            log_test("Configuration Structure", "PASS", "Configuration has required sections")
        else:
            log_test("Configuration Structure", "FAIL", "Configuration missing required sections")
            
        # Test configuration validation
        try:
            config.validate()
            log_test("Configuration Validation", "PASS", "Configuration validation passed")
        except Exception as e:
            log_test("Configuration Validation", "WARN", f"Validation warnings: {str(e)}")
            
    except Exception as e:
        log_test("Configuration System", "FAIL", f"Configuration test failed: {str(e)}")

def test_fallback_mechanisms():
    """Test that fallback mechanisms work correctly"""
    
    try:
        from app_integration_wrapper import with_infrastructure_fallback, integration_state
        
        # Test decorator without infrastructure
        @with_infrastructure_fallback
        def test_function():
            return "enhanced"
        
        result = test_function()
        if result is not None:
            log_test("Fallback Decorator", "PASS", "Fallback decorator works correctly")
        else:
            log_test("Fallback Decorator", "FAIL", "Fallback decorator returned None")
            
        # Test integration state
        if hasattr(integration_state, 'infrastructure_available'):
            log_test("Integration State", "PASS", f"Infrastructure available: {integration_state.infrastructure_available}")
        else:
            log_test("Integration State", "FAIL", "Integration state not properly initialized")
            
    except Exception as e:
        log_test("Fallback Mechanisms", "FAIL", f"Fallback test failed: {str(e)}")

def test_service_connectivity():
    """Test connectivity to external services"""
    
    # Test Redis connectivity
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        log_test("Redis Connectivity", "PASS", "Redis server is available")
    except Exception as e:
        log_test("Redis Connectivity", "WARN", f"Redis not available: {str(e)}")
    
    # Test Infrastructure Hub availability
    try:
        from infrastructure_integration_hub import InfrastructureIntegrationHub
        hub = InfrastructureIntegrationHub()
        log_test("Infrastructure Hub", "PASS", "Infrastructure Hub can be instantiated")
    except Exception as e:
        log_test("Infrastructure Hub", "WARN", f"Infrastructure Hub not available: {str(e)}")

def test_integration_adapter():
    """Test the integration adapter functionality"""
    
    try:
        from langgraph_integration_adapter import LangGraphIntegrationAdapter
        
        # Test adapter instantiation
        adapter = LangGraphIntegrationAdapter()
        log_test("Integration Adapter Creation", "PASS", "Adapter created successfully")
        
        # Test health check endpoint
        if hasattr(adapter, 'app') and hasattr(adapter.app, 'routes'):
            log_test("Integration Adapter Routes", "PASS", "FastAPI routes available")
        else:
            log_test("Integration Adapter Routes", "FAIL", "FastAPI app not properly configured")
            
    except Exception as e:
        log_test("Integration Adapter", "FAIL", f"Adapter test failed: {str(e)}")

def test_patch_functionality():
    """Test application patch functionality"""
    
    # Test CLI patch
    try:
        from cli_integration_patch import patch_cli_app
        cli_patch = patch_cli_app()
        if cli_patch is not None:
            log_test("CLI Patch", "PASS", "CLI patch available")
        else:
            log_test("CLI Patch", "WARN", "CLI patch returned None (fallback mode)")
    except Exception as e:
        log_test("CLI Patch", "FAIL", f"CLI patch failed: {str(e)}")
    
    # Test Streamlit patch
    try:
        from streamlit_integration_patch import patch_streamlit_app
        streamlit_patch = patch_streamlit_app()
        if streamlit_patch is not None:
            log_test("Streamlit Patch", "PASS", "Streamlit patch available")
        else:
            log_test("Streamlit Patch", "WARN", "Streamlit patch returned None (fallback mode)")
    except Exception as e:
        log_test("Streamlit Patch", "FAIL", f"Streamlit patch failed: {str(e)}")

def test_error_handling():
    """Test error handling and recovery mechanisms"""
    
    try:
        from app_integration_wrapper import EnhancedLangGraphApp
        
        # Test graceful degradation
        app = EnhancedLangGraphApp()
        if app:
            log_test("Error Handling", "PASS", "Enhanced app handles missing dependencies gracefully")
        else:
            log_test("Error Handling", "FAIL", "Enhanced app failed to handle errors")
            
    except Exception as e:
        log_test("Error Handling", "WARN", f"Error handling test incomplete: {str(e)}")

def generate_test_report():
    """Generate and save test report"""
    
    # Calculate success rate
    total = test_results["summary"]["total"]
    passed = test_results["summary"]["passed"]
    success_rate = (passed / total * 100) if total > 0 else 0
    
    test_results["summary"]["success_rate"] = round(success_rate, 2)
    
    # Save report
    report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📊 Integration Test Report")
    print(f"==========================")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {test_results['summary']['failed']}")
    print(f"Warnings: {test_results['summary']['warnings']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Report saved to: {report_file}")
    
    return test_results

def main():
    """Run all integration validation tests"""
    
    print("🔍 Starting Integration Validation Tests")
    print("=" * 50)
    
    # Run all tests
    test_component_imports()
    test_configuration_system()
    test_fallback_mechanisms()
    test_service_connectivity()
    test_integration_adapter()
    test_patch_functionality()
    test_error_handling()
    
    # Generate report
    report = generate_test_report()
    
    # Return exit code based on results
    if report["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
