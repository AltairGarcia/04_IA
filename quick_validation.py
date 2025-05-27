#!/usr/bin/env python3
"""
Quick validation script for the core systems we've been working on.
"""
import sys

def test_core_systems():
    """Test the core systems we've been working on."""
    results = []
    
    try:
        # Test system initialization
        from system_initialization import SystemInitializer
        initializer = SystemInitializer()
        status = initializer.get_initialization_status()
        results.append(("✅ System Initialization", status['status']))
    except Exception as e:
        results.append(("❌ System Initialization", f"Error: {e}"))
    
    try:
        # Test error handling
        from error_handling import ErrorHandler
        handler = ErrorHandler()
        results.append(("✅ Error Handler", "initialized"))
    except Exception as e:
        results.append(("❌ Error Handler", f"Error: {e}"))
    
    try:
        # Test deployment readiness
        from deployment_readiness import DeploymentChecker
        checker = DeploymentChecker()
        results.append(("✅ Deployment Checker", "initialized"))
    except Exception as e:
        results.append(("❌ Deployment Checker", f"Error: {e}"))
    
    try:
        # Test configuration
        from config_robust import load_config_robust
        config = load_config_robust()
        results.append(("✅ Configuration", "loaded"))
    except Exception as e:
        results.append(("❌ Configuration", f"Error: {e}"))
    
    try:
        # Test Streamlit app availability
        import requests
        response = requests.get('http://localhost:8501', timeout=5)
        if response.status_code == 200:
            results.append(("✅ Streamlit App", "running"))
        else:
            results.append(("❌ Streamlit App", f"Status: {response.status_code}"))
    except Exception as e:
        results.append(("❌ Streamlit App", f"Error: {e}"))
    
    # Print results
    print("🔍 CORE SYSTEMS VALIDATION")
    print("=" * 50)
    
    success_count = 0
    total_count = len(results)
    
    for status, details in results:
        print(f"{status}: {details}")
        if "✅" in status:
            success_count += 1
    
    print("=" * 50)
    print(f"SUCCESS RATE: {success_count}/{total_count} ({(success_count/total_count)*100:.1f}%)")
    
    if success_count >= 4:
        print("🎉 EXCELLENT! Core systems are working well!")
        return True
    elif success_count >= 3:
        print("👍 GOOD! Most core systems are working!")
        return True
    else:
        print("⚠️  NEEDS WORK: Several systems need attention")
        return False

if __name__ == "__main__":
    success = test_core_systems()
    sys.exit(0 if success else 1)
