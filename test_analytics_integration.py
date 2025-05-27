#!/usr/bin/env python3
"""
Test script for Phase 3 Analytics Integration

This script tests the integration of the advanced analytics system components
to ensure everything is working properly.
"""

import sys
import traceback
from datetime import datetime

def test_analytics_imports():
    """Test if all analytics components can be imported."""
    print("🔍 Testing analytics component imports...")
    
    try:
        from analytics.dashboard_components import AdvancedAnalyticsDashboard
        print("✅ AdvancedAnalyticsDashboard imported successfully")
        
        from analytics.analytics_logger import AnalyticsLogger
        print("✅ AnalyticsLogger imported successfully")
        
        from analytics.real_time_analytics import RealTimeAnalytics
        print("✅ RealTimeAnalytics imported successfully")
        
        from analytics.user_behavior_analyzer import UserBehaviorAnalyzer
        print("✅ UserBehaviorAnalyzer imported successfully")
        
        from analytics.performance_tracker import PerformanceTracker
        print("✅ PerformanceTracker imported successfully")
        
        from analytics.custom_reports_generator import CustomReportsGenerator
        print("✅ CustomReportsGenerator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_google_provider_analytics():
    """Test Google provider analytics integration."""
    print("🔍 Testing Google provider analytics integration...")
    
    try:
        from ai_providers.google_provider import GoogleProvider
        print("✅ GoogleProvider imported successfully")
          # Test if the new methods exist (using a mock API key for testing)
        provider = GoogleProvider(model_id="gemini-pro", api_key="test_key_for_method_testing")
        
        # Check for new methods
        if hasattr(provider, 'get_safety_settings'):
            print("✅ get_safety_settings method available")
        else:
            print("❌ get_safety_settings method missing")
            
        if hasattr(provider, 'batch_predict'):
            print("✅ batch_predict method available")
        else:
            print("❌ batch_predict method missing")
            
        if hasattr(provider, 'get_usage_stats'):
            print("✅ get_usage_stats method available")
        else:
            print("❌ get_usage_stats method missing")
            
        if hasattr(provider, 'health_check'):
            print("✅ health_check method available")
        else:
            print("❌ health_check method missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Google provider test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app_syntax():
    """Test if the main Streamlit app has no syntax errors."""
    print("🔍 Testing Streamlit app syntax...")
    
    try:
        import py_compile
        py_compile.compile('streamlit_app.py', doraise=True)
        print("✅ streamlit_app.py syntax is valid")
        return True
        
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error in streamlit_app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dashboard_instantiation():
    """Test if dashboard components can be instantiated."""
    print("🔍 Testing dashboard component instantiation...")
    
    try:
        from analytics.dashboard_components import AdvancedAnalyticsDashboard
        
        dashboard = AdvancedAnalyticsDashboard()
        print("✅ AdvancedAnalyticsDashboard instantiated successfully")
          # Check if key methods exist
        if hasattr(dashboard, 'render_main_dashboard'):
            print("✅ render_main_dashboard method available")
        else:
            print("❌ render_main_dashboard method missing")
            
        if hasattr(dashboard, 'render_real_time_dashboard'):
            print("✅ render_real_time_dashboard method available")
        else:
            print("❌ render_real_time_dashboard method missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard instantiation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Phase 3 Analytics Integration Tests")
    print("=" * 50)
    print(f"📅 Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Analytics Imports", test_analytics_imports),
        ("Google Provider Analytics", test_google_provider_analytics),
        ("Streamlit App Syntax", test_streamlit_app_syntax),
        ("Dashboard Instantiation", test_dashboard_instantiation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🔧 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 50)
    print("📊 Test Results Summary:")
    print()
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Phase 3 analytics integration is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
