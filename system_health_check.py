"""
System Health Check Script
Run this to validate current system status
"""

import os
import sys
import asyncio
from datetime import datetime

def check_environment():
    """Check environment setup"""
    print("🔍 ENVIRONMENT CHECK")
    print("-" * 40)

    # Check .env file
    env_exists = os.path.exists('.env')
    print(f"📄 .env file: {'✅ Exists' if env_exists else '❌ Missing'}")

    # Check key directories
    dirs_to_check = ['analytics_data', 'tests', 'coverage_report']
    for dir_name in dirs_to_check:
        exists = os.path.exists(dir_name)
        print(f"📁 {dir_name}/: {'✅ Exists' if exists else '❌ Missing'}")

    # Check critical files
    files_to_check = [
        'content_creation.py',
        'workflow_optimization.py',
        'content_quality.py',
        'api_analytics.py',
        'database_manager.py'
    ]

    for file_name in files_to_check:
        exists = os.path.exists(file_name)
        print(f"📋 {file_name}: {'✅ Exists' if exists else '❌ Missing'}")

def check_dependencies():
    """Check critical dependencies"""
    print("\n🔧 DEPENDENCY CHECK")
    print("-" * 40)

    critical_imports = [
        'langchain',
        'langgraph',
        'streamlit',
        'google.generativeai',
        'sqlite3',
        'requests',
        'aiohttp',
        'seaborn',
        'numpy',
        'pandas'
    ]

    for module in critical_imports:
        try:
            __import__(module)
            print(f"📦 {module}: ✅ Available")
        except ImportError as e:
            print(f"📦 {module}: ❌ Missing ({e})")

def check_core_modules():
    """Check core application modules"""
    print("\n🧩 CORE MODULE CHECK")
    print("-" * 40)

    core_modules = [
        'content_creation',
        'workflow_optimization',
        'content_quality',
        'api_analytics',
        'database_manager'
    ]

    for module in core_modules:
        try:
            __import__(module)
            print(f"🎯 {module}: ✅ Importable")
        except Exception as e:
            print(f"🎯 {module}: ❌ Error ({e})")

def check_api_keys():
    """Check API key configuration"""
    print("\n🔑 API KEY CHECK")
    print("-" * 40)

    api_keys = [
        'API_KEY',
        'GEMINI_API_KEY',
        'TAVILY_API_KEY',
        'PEXELS_API_KEY',
        'PIXABAY_API_KEY',
        'ELEVENLABS_API_KEY',
        'ASSEMBLYAI_API_KEY'
    ]

    for key in api_keys:
        value = os.environ.get(key)
        if value:
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"🔐 {key}: ✅ Set ({masked})")
        else:
            print(f"🔐 {key}: ⚠️ Not set")

async def check_async_functionality():
    """Check async functionality"""
    print("\n⚡ ASYNC FUNCTIONALITY CHECK")
    print("-" * 40)

    try:
        # Test basic async operation
        await asyncio.sleep(0.1)
        print("🔄 Basic async: ✅ Working")

        # Test async module imports
        from content_creation import ContentCreator
        print("🎬 ContentCreator async import: ✅ Working")

    except Exception as e:
        print(f"🔄 Async functionality: ❌ Error ({e})")

def generate_health_report():
    """Generate comprehensive health report"""
    print("\n" + "=" * 60)
    print("🏥 SYSTEM HEALTH SUMMARY")
    print("=" * 60)

    # Basic system info
    print(f"🖥️ Python Version: {sys.version.split()[0]}")
    print(f"📅 Check Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Working Directory: {os.getcwd()}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("1. Fix any ❌ Missing dependencies or modules")
    print("2. Setup missing 🔐 API keys for full functionality")
    print("3. Run integration tests to validate functionality")
    print("4. Check content quality scoring algorithms")

    print("\n🎯 NEXT STEPS:")
    print("• Run: python test_integration_working.py")
    print("• Check: content quality scoring thresholds")
    print("• Setup: missing API keys")
    print("• Optimize: performance metrics")

async def main():
    """Main health check function"""
    print("🚀 STARTING SYSTEM HEALTH CHECK")
    print("=" * 60)

    check_environment()
    check_dependencies()
    check_core_modules()
    check_api_keys()
    await check_async_functionality()
    generate_health_report()

    print("\n✨ Health check completed!")

if __name__ == "__main__":
    asyncio.run(main())
