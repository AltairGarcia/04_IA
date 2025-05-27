#!/usr/bin/env python3
"""
LangGraph 101 Application Launcher with Robust Monitoring

This script provides a comprehensive launch system for the LangGraph 101 application
with health monitoring, deployment readiness checks, and automated recovery.
"""

import sys
import os
import time
import signal
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our robust systems
from deployment_readiness import run_comprehensive_deployment_check, generate_deployment_report
from config_robust import load_config_robust
from app_health import get_health_summary, start_health_monitoring
from langchain_robust import suppress_langchain_warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('langgraph_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LangGraphLauncher:
    """Comprehensive launcher for LangGraph 101 application."""
    
    def __init__(self):
        self.streamlit_process = None
        self.health_monitoring_active = False
        
    def print_banner(self):
        """Print application banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                             LangGraph 101                                   ║
║                      Robust AI Agent Platform                               ║
║                                                                              ║
║  🚀 Enhanced with:                                                           ║
║    • Robust Configuration Management                                         ║
║    • Comprehensive Health Monitoring                                         ║
║    • Automated Deployment Readiness Checks                                   ║
║    • Graceful Error Handling & Recovery                                      ║
║    • Real-time System Diagnostics                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def run_deployment_check(self) -> bool:
        """Run comprehensive deployment readiness check."""
        logger.info("[DEBUG] Running deployment readiness check...")
        
        try:
            with suppress_langchain_warnings():
                results = run_comprehensive_deployment_check()
            
            summary = results['summary']
            
            # Print summary
            print(f"\n📊 Deployment Check Results:")
            print(f"   Overall Status: {summary['overall_status'].upper()}")
            print(f"   Checks Passed: {summary['checks_passed']}/{summary['total_checks']}")
            print(f"   Warnings: {summary['checks_warning']}")
            print(f"   Failed: {summary['checks_failed']}")
            print(f"   Deployment Ready: {'YES' if summary['deployment_ready'] else 'NO'}")
            
            # Show critical issues
            if summary['critical_issues']:
                print(f"\n🚨 Critical Issues:")
                for issue in summary['critical_issues']:
                    print(f"   ❌ {issue}")
            
            # Show warnings
            if summary['warnings']:
                print(f"\n⚠️ Warnings:")
                for warning in summary['warnings']:
                    print(f"   ⚠️ {warning}")
            
            # Auto-install missing packages if needed
            pkg_results = results['detailed_results'].get('package_dependencies', {})
            missing_required = pkg_results.get('missing_required', [])
            
            if missing_required:
                print(f"\n📦 Missing Required Packages: {', '.join(missing_required)}")
                
                response = input("Would you like to install missing packages automatically? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    print("Installing missing packages...")
                    from deployment_readiness import install_missing_packages
                    install_result = install_missing_packages(missing_required)
                    
                    if install_result['status'] == 'success':
                        print("✅ Packages installed successfully!")
                        # Re-run check
                        print("Re-running deployment check...")
                        return self.run_deployment_check()
                    else:
                        print(f"❌ Installation failed: {install_result.get('failed', [])}")
                        return False
            
                        return summary['deployment_ready']
            
        except Exception as e:
            logger.error(f"Deployment check failed: {str(e)}")
            print(f"❌ Deployment check failed: {str(e)}")
            return False
    
    def check_configuration(self) -> bool:
        """Check if configuration is valid."""
        logger.info("Checking configuration...")
        
        try:
            config = load_config_robust()
            print(f"✅ Configuration loaded successfully")
            print(f"   Development mode: {config.development_mode}")
            print(f"   Debug mode: {config.debug}")
            print(f"   Log level: {config.log_level}")
            
            # Check for essential API keys
            missing_keys = []
            if not config.api_key or config.api_key.startswith('your-') or 'placeholder' in config.api_key.lower():
                missing_keys.append('API_KEY (Google Gemini)')
            if not config.tavily_api_key or config.tavily_api_key.startswith('tvly-') or 'placeholder' in config.tavily_api_key.lower():
                missing_keys.append('TAVILY_API_KEY')
            
            if missing_keys:
                print(f"⚠️ Placeholder API keys detected: {', '.join(missing_keys)}")
                if config.development_mode:
                    print("   This is OK for development mode")
                else:
                    print("   Set real API keys for production use")
            else:
                print("✅ All API keys properly configured")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration check failed: {str(e)}")
            print(f"❌ Configuration error: {str(e)}")
            return False
    
    def start_health_monitoring(self):
        """Start background health monitoring."""
        logger.info("🏥 Starting health monitoring...")
        
        try:
            start_health_monitoring()
            self.health_monitoring_active = True
            print("✅ Health monitoring started")
        except Exception as e:
            logger.warning(f"Health monitoring failed to start: {str(e)}")
            print(f"⚠️ Health monitoring unavailable: {str(e)}")
    
    def show_system_status(self):
        """Display current system status."""
        try:
            health = get_health_summary()
            
            print(f"\n📊 System Status:")
            print(f"   Overall Health: {health['overall_status'].upper()}")
            
            # Show component status
            components = ['python', 'packages', 'configuration', 'memory', 'disk']
            for component in components:
                if component in health:
                    status = health[component].get('status', 'unknown')
                    icon = '✅' if status == 'healthy' else '⚠️' if status == 'warning' else '❌'
                    print(f"   {icon} {component.title()}: {status}")
            
        except Exception as e:
            print(f"⚠️ Unable to get system status: {str(e)}")
    
    def start_streamlit(self, port: int = 8501, host: str = "localhost"):
        """Start Streamlit application."""
        logger.info(f"🚀 Starting Streamlit application on {host}:{port}...")
        
        try:
            cmd = [
                sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
                '--server.port', str(port),
                '--server.address', host,
                '--server.headless', 'false'
            ]
            
            print(f"Starting Streamlit on http://{host}:{port}")
            self.streamlit_process = subprocess.Popen(cmd)
            
            print("✅ Streamlit application started successfully!")
            print(f"🌐 Open your browser to: http://{host}:{port}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {str(e)}")
            print(f"❌ Failed to start Streamlit: {str(e)}")
            return False
    
    def stop_streamlit(self):
        """Stop Streamlit application."""
        if self.streamlit_process:
            logger.info("🛑 Stopping Streamlit application...")
            self.streamlit_process.terminate()
            self.streamlit_process.wait()
            self.streamlit_process = None
            print("✅ Streamlit application stopped")
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutdown signal received...")
        self.stop_streamlit()
        print("👋 Goodbye!")
        sys.exit(0)
    
    def monitor_application(self, interval: int = 60):
        """Monitor application health continuously."""
        logger.info(f"👁️ Starting application monitoring (interval: {interval}s)...")
        
        try:
            while True:
                # Check if Streamlit is still running
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    print("⚠️ Streamlit process died - attempting restart...")
                    self.start_streamlit()
                
                # Check system health
                try:
                    health = get_health_summary()
                    if health['overall_status'] == 'critical':
                        print(f"🚨 Critical system health detected at {time.strftime('%H:%M:%S')}")
                except Exception as e:
                    logger.warning(f"Health check failed: {str(e)}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👁️ Monitoring stopped by user")
    
    def run(self, args):
        """Main application runner."""
        self.print_banner()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        # Run pre-flight checks
        print("🔍 Running pre-flight checks...")
        
        if not args.skip_deployment_check:
            if not self.run_deployment_check():
                if not args.force:
                    print("❌ Deployment check failed. Use --force to start anyway.")
                    return 1
                else:
                    print("⚠️ Starting despite deployment issues (--force used)")
        
        if not self.check_configuration():
            if not args.force:
                print("❌ Configuration check failed. Use --force to start anyway.")
                return 1
            else:
                print("⚠️ Starting despite configuration issues (--force used)")
        
        # Start health monitoring
        if not args.skip_health_monitoring:
            self.start_health_monitoring()
        
        # Show system status
        self.show_system_status()
        
        # Start Streamlit
        if not self.start_streamlit(port=args.port, host=args.host):
            print("❌ Failed to start application")
            return 1
        
        print(f"\n🎉 LangGraph 101 is ready!")
        print(f"📱 Web Interface: http://{args.host}:{args.port}")
        print(f"📋 System Health: Available in 'System Health' tab")
        print(f"📊 Analytics: Available in 'Analytics' tab")
        
        if args.monitor:
            try:
                self.monitor_application(interval=args.monitor_interval)
            except KeyboardInterrupt:
                pass
        else:
            try:
                # Wait for Streamlit process
                self.streamlit_process.wait()
            except KeyboardInterrupt:
                pass
        
        return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LangGraph 101 - Robust AI Agent Platform Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                           # Normal startup with all checks
  python launcher.py --force                   # Force start despite issues
  python launcher.py --skip-deployment-check  # Skip deployment readiness check
  python launcher.py --monitor                 # Start with continuous monitoring
  python launcher.py --port 8080              # Use custom port
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='Port for Streamlit server (default: 8501)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host for Streamlit server (default: localhost)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force start even if checks fail'
    )
    
    parser.add_argument(
        '--skip-deployment-check',
        action='store_true',
        help='Skip deployment readiness check'
    )
    
    parser.add_argument(
        '--skip-health-monitoring',
        action='store_true',
        help='Skip health monitoring startup'
    )
    
    parser.add_argument(
        '--monitor', '-m',
        action='store_true',
        help='Enable continuous application monitoring'
    )
    
    parser.add_argument(
        '--monitor-interval',
        type=int,
        default=60,
        help='Monitoring interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    launcher = LangGraphLauncher()
    return launcher.run(args)


if __name__ == "__main__":
    sys.exit(main())
