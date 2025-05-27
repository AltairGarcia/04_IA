#!/usr/bin/env python3
"""
Test script for the unified monitoring system
"""

import sys
import time
from unified_monitoring_system import UnifiedMonitoringSystem

def test_unified_monitoring():
    """Test the unified monitoring system"""
    print("Testing Unified Monitoring System...")
    
    try:        # Initialize the monitoring system
        monitor = UnifiedMonitoringSystem()
        print("✓ Unified monitoring system initialized successfully")
        
        # Manually collect initial metrics
        metrics = monitor.collect_system_metrics()
        monitor.last_metrics = metrics
        print("✓ Initial metrics collected")
        
        # Get current status
        status = monitor.get_current_status()
        print(f"✓ Current memory usage: {status['memory_percent']:.1f}%")
        print(f"✓ CPU usage: {status['cpu_percent']:.1f}%")
        print(f"✓ System status: {status['status']}")
        
        # Start monitoring
        monitor.start_monitoring()
        print("✓ Monitoring started")
        
        # Wait a moment and collect some metrics
        time.sleep(3)
        
        # Get updated status
        status = monitor.get_current_status()
        print(f"✓ Updated memory usage: {status['memory_percent']:.1f}%")
          # Check if we have any alerts
        if hasattr(monitor, 'alerts_buffer') and monitor.alerts_buffer:
            print(f"⚠ Active alerts: {len(monitor.alerts_buffer)}")
            for alert in list(monitor.alerts_buffer)[-3:]:  # Show last 3 alerts
                if hasattr(alert, 'severity') and hasattr(alert, 'message'):
                    print(f"  - {alert.severity}: {alert.message}")
                else:
                    print(f"  - Alert: {str(alert)}")
        else:
            print("✓ No active alerts")
          # Test emergency cleanup if memory is high
        if status['memory_percent'] > 85:
            print("⚠ High memory usage detected, testing emergency cleanup...")
            monitor._emergency_cleanup()
            time.sleep(2)
            new_status = monitor.get_current_status()
            print(f"✓ Memory after cleanup: {new_status['memory_percent']:.1f}%")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("✓ Monitoring stopped")
        
        print("\n" + "="*50)
        print("✅ UNIFIED MONITORING SYSTEM TEST PASSED")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unified_monitoring()
    sys.exit(0 if success else 1)
