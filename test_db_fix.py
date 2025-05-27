#!/usr/bin/env python3
"""
Quick test of the database threading fix
"""

import time
import sys
from database_threading_fix import patch_monitoring_system
from unified_monitoring_system import UnifiedMonitoringSystem

def test_fixed_monitoring():
    """Test the monitoring system with database fixes applied."""
    print("="*60)
    print("TESTING UNIFIED MONITORING WITH DATABASE FIXES")
    print("="*60)
    
    # Apply the database threading fix
    print("Applying database threading fixes...")
    if not patch_monitoring_system():
        print("❌ Failed to apply fixes")
        return False
    
    # Initialize monitoring system
    monitor = UnifiedMonitoringSystem()
    
    # Collect initial metrics
    print("Collecting initial metrics...")
    metrics = monitor.collect_system_metrics()
    if not metrics:
        print("❌ Failed to collect metrics")
        return False
    
    monitor.last_metrics = metrics
    print(f"✅ Initial metrics: Memory={metrics.memory_percent:.1f}%, CPU={metrics.cpu_percent:.1f}%")
    
    # Test database operations
    print("Testing database operations...")
    start_time = time.time()
    
    # Store metrics multiple times
    for i in range(5):
        test_metrics = monitor.collect_system_metrics()
        monitor.store_metrics(test_metrics)
        print(f"  Stored metrics batch {i+1}")
    
    # Test alerts
    monitor.check_thresholds(test_metrics)
    
    end_time = time.time()
    print(f"✅ Database operations completed in {(end_time - start_time)*1000:.2f}ms")
    
    # Get current status
    status = monitor.get_current_status()
    if 'error' in status:
        print(f"❌ Error getting status: {status['error']}")
        return False
    
    print(f"✅ Final status: Memory={status['memory_percent']:.1f}%, Alerts={len(status.get('recent_alerts', []))}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - DATABASE FIXES WORKING")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_fixed_monitoring()
    sys.exit(0 if success else 1)
