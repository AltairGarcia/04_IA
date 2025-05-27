#!/usr/bin/env python3
"""
Final performance test with all fixes applied
"""

import time
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor
from database_threading_fix import patch_monitoring_system
from unified_monitoring_system import UnifiedMonitoringSystem

def simulate_workload():
    """Simulate some system workload."""
    data = []
    for i in range(5000):
        data.append(f"test_data_{i}" * 50)
    
    total = sum(i**2 for i in range(50000))
    return len(data)

def final_performance_test():
    """Final comprehensive performance test."""
    print("="*70)
    print("FINAL PERFORMANCE TEST - UNIFIED MONITORING SYSTEM")
    print("="*70)
    
    # Apply database fixes
    print("Applying database threading fixes...")
    if not patch_monitoring_system():
        print("❌ Failed to apply database fixes")
        return False
    
    # Initialize monitoring
    monitor = UnifiedMonitoringSystem()
    
    # Collect baseline metrics
    print("Collecting baseline metrics...")
    initial_metrics = monitor.collect_system_metrics()
    monitor.last_metrics = initial_metrics
    
    print(f"Baseline - Memory: {initial_metrics.memory_percent:.1f}%, CPU: {initial_metrics.cpu_percent:.1f}%, Threads: {initial_metrics.active_threads}")
    
    # Start monitoring system
    print("Starting monitoring system...")
    monitor.start_monitoring()
    time.sleep(2)  # Let it stabilize
    
    # Performance test 1: Database operations
    print("\n📊 TEST 1: Database Performance")
    db_start = time.time()
    
    for i in range(10):
        test_metrics = monitor.collect_system_metrics()
        monitor.store_metrics(test_metrics)
        # Force some alerts
        if i == 5:
            monitor.thresholds['memory_warning'] = 50.0  # Lower threshold to trigger alerts
            monitor.check_thresholds(test_metrics)
            monitor.thresholds['memory_warning'] = 80.0  # Restore
    
    db_end = time.time()
    print(f"✅ 10 database operations: {(db_end - db_start)*1000:.1f}ms ({(db_end - db_start)*100:.1f}ms avg)")
    
    # Performance test 2: System under load
    print("\n🔥 TEST 2: System Under Load")
    load_start = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(simulate_workload) for _ in range(3)]
        
        # Monitor during load
        for i in range(5):
            current_metrics = monitor.collect_system_metrics()
            monitor.last_metrics = current_metrics
            print(f"  Load +{i+1}s: Memory={current_metrics.memory_percent:.1f}%, CPU={current_metrics.cpu_percent:.1f}%")
            time.sleep(1)
        
        results = [future.result() for future in futures]
    
    load_end = time.time()
    print(f"✅ Load test completed in {load_end - load_start:.1f}s")
    
    # Performance test 3: Monitoring efficiency
    print("\n⚡ TEST 3: Monitoring Efficiency")
    efficiency_start = time.time()
    
    for _ in range(50):
        metrics = monitor.collect_system_metrics()
    
    efficiency_end = time.time()
    print(f"✅ 50 metrics collections: {(efficiency_end - efficiency_start)*1000:.1f}ms ({(efficiency_end - efficiency_start)*20:.1f}ms avg)")
    
    # Final status check
    print("\n📈 FINAL STATUS")
    final_status = monitor.get_current_status()
    
    if 'error' not in final_status:
        print(f"Memory Usage: {final_status['memory_percent']:.1f}%")
        print(f"CPU Usage: {final_status['cpu_percent']:.1f}%")
        print(f"Active Threads: {final_status['active_threads']}")
        print(f"GC Objects: {final_status['gc_objects']:,}")
        print(f"System Status: {final_status['status']}")
        print(f"Registered Components: {final_status['registered_components']}")
        print(f"Recent Alerts: {len(final_status.get('recent_alerts', []))}")
        
        if final_status.get('recent_alerts'):
            print("Latest Alerts:")
            for alert in final_status['recent_alerts'][-3:]:
                print(f"  - [{alert['severity']}] {alert['message']}")
    else:
        print(f"❌ Error getting final status: {final_status['error']}")
        return False
    
    # Stop monitoring
    print("\nStopping monitoring system...")
    monitor.stop_monitoring()
    
    # Performance summary
    memory_change = final_status['memory_percent'] - initial_metrics.memory_percent
    thread_change = final_status['active_threads'] - initial_metrics.active_threads
    
    print("\n" + "="*70)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Memory Change: {memory_change:+.1f}%")
    print(f"Thread Change: {thread_change:+d}")
    print(f"Database Performance: {(db_end - db_start)*100:.1f}ms per operation")
    print(f"Monitoring Overhead: {(efficiency_end - efficiency_start)*20:.1f}ms per collection")
    print(f"Load Test Duration: {load_end - load_start:.1f}s")
    print(f"System Status: {final_status['status']}")
    print("="*70)
    print("✅ ALL PERFORMANCE TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = final_performance_test()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
