#!/usr/bin/env python3
"""
Performance test for the unified monitoring system
Tests the system under load and measures performance improvements
"""

import time
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor
from unified_monitoring_system import UnifiedMonitoringSystem

def simulate_workload():
    """Simulate CPU and memory workload"""
    # Create some memory pressure
    data = []
    for i in range(10000):
        data.append(f"test_data_{i}" * 100)
    
    # Simulate CPU work
    total = 0
    for i in range(100000):
        total += i ** 2
    
    return len(data)

def test_performance():
    """Test system performance under load"""
    print("="*60)
    print("PERFORMANCE TEST - Unified Monitoring System")
    print("="*60)
    
    # Initialize monitoring
    monitor = UnifiedMonitoringSystem()
    monitor.start_monitoring()
    
    # Collect initial metrics
    initial_metrics = monitor.collect_system_metrics()
    monitor.last_metrics = initial_metrics
    
    print(f"Initial Memory: {initial_metrics.memory_percent:.1f}%")
    print(f"Initial CPU: {initial_metrics.cpu_percent:.1f}%")
    print(f"Initial Threads: {initial_metrics.active_threads}")
    
    # Simulate workload with multiple threads
    print("\nStarting workload simulation...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(simulate_workload) for _ in range(5)]
        
        # Monitor during workload
        for i in range(10):
            time.sleep(1)
            current_metrics = monitor.collect_system_metrics()
            monitor.last_metrics = current_metrics
            
            print(f"  Time +{i+1}s: Memory={current_metrics.memory_percent:.1f}%, "
                  f"CPU={current_metrics.cpu_percent:.1f}%, "
                  f"Threads={current_metrics.active_threads}")
        
        # Wait for completion
        results = [future.result() for future in futures]
    
    end_time = time.time()
    
    # Final metrics
    final_metrics = monitor.collect_system_metrics()
    monitor.last_metrics = final_metrics
    
    print(f"\nWorkload completed in {end_time - start_time:.2f} seconds")
    print(f"Final Memory: {final_metrics.memory_percent:.1f}%")
    print(f"Final CPU: {final_metrics.cpu_percent:.1f}%")
    print(f"Final Threads: {final_metrics.active_threads}")
    
    # Test monitoring efficiency
    print("\nTesting monitoring system efficiency...")
    
    # Measure monitoring overhead
    monitor_start = time.time()
    for _ in range(100):
        metrics = monitor.collect_system_metrics()
    monitor_end = time.time()
    
    print(f"100 metrics collections took: {(monitor_end - monitor_start)*1000:.2f}ms")
    print(f"Average per collection: {(monitor_end - monitor_start)*10:.2f}ms")
    
    # Test alert system
    print("\nTesting alert system...")
    alerts_before = len(monitor.alerts_buffer)
    
    # Force some alerts by setting low thresholds temporarily
    original_thresholds = monitor.thresholds.copy()
    monitor.thresholds['memory_warning'] = 50.0
    monitor.thresholds['cpu_warning'] = 50.0
    
    # Collect metrics to trigger alerts
    test_metrics = monitor.collect_system_metrics()
    monitor.check_thresholds(test_metrics)
    
    alerts_after = len(monitor.alerts_buffer)
    
    # Restore original thresholds
    monitor.thresholds = original_thresholds
    
    print(f"Alerts generated: {alerts_after - alerts_before}")
    
    # Test database operations
    print("\nTesting database performance...")
    db_start = time.time()
    
    # Store multiple metrics
    for i in range(10):
        test_metrics = monitor.collect_system_metrics()
        monitor.store_metrics(test_metrics)
    
    db_end = time.time()
    print(f"10 database operations took: {(db_end - db_start)*1000:.2f}ms")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("\n" + "="*60)
    print("✅ PERFORMANCE TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    
    # Performance summary
    memory_change = final_metrics.memory_percent - initial_metrics.memory_percent
    thread_change = final_metrics.active_threads - initial_metrics.active_threads
    
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"- Memory change: {memory_change:+.1f}%")
    print(f"- Thread change: {thread_change:+d}")
    print(f"- Monitoring overhead: {(monitor_end - monitor_start)*10:.2f}ms per collection")
    print(f"- Database performance: {(db_end - db_start)*100:.2f}ms per operation")
    print(f"- Alerts generated: {alerts_after - alerts_before}")

if __name__ == "__main__":
    test_performance()
