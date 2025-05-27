#!/usr/bin/env python3
"""
Redis Integration Test for Day 1 - Redis Integration Fixes

This script tests the Redis integration across all components:
- Enhanced Redis Manager
- Cache Manager
- Message Queue System
- Fallback mechanisms
"""

import os
import sys
import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_redis_manager():
    """Test the enhanced Redis manager with the new Redis instance"""
    try:
        from enhanced_redis_manager import get_redis_manager, RedisConfig
        
        # Configure for port 6380
        config = RedisConfig(
            host='localhost',
            port=6380,
            db=0,
            socket_timeout=5.0,
            socket_connect_timeout=5.0
        )
        
        # Test Redis manager
        redis_manager = get_redis_manager(config)
        
        # Test basic operations
        test_key = "test_enhanced_redis"
        test_value = "enhanced_redis_value"
        
        # Set operation
        result = redis_manager.set(test_key, test_value, ex=60)
        assert result, "Redis set operation failed"
        logger.info("✅ Redis set operation successful")
        
        # Get operation
        retrieved_value = redis_manager.get(test_key)
        assert retrieved_value == test_value, f"Expected {test_value}, got {retrieved_value}"
        logger.info("✅ Redis get operation successful")
        
        # Increment operation
        counter_key = "test_counter"
        count = redis_manager.incr(counter_key)
        assert count == 1, f"Expected 1, got {count}"
        logger.info("✅ Redis incr operation successful")
        
        # TTL operation
        ttl = redis_manager.ttl(test_key)
        assert ttl > 0 and ttl <= 60, f"Expected TTL between 1-60, got {ttl}"
        logger.info("✅ Redis TTL operation successful")
        
        # Ping operation
        ping_result = redis_manager.ping()
        assert ping_result, "Redis ping failed"
        logger.info("✅ Redis ping successful")
        
        # Stats
        stats = redis_manager.get_stats()
        logger.info(f"✅ Redis stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Redis Manager test failed: {e}")
        return False

def test_cache_manager():
    """Test the cache manager with Redis integration"""
    try:
        from cache_manager import CacheManager
        
        # Initialize cache manager with new Redis URL
        cache_manager = CacheManager(redis_url='redis://localhost:6380/0')
        
        # Test basic cache operations
        test_key = "test_cache_key"
        test_data = {"message": "cache test", "timestamp": time.time()}
        
        # Set cache
        result = cache_manager.set(test_key, test_data, ttl=300)
        assert result, "Cache set operation failed"
        logger.info("✅ Cache set operation successful")
        
        # Get cache
        retrieved_data = cache_manager.get(test_key)
        assert retrieved_data == test_data, f"Expected {test_data}, got {retrieved_data}"
        logger.info("✅ Cache get operation successful")
        
        # Test cache with default fallback
        nonexistent_key = "nonexistent_key"
        default_value = "default"
        result = cache_manager.get(nonexistent_key, default_value)
        assert result == default_value, f"Expected {default_value}, got {result}"
        logger.info("✅ Cache default fallback successful")
        
        # Health check
        health = cache_manager.health_check()
        assert health['redis_available'], "Redis should be available"
        logger.info(f"✅ Cache health check: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache Manager test failed: {e}")
        return False

def test_enhanced_cache_manager():
    """Test the enhanced cache manager"""
    try:
        from enhanced_cache_manager import EnhancedCacheManager
        
        # Initialize enhanced cache manager
        enhanced_cache = EnhancedCacheManager(redis_url='redis://localhost:6380/1')
        
        # Test operations
        test_key = "test_enhanced_cache"
        test_data = {"enhanced": True, "performance": "optimized"}
        
        # Set operation
        result = enhanced_cache.set(test_key, test_data, ttl=600)
        assert result, "Enhanced cache set failed"
        logger.info("✅ Enhanced cache set operation successful")
        
        # Get operation
        retrieved_data = enhanced_cache.get(test_key)
        assert retrieved_data == test_data, f"Expected {test_data}, got {retrieved_data}"
        logger.info("✅ Enhanced cache get operation successful")
        
        # Performance test
        start_time = time.time()
        for i in range(100):
            enhanced_cache.set(f"perf_test_{i}", f"value_{i}", ttl=60)
        
        for i in range(100):
            value = enhanced_cache.get(f"perf_test_{i}")
            assert value == f"value_{i}", f"Performance test failed at {i}"
        
        end_time = time.time()
        operation_time = end_time - start_time
        logger.info(f"✅ Enhanced cache performance test: 200 operations in {operation_time:.3f}s")
        
        # Health check
        health = enhanced_cache.health_check()
        logger.info(f"✅ Enhanced cache health: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Cache Manager test failed: {e}")
        return False

def test_redis_fallback():
    """Test Redis fallback functionality"""
    try:
        from enhanced_redis_manager import EnhancedRedisManager, RedisConfig
        
        # Test with non-existent Redis to trigger fallback
        fallback_config = RedisConfig(
            host='localhost',
            port=9999,  # Non-existent port
            socket_timeout=1.0,
            socket_connect_timeout=1.0
        )
        
        # This should use fallback
        fallback_manager = EnhancedRedisManager(fallback_config)
        
        # Test fallback operations
        test_key = "fallback_test"
        test_value = "fallback_value"
        
        # Set operation (should use fallback)
        result = fallback_manager.set(test_key, test_value)
        assert result, "Fallback set operation failed"
        logger.info("✅ Fallback set operation successful")
        
        # Get operation (should use fallback)
        retrieved_value = fallback_manager.get(test_key)
        assert retrieved_value == test_value, f"Expected {test_value}, got {retrieved_value}"
        logger.info("✅ Fallback get operation successful")
        
        # Verify Redis is not available but operations work
        assert not fallback_manager.is_redis_available(), "Redis should not be available for fallback test"
        logger.info("✅ Fallback system working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis Fallback test failed: {e}")
        return False

def test_message_queue_system():
    """Test message queue system with Redis"""
    try:
        from message_queue_system import MessageQueueSystem, MessageQueueConfig
        
        # Configure for new Redis instance
        os.environ['REDIS_HOST'] = 'localhost'
        os.environ['REDIS_PORT'] = '6380'
        os.environ['REDIS_DB'] = '2'
        
        # Initialize message queue
        config = MessageQueueConfig()
        mq_system = MessageQueueSystem(config)
        
        # Test Redis connection
        mq_system._initialize_redis()
        
        # Test basic Redis operations
        test_result = mq_system.redis_client.ping()
        assert test_result, "Message queue Redis ping failed"
        logger.info("✅ Message queue Redis connection successful")
        
        # Test queue operations (basic)
        mq_system.redis_client.set("mq_test", "message_queue_value")
        value = mq_system.redis_client.get("mq_test")
        assert value == "message_queue_value", f"Expected message_queue_value, got {value}"
        logger.info("✅ Message queue basic operations successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Message Queue System test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive Redis integration tests"""
    logger.info("🚀 Starting Redis Integration Tests - Day 1")
    
    test_results = {}
    
    # Test 1: Enhanced Redis Manager
    logger.info("📋 Test 1: Enhanced Redis Manager")
    test_results['enhanced_redis_manager'] = test_enhanced_redis_manager()
    
    # Test 2: Cache Manager
    logger.info("📋 Test 2: Cache Manager")
    test_results['cache_manager'] = test_cache_manager()
    
    # Test 3: Enhanced Cache Manager
    logger.info("📋 Test 3: Enhanced Cache Manager")
    test_results['enhanced_cache_manager'] = test_enhanced_cache_manager()
    
    # Test 4: Redis Fallback
    logger.info("📋 Test 4: Redis Fallback")
    test_results['redis_fallback'] = test_redis_fallback()
    
    # Test 5: Message Queue System
    logger.info("📋 Test 5: Message Queue System")
    test_results['message_queue_system'] = test_message_queue_system()
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\n🏁 Redis Integration Test Results:")
    logger.info(f"   ✅ Passed: {passed_tests}/{total_tests}")
    logger.info(f"   ❌ Failed: {total_tests - passed_tests}/{total_tests}")
    logger.info(f"   📊 Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status}: {test_name}")
    
    if passed_tests == total_tests:
        logger.info("🎉 All Redis integration tests passed!")
        return True
    else:
        logger.warning("⚠️ Some Redis integration tests failed!")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
