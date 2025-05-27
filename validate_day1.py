#!/usr/bin/env python3
"""
Simple Redis Validation for Day 1 Completion
"""

import redis
import json

def main():
    print("🔍 Day 1 Redis Integration Validation")
    print("="*50)
    
    try:
        # Test Redis connection
        r = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)
        
        # Test basic operations
        ping_result = r.ping()
        print(f"✅ Redis Ping: {ping_result}")
        
        # Test set/get
        r.set("validation_test", "day1_complete", ex=60)
        value = r.get("validation_test")
        print(f"✅ Redis Set/Get: {value}")
        
        # Test Redis info
        info = r.info()
        print(f"✅ Redis Version: {info.get('redis_version', 'unknown')}")
        print(f"✅ Redis Port: {info.get('tcp_port', 'unknown')}")
        
        # Test different databases
        for db_num in [0, 1, 2]:
            db_redis = redis.Redis(host='localhost', port=6380, db=db_num, decode_responses=True)
            db_redis.set(f"db{db_num}_test", f"database_{db_num}_working")
            result = db_redis.get(f"db{db_num}_test")
            print(f"✅ Database {db_num}: {result}")
        
        print("\n🎉 ALL REDIS TESTS PASSED!")
        print("✅ Redis server is running correctly on port 6380")
        print("✅ All databases (0, 1, 2) are accessible")
        print("✅ Set/Get operations working")
        print("✅ Day 1 Redis Integration: COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("🎊 DAY 1 REDIS INTEGRATION FIXES: COMPLETE! 🎊")
        print("="*60)
    else:
        print("\n❌ Day 1 validation failed")
