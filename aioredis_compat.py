#!/usr/bin/env python3
"""
Enhanced aioredis Compatibility for Python 3.13

This module provides comprehensive compatibility for aioredis in Python 3.13
with proper fallback mechanisms and enhanced error handling.
"""

import sys
import warnings
import logging
from typing import Optional, Union, Any, Dict

logger = logging.getLogger(__name__)

# Apply suppression before any imports
if sys.version_info >= (3, 13):
    warnings.filterwarnings("ignore", category=RuntimeWarning, 
                          message=".*TimeoutError.*duplicate base class.*")

# Safe aioredis import with fallback
try:
    import aioredis as _aioredis
    from aioredis import *
    from aioredis.client import Redis as AioRedis, StrictRedis
    AIOREDIS_AVAILABLE = True
    logger.info("✅ aioredis imported successfully")
    
except Exception as e:
    logger.warning(f"⚠️ aioredis not available: {e}")
    AIOREDIS_AVAILABLE = False
    
    # Fallback implementation
    class MockAioRedis:
        """Mock async Redis client"""
        def __init__(self, *args, **kwargs):
            self.connected = False
            logger.info("🔄 Using mock Redis client")
            
        async def ping(self) -> bool:
            return False
            
        async def set(self, key: str, value: Any, **kwargs) -> bool:
            return True
            
        async def get(self, key: str) -> Optional[str]:
            return None
            
        async def delete(self, *keys) -> int:
            return 0
            
        async def exists(self, *keys) -> int:
            return 0
            
        async def close(self):
            pass
            
        async def flushdb(self) -> bool:
            return True
    
    # Create mock exports
    AioRedis = MockAioRedis
    StrictRedis = MockAioRedis
    
    def from_url(url: str, **kwargs) -> MockAioRedis:
        return MockAioRedis()
    
    def create_redis_pool(*args, **kwargs):
        return MockAioRedis()

# Safe client creation function
def get_redis_client(url: str = "redis://localhost:6380", **kwargs):
    """Get Redis client with enhanced fallback"""
    if AIOREDIS_AVAILABLE:
        try:
            return _aioredis.from_url(url, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Failed to create real Redis client: {e}")
            return MockAioRedis()
    else:
        return MockAioRedis()

# Export compatibility flag
__all__ = ['AioRedis', 'StrictRedis', 'from_url', 'get_redis_client', 'AIOREDIS_AVAILABLE']
