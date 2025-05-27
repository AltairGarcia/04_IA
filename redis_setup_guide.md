# Redis Setup Guide for LangGraph 101 Agent

## Quick Start (Recommended)

### Option 1: Redis on Windows (Memurai - Redis-compatible)
```bash
# Download and install Memurai (Redis for Windows)
# Visit: https://www.memurai.com/get-memurai
# Or use Chocolatey:
choco install memurai-developer

# Start Memurai service
net start memurai
```

### Option 2: Docker Redis (Cross-platform)
```bash
# Install Docker Desktop first
# Then run Redis in Docker:
docker run -d --name redis-langgraph -p 6379:6379 redis:latest

# Or with persistent storage:
docker run -d --name redis-langgraph -p 6379:6379 -v redis-data:/data redis:latest redis-server --appendonly yes
```

### Option 3: WSL2 Redis (Windows Subsystem for Linux)
```bash
# In WSL2 terminal:
sudo apt update
sudo apt install redis-server
sudo service redis-server start

# Test connection:
redis-cli ping
```

## Configuration Verification

After installing Redis, verify the installation:

```python
# Test script
import redis
r = redis.Redis(host='localhost', port=6379, socket_timeout=5)
print("Redis ping:", r.ping())
```

## Fallback Mode

If Redis is not available, the system automatically uses an enhanced in-memory fallback with file persistence.

## Production Recommendations

1. **High Availability**: Use Redis Cluster or Sentinel
2. **Persistence**: Configure both RDB and AOF
3. **Security**: Enable AUTH and SSL/TLS
4. **Memory**: Set appropriate maxmemory and eviction policies
5. **Monitoring**: Use Redis metrics and health checks
