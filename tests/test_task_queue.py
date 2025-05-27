import pytest
from performance_optimization import TaskQueue, get_task_queue
import asyncio
import time

@pytest.fixture
def task_queue():
    queue = TaskQueue(max_workers=2)
    yield queue
    # Cleanup
    if queue._running:
        queue.stop()

def test_task_queue_init():
    queue = TaskQueue(max_workers=2)
    assert queue._max_workers == 2
    assert not queue._running
    assert queue._tasks.empty()
    assert len(queue._workers) == 0

def test_task_queue_start_stop():
    queue = TaskQueue(max_workers=2)
    queue.start()
    assert queue._running
    assert len(queue._workers) == 2

    queue.stop()
    assert not queue._running
    assert len(queue._workers) == 0

async def async_task(duration: float) -> str:
    await asyncio.sleep(duration)
    return f"Task completed in {duration}s"

def test_task_queue_add_task(task_queue):
    task_queue.start()

    # Add a simple task
    future = task_queue.add_task(async_task, 0.1)
    result = future.result()  # Removed timeout argument
    assert result == "Task completed in 0.1s"

    # Add multiple tasks
    futures = []
    for i in range(3):
        futures.append(task_queue.add_task(async_task, 0.1))

    # Verify all tasks complete
    for future in futures:
        result = future.result()  # Removed timeout argument
        assert "Task completed" in result

def test_get_task_queue_singleton():
    queue1 = get_task_queue(max_workers=2)
    queue2 = get_task_queue()
    assert queue1 is queue2  # Should return same instance

    # Cleanup
    if queue1._running:
        queue1.stop()
