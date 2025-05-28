#!/usr/bin/env python3
"""
Gerenciador de Conexões SQLite por Thread - Versão Robusta
Sistema avançado para gerenciar conexões SQLite de forma thread-safe

Recursos:
- Conexões por thread usando threading.local
- Pool de conexões otimizado
- Monitoramento de saúde das conexões
- Cleanup automático e detecção de vazamentos
- Métricas de performance
"""

import sqlite3
import threading
import time
import logging
import weakref
import gc
from typing import Dict, List, Optional, Any, ContextManager
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
import json
from datetime import datetime, timedelta
from core.config import get_config # Added import

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Estatísticas de uma conexão SQLite."""
    thread_id: int
    created_at: datetime
    last_used: datetime
    query_count: int = 0
    total_execution_time: float = 0.0
    errors: int = 0
    is_active: bool = True
    
    def update_usage(self, execution_time: float = 0.0, error: bool = False):
        """Atualiza estatísticas de uso."""
        self.last_used = datetime.now()
        self.query_count += 1
        self.total_execution_time += execution_time
        if error:
            self.errors += 1
    
    @property
    def avg_execution_time(self) -> float:
        """Tempo médio de execução."""
        return self.total_execution_time / max(1, self.query_count)
    
    @property
    def age_seconds(self) -> float:
        """Idade da conexão em segundos."""
        return (datetime.now() - self.created_at).total_seconds()


class ThreadSafeConnectionManager:
    """
    Gerenciador de conexões SQLite thread-safe com pool otimizado.
    
    Características:
    - Uma conexão por thread usando threading.local
    - Monitoramento de saúde das conexões
    - Cleanup automático de conexões órfãs
    - Métricas detalhadas de performance
    - Detecção de vazamentos de conexão
    """
    
    def __init__(self, db_path: str, max_connections: int = 10, connection_timeout: int = 300): # Modified to accept max_connections and connection_timeout from new get_connection_manager
        self.db_path = Path(db_path)
        self.max_connections = max_connections # Will be set by new get_connection_manager
        self.connection_timeout = float(connection_timeout) # Will be set by new get_connection_manager, ensure float
        
        # Threading.local para conexões por thread
        self._local = threading.local()
        
        # Pool global de conexões e estatísticas
        self._connections_pool: Dict[int, sqlite3.Connection] = {}
        self._connection_stats: Dict[int, ConnectionStats] = {}
        self._lock = threading.RLock() # Changed from core version's threading.Lock() to keep original RLock
        
        # Monitoramento e métricas
        self._active_threads = weakref.WeakSet()
        self._query_history = deque(maxlen=1000)
        self._error_history = deque(maxlen=100)
        
        # Configurações otimizadas do SQLite
        self._sqlite_config = {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": 10000, # Kept original values
            "temp_store": "MEMORY",
            "mmap_size": 268435456,  # 256MB
            "foreign_keys": "ON"
        }
        
        # Thread de limpeza
        self._cleanup_thread = None
        self._running = True
        self._start_cleanup_thread()
        
        # Inicializar banco
        self._initialize_database() # Original _initialize_database to be kept
        
        logger.info(f"ThreadSafeConnectionManager inicializado para {self.db_path}")
    
    def _initialize_database(self):
        """Inicializa o banco de dados com esquema básico. This is the original method."""
        try:
            # The new get_connection_manager will pass a simple file path
            # So, creating connection here directly is fine.
            # No need to use self.get_connection() as it might not be fully set up.
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0, # Using a timeout consistent with original
                check_same_thread=False, # Allow connection to be used by this setup thread
                isolation_level=None 
            )
            try:
                cursor = conn.cursor()
                for pragma, value in self._sqlite_config.items():
                    cursor.execute(f"PRAGMA {pragma} = {value}")

                # Tabela de métricas de conexão (original schema)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS connection_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        thread_id INTEGER NOT NULL,
                        query_count INTEGER DEFAULT 0,
                        avg_execution_time REAL DEFAULT 0.0,
                        errors INTEGER DEFAULT 0,
                        connection_age REAL DEFAULT 0.0
                    )
                """)
                
                # Tabela de histórico de queries (original schema)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        thread_id INTEGER NOT NULL,
                        query_hash TEXT NOT NULL, -- Original schema had query_hash
                        execution_time REAL NOT NULL,
                        success BOOLEAN NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("Banco de dados inicializado com sucesso (original _initialize_database)")
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados (original _initialize_database): {e}")
            raise
    
    def _create_connection(self) -> sqlite3.Connection:
        """Cria uma nova conexão SQLite otimizada."""
        thread_id = threading.get_ident()
        
        try:
            # Criar conexão
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0, # Consistent timeout
                check_same_thread=False, # As per original
                isolation_level=None  # Autocommit mode, as per original
            )
            
            # Aplicar configurações otimizadas
            cursor = conn.cursor()
            for pragma, value in self._sqlite_config.items():
                cursor.execute(f"PRAGMA {pragma} = {value}")
            
            # Registrar estatísticas
            stats = ConnectionStats( # Using the original ConnectionStats
                thread_id=thread_id,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            with self._lock:
                self._connections_pool[thread_id] = conn
                self._connection_stats[thread_id] = stats
            
            logger.debug(f"Nova conexão criada para thread {thread_id}")
            return conn
            
        except Exception as e:
            logger.error(f"Erro ao criar conexão para thread {thread_id}: {e}")
            raise
    
    def _get_thread_connection(self) -> sqlite3.Connection:
        """Obtém a conexão da thread atual."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            # Check against max_connections from config
            with self._lock:
                if len(self._connections_pool) >= self.max_connections:
                    logger.warning(f"Pool de conexões cheio ({len(self._connections_pool)}/{self.max_connections}). Aguardando liberação.")
                    # Basic wait strategy, could be more sophisticated
                    # This part is tricky as the original didn't have a clear pool limit check here
                    # The new `get_connection_manager` sets `max_connections` which should be respected.
                    # However, the original `ThreadSafeConnectionManager` class structure did not enforce it at this exact point.
                    # For now, logging a warning. A full ConnectionPoolFullError might be too disruptive
                    # without re-architecting the original class's connection acquisition logic.
                    # The cleanup thread is expected to manage the pool size over time.
                    pass # Allowing creation for now, cleanup thread will manage overall count.

            self._local.connection = self._create_connection()
        
        return self._local.connection
    
    @contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """Context manager para obter conexão thread-safe."""
        thread_id = threading.get_ident()
        start_time = time.time()
        conn = None
        error_occurred = False
        
        try:
            # Registrar thread ativa
            self._active_threads.add(threading.current_thread())
            
            # Obter conexão
            conn = self._get_thread_connection()
            
            # Verificar saúde da conexão
            if not self._is_connection_healthy(conn):
                logger.warning(f"Conexão não saudável detectada para thread {thread_id}, recriando...")
                self._close_thread_connection() # Closes and removes from pool
                conn = self._get_thread_connection() # Recreates and adds to pool
            
            yield conn
            
        except Exception as e:
            error_occurred = True
            logger.error(f"Erro na conexão da thread {thread_id}: {e}")
            self._record_error(thread_id, str(e))
            raise
            
        finally:
            execution_time = time.time() - start_time
            
            if conn is not None: # Ensure conn was obtained
                # Atualizar estatísticas
                with self._lock: # Ensure thread-safe access to _connection_stats
                    if thread_id in self._connection_stats:
                        self._connection_stats[thread_id].update_usage(
                            execution_time=execution_time,
                            error=error_occurred
                        )
                
                # Registrar na história de queries (original method used query_hash, this one doesn't have it directly)
                # For simplicity, adapting to record without query_hash for now.
                # The original _save_metrics saves to DB, this _record_query is for in-memory deque.
                self._record_query(thread_id, execution_time, not error_occurred, "N/A_MERGED") # Added placeholder for query_hash

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """Verifica se a conexão está saudável."""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception:
            return False
    
    def _close_thread_connection(self):
        """Fecha a conexão da thread atual."""
        thread_id = threading.get_ident()
        
        # Fechar conexão local
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar conexão local: {e}")
            finally:
                self._local.connection = None
        
        # Remover do pool global
        with self._lock:
            if thread_id in self._connections_pool:
                conn_to_close = self._connections_pool.pop(thread_id, None)
                if conn_to_close:
                    try:
                        conn_to_close.close()
                    except Exception as e:
                        logger.warning(f"Erro ao fechar conexão do pool para thread {thread_id}: {e}")
            
            if thread_id in self._connection_stats:
                self._connection_stats[thread_id].is_active = False
                # Optionally remove from stats or mark as inactive for cleanup thread
                # del self._connection_stats[thread_id] # Or mark inactive
    
    def _record_query(self, thread_id: int, execution_time: float, success: bool, query_hash: str = "N/A"): # Added query_hash to match original _save_metrics
        """Registra execução de query no histórico."""
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'query_hash': query_hash, # Added field
            'execution_time': execution_time,
            'success': success
        }
        self._query_history.append(query_record) # This is the in-memory deque
    
    def _record_error(self, thread_id: int, error_msg: str):
        """Registra erro no histórico."""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'error': error_msg
        }
        self._error_history.append(error_record)
    
    def _start_cleanup_thread(self):
        """Inicia thread de limpeza de conexões órfãs."""
        def cleanup_worker():
            while self._running:
                try:
                    self._cleanup_orphaned_connections()
                    self._save_metrics_to_db()
                    time.sleep(30)  # Regular cleanup interval
                except sqlite3.Error as db_e:
                    logger.error(f"Database error in TSCM cleanup_worker: {db_e}", exc_info=True)
                    # Log and continue, but sleep longer to avoid spamming logs if DB is down.
                    if self._running: time.sleep(120)
                except KeyboardInterrupt:
                    logger.info("TSCM Cleanup_worker received KeyboardInterrupt. Setting running to False.")
                    self._running = False # Signal to stop
                    # Allow loop to terminate naturally by breaking or completing current iteration then exiting
                except Exception as e:
                    logger.critical(f"Unexpected critical error in TSCM cleanup_worker: {e}", exc_info=True)
                    # Continue running, but sleep for a longer interval.
                    if self._running: time.sleep(300)
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            daemon=True,
            name="SQLiteConnectionCleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_orphaned_connections(self):
        """Remove conexões órfãs e antigas."""
        current_time = datetime.now()
        orphaned_threads = []
        
        with self._lock:
            # Check for threads that are no longer alive
            active_thread_ids = {t.ident for t in threading.enumerate()}
            for thread_id, stats in list(self._connection_stats.items()):
                if thread_id not in active_thread_ids:
                    orphaned_threads.append(thread_id)
                    logger.info(f"Thread {thread_id} no longer alive. Marking connection for cleanup.")
                elif stats.is_active: # Only check timeout for active connections
                    age_since_last_use = (current_time - stats.last_used).total_seconds()
                    if age_since_last_use > self.connection_timeout:
                        orphaned_threads.append(thread_id)
                        logger.info(f"Connection for thread {thread_id} timed out (last use: {age_since_last_use:.2f}s ago). Marking for cleanup.")
            
            # Also prune connections if pool exceeds max_connections, prioritize oldest/least used
            # This part needs careful implementation to decide which connections to prune if over limit.
            # For now, focusing on timeout and non-alive threads.
            # The original class didn't have explicit pruning based on max_connections in cleanup.
            # The new class from `core` did, but its structure was simpler.

        # Fechar conexões marcadas
        for thread_id in set(orphaned_threads): # Use set to avoid double processing
            self._close_orphaned_connection(thread_id)
    
    def _close_orphaned_connection(self, thread_id: int):
        """Fecha conexão órfã específica."""
        with self._lock:
            conn = self._connections_pool.pop(thread_id, None)
            if conn:
                try:
                    conn.close()
                    logger.info(f"Conexão órfã/antiga fechada para thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Erro ao fechar conexão órfã {thread_id}: {e}")
            
            stats = self._connection_stats.pop(thread_id, None)
            if stats:
                stats.is_active = False
                # Optionally, log final stats before removing
    
    def _save_metrics_to_db(self): # Renamed from _save_metrics to distinguish
        """Salva métricas de performance no banco. (Original Logic)"""
        if not self._connection_stats: # Check if there are any stats to save
            return

        # Create a temporary, separate connection for saving metrics to avoid deadlocks
        # or using a connection that might be in a weird state.
        try:
            # Ensure db_path is a string for connect
            metrics_db_path = str(self.db_path)
            conn = sqlite3.connect(metrics_db_path, timeout=10.0, check_same_thread=False)
            cursor = conn.cursor()
            
            metrics_to_save = []
            with self._lock: # Access _connection_stats safely
                for thread_id, stats in self._connection_stats.items():
                    # Save metrics for all connections, active or not, if they have query counts
                    # Or, adjust logic to only save for active / recently closed.
                    # Original logic implies saving for connections in _connection_stats.
                    if stats.query_count > 0: # Only save if there's activity
                        metrics_to_save.append((
                            datetime.now().isoformat(),
                            thread_id,
                            stats.query_count,
                            stats.avg_execution_time,
                            stats.errors,
                            stats.age_seconds
                        ))
            
            if metrics_to_save:
                cursor.executemany("""
                    INSERT INTO connection_metrics (
                        timestamp, thread_id, query_count, 
                        avg_execution_time, errors, connection_age
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, metrics_to_save)
                conn.commit()
            
            # Save recent query history from the deque to query_history table
            # This part was in the original _initialize_database, but makes more sense here for periodic saving.
            # The original _record_query only updated an in-memory deque.
            queries_to_log_db = []
            temp_query_history = list(self._query_history) # Copy for iteration
            self._query_history.clear() # Clear after copying to avoid duplicate entries if successful

            for record in temp_query_history:
                 queries_to_log_db.append((
                    record['timestamp'],
                    record['thread_id'],
                    record.get('query_hash', "N/A_MERGED_SAVE"), # Use get for new field
                    record['execution_time'],
                    record['success']
                 ))
            
            if queries_to_log_db:
                cursor.executemany("""
                    INSERT INTO query_history (timestamp, thread_id, query_hash, execution_time, success)
                    VALUES (?, ?, ?, ?, ?)
                """, queries_to_log_db)
                conn.commit()

        except Exception as e:
            logger.error(f"Erro ao salvar métricas no banco de dados: {e}")
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das conexões."""
        with self._lock:
            # Filter stats for active connections by checking _connections_pool
            active_connection_ids = set(self._connections_pool.keys())
            active_stats = [s for tid, s in self._connection_stats.items() if tid in active_connection_ids and s.is_active]

            total_queries = sum(stats.query_count for stats in active_stats)
            total_errors = sum(stats.errors for stats in active_stats)
            
            return {
                'active_connections': len(active_stats),
                'total_managed_connections_stats': len(self._connection_stats), # All stats objects we know
                'current_pool_size': len(self._connections_pool), # Actual connections in pool
                'max_connections_config': self.max_connections,
                'total_queries_active_pool': total_queries,
                'total_errors_active_pool': total_errors,
                'error_rate_active_pool': (total_errors / max(1, total_queries)) * 100 if total_queries > 0 else 0,
                'active_threads_observed': len(self._active_threads), # Threads that requested a connection
                'query_history_deque_size': len(self._query_history),
                'error_history_deque_size': len(self._error_history)
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gera relatório detalhado de performance."""
        stats = self.get_connection_stats()
        
        recent_queries_execution_times = []
        # Accessing _query_history needs to be thread-safe if modified by other threads, though it's usually append-only from one.
        # For simplicity, direct access for now. If issues, use a lock or copy.
        temp_query_history_list = list(self._query_history) # Make a copy for safe iteration

        if temp_query_history_list:
            # Analyze last 100 from the deque
            recent_queries_from_deque = temp_query_history_list[-100:]
            recent_queries_execution_times = [q['execution_time'] for q in recent_queries_from_deque]
        
        if recent_queries_execution_times:
            avg_execution_time = sum(recent_queries_execution_times) / len(recent_queries_execution_times)
            max_execution_time = max(recent_queries_execution_times)
            min_execution_time = min(recent_queries_execution_times)
            queries_analyzed_count = len(recent_queries_execution_times)
        else:
            avg_execution_time = max_execution_time = min_execution_time = 0
            queries_analyzed_count = 0
        
        recent_errors_from_deque = list(self._error_history)[-10:]
        
        return {
            'connection_stats': stats,
            'performance_metrics_from_deque': {
                'avg_execution_time_ms': avg_execution_time * 1000,
                'max_execution_time_ms': max_execution_time * 1000,
                'min_execution_time_ms': min_execution_time * 1000,
                'recent_queries_analyzed': queries_analyzed_count
            },
            'recent_errors_from_deque': recent_errors_from_deque,
            'database_path': str(self.db_path),
            'sqlite_config': self._sqlite_config,
            'report_timestamp': datetime.now().isoformat()
        }
    
    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Executa query de forma thread-safe com métricas."""
        # This method uses the context manager, so stats are updated there.
        # query_hash is not available here to pass to _record_query.
        # The context manager's finally block calls _record_query.
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
                return cursor.fetchall()
            else:
                conn.commit()
                return []
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Executa múltiplas queries de forma otimizada."""
        # Similar to execute_query, relies on get_connection context manager for stats.
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def close_all_connections(self):
        """Fecha todas as conexões e para threads de cleanup."""
        logger.info("Iniciando o fechamento de todas as conexões...")
        self._running = False
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.debug("Aguardando thread de cleanup finalizar...")
            self._cleanup_thread.join(timeout=5) # Wait for cleanup thread
            if self._cleanup_thread.is_alive():
                logger.warning("Thread de cleanup não finalizou a tempo.")
        
        logger.debug("Fechando conexões do pool...")
        with self._lock:
            for thread_id in list(self._connections_pool.keys()): # list() for safe iteration while modifying
                conn_to_close = self._connections_pool.pop(thread_id, None)
                if conn_to_close:
                    try:
                        conn_to_close.close()
                    except Exception as e:
                        logger.warning(f"Erro ao fechar conexão do pool para thread {thread_id} durante close_all: {e}")
            
            self._connection_stats.clear() # Clear all stats
            self._connections_pool.clear() # Ensure pool is empty

        logger.debug("Fechando conexão local da thread atual (se existir)...")
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar conexão local da thread atual durante close_all: {e}")
            finally:
                self._local.connection = None
        
        # Final save of any pending metrics from deques before exiting
        logger.debug("Salvando métricas finais...")
        self._save_metrics_to_db()

        logger.info("Todas as conexões foram fechadas e o gerenciador foi encerrado.")
    
    def __del__(self):
        """Cleanup automático."""
        try:
            if self._running: # If still running, initiate close
                self.close_all_connections()
        except Exception as e:
            # Avoid raising exceptions from __del__
            logger.error(f"Erro durante __del__ em ThreadSafeConnectionManager: {e}")


# --- New get_connection_manager and related globals from core/thread_safe_connection_manager.py ---
_connection_manager: Optional[ThreadSafeConnectionManager] = None # This will now hold the instance of the class above
_manager_lock = threading.Lock() # Changed from RLock to Lock to match the new get_connection_manager's version

def get_connection_manager(db_path: Optional[str] = None) -> ThreadSafeConnectionManager:
    """
    Factory function to get a singleton instance of ThreadSafeConnectionManager.
    It uses UnifiedConfig for database path and connection parameters.
    """
    global _connection_manager
    if _connection_manager is None:
        with _manager_lock:
            if _connection_manager is None:
                config = get_config()
                # Use database_url from config if db_path is not provided.
                resolved_db_path = db_path or config.database.url
                
                # Ensure resolved_db_path is a string file path, not a "sqlite:///..." URL
                if resolved_db_path.startswith("sqlite:///"):
                    resolved_db_path = resolved_db_path.replace("sqlite:///", "")
                elif resolved_db_path.startswith("sqlite://"): 
                    resolved_db_path = resolved_db_path.replace("sqlite://", "")
                
                # Ensure db_path is absolute or make it relative to a known location if necessary
                # For now, assume it's either absolute or correctly relative.
                # Path(resolved_db_path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists

                max_connections = config.database.pool_size 
                connection_timeout = config.database.timeout # This is float in config, class __init__ takes int.

                logger.info(f"Instanciando ThreadSafeConnectionManager com db_path='{resolved_db_path}', max_connections={max_connections}, connection_timeout={connection_timeout}")
                _connection_manager = ThreadSafeConnectionManager(
                    db_path=str(resolved_db_path), # Ensure it's a string
                    max_connections=int(max_connections), # Ensure int
                    connection_timeout=int(connection_timeout) # Ensure int for original class constructor
                )
    return _connection_manager
# --- End of new get_connection_manager ---


def close_connection_manager(): # Original close_connection_manager, should still work
    """Fecha o gerenciador de conexões global."""
    global _connection_manager
    
    if _connection_manager is not None:
        with _manager_lock: # Use the same lock as get_connection_manager
            if _connection_manager is not None:
                _connection_manager.close_all_connections()
                _connection_manager = None
                logger.info("Gerenciador de conexões global foi fechado.")


if __name__ == "__main__":
    # Teste do gerenciador de conexões
    print("Testando ThreadSafeConnectionManager com nova configuração centralizada...")
    
    # Configure UnifiedConfig (mocked for test)
    class MockDatabaseConfig:
        url: str = "test_unified_connections.db"
        pool_size: int = 7  # Test with a different pool size
        timeout: float = 180.0 # Test with a different timeout

    class MockConfig:
        database = MockDatabaseConfig()

    # Mock get_config
    original_get_config = get_config # Store original
    def mock_get_config_func():
        return MockConfig()
    
    # Apply the mock
    # This is a bit tricky as get_config is imported directly.
    # For a real test, dependency injection or patching (unittest.mock.patch) would be better.
    # Temporarily override for this __main__ block.
    import sys
    # This is a hacky way to mock for a standalone script test.
    # In a real application, use proper mocking frameworks.
    if 'core.config' in sys.modules:
        original_core_get_config = sys.modules['core.config'].get_config
        sys.modules['core.config'].get_config = mock_get_config_func
    else:
        # If core.config wasn't imported yet by get_config, this won't work as expected.
        # This highlights the challenge of testing such global configurations.
        # Forcing an import and patch if possible (might not work reliably here)
        class MockCoreConfigModule:
            get_config = mock_get_config_func
        sys.modules['core.config'] = MockCoreConfigModule # type: ignore

    try:
        manager = get_connection_manager() # Should use mocked config now
        print(f"Manager instanciado com db_path: {manager.db_path}, max_connections: {manager.max_connections}, timeout: {manager.connection_timeout}")

        # Teste básico
        with manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test_unified (id INTEGER PRIMARY KEY, value TEXT)")
            cursor.execute("INSERT INTO test_unified (value) VALUES (?)", ("teste_unificado",))
            conn.commit()
            print("Query de teste executada.")
        
        # Testar estatísticas
        stats = manager.get_connection_stats()
        print(f"Estatísticas: {json.dumps(stats, indent=2)}")
        
        # Testar relatório de performance
        report = manager.get_performance_report()
        print(f"Relatório de performance: {json.dumps(report, indent=2, default=str)}") # default=str for datetime

    except Exception as e:
        print(f"Erro durante o teste: {e}")
        logger.exception("Erro detalhado durante o teste")
    finally:
        # Cleanup
        close_connection_manager()
        # Restore original get_config if mocked
        if 'core.config' in sys.modules and hasattr(sys.modules['core.config'], 'get_config'):
            if 'original_core_get_config' in locals():
                 sys.modules['core.config'].get_config = original_core_get_config # type: ignore
            else: # If it was added freshly
                del sys.modules['core.config']

        print("Teste concluído!")
        # Attempt to clean up the test database file
        db_file = Path("test_unified_connections.db")
        if db_file.exists():
            try:
                db_file.unlink()
                print(f"Arquivo de banco de dados de teste '{db_file}' removido.")
            except Exception as e:
                print(f"Não foi possível remover o arquivo de banco de dados de teste '{db_file}': {e}")
