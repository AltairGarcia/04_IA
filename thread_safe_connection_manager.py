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
    
    def __init__(self, db_path: str, max_connections: int = 10, connection_timeout: int = 300):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        
        # Threading.local para conexões por thread
        self._local = threading.local()
        
        # Pool global de conexões e estatísticas
        self._connections_pool: Dict[int, sqlite3.Connection] = {}
        self._connection_stats: Dict[int, ConnectionStats] = {}
        self._lock = threading.RLock()
        
        # Monitoramento e métricas
        self._active_threads = weakref.WeakSet()
        self._query_history = deque(maxlen=1000)
        self._error_history = deque(maxlen=100)
        
        # Configurações otimizadas do SQLite
        self._sqlite_config = {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": 10000,
            "temp_store": "MEMORY",
            "mmap_size": 268435456,  # 256MB
            "foreign_keys": "ON"
        }
        
        # Thread de limpeza
        self._cleanup_thread = None
        self._running = True
        self._start_cleanup_thread()
        
        # Inicializar banco
        self._initialize_database()
        
        logger.info(f"ThreadSafeConnectionManager inicializado para {self.db_path}")
    
    def _initialize_database(self):
        """Inicializa o banco de dados com esquema básico."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabela de métricas de conexão
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
                
                # Tabela de histórico de queries
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        thread_id INTEGER NOT NULL,
                        query_hash TEXT NOT NULL,
                        execution_time REAL NOT NULL,
                        success BOOLEAN NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("Banco de dados inicializado com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise
    
    def _create_connection(self) -> sqlite3.Connection:
        """Cria uma nova conexão SQLite otimizada."""
        thread_id = threading.get_ident()
        
        try:
            # Criar conexão
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            
            # Aplicar configurações otimizadas
            cursor = conn.cursor()
            for pragma, value in self._sqlite_config.items():
                cursor.execute(f"PRAGMA {pragma} = {value}")
            
            # Registrar estatísticas
            stats = ConnectionStats(
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
                self._close_thread_connection()
                conn = self._get_thread_connection()
            
            yield conn
            
        except Exception as e:
            error_occurred = True
            logger.error(f"Erro na conexão da thread {thread_id}: {e}")
            self._record_error(thread_id, str(e))
            raise
            
        finally:
            execution_time = time.time() - start_time
            
            # Atualizar estatísticas
            if thread_id in self._connection_stats:
                self._connection_stats[thread_id].update_usage(
                    execution_time=execution_time,
                    error=error_occurred
                )
            
            # Registrar na história de queries
            self._record_query(thread_id, execution_time, not error_occurred)
    
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
                try:
                    self._connections_pool[thread_id].close()
                except Exception as e:
                    logger.warning(f"Erro ao fechar conexão do pool: {e}")
                finally:
                    del self._connections_pool[thread_id]
            
            if thread_id in self._connection_stats:
                self._connection_stats[thread_id].is_active = False
    
    def _record_query(self, thread_id: int, execution_time: float, success: bool):
        """Registra execução de query no histórico."""
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'execution_time': execution_time,
            'success': success
        }
        
        self._query_history.append(query_record)
    
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
                    self._save_metrics()
                    time.sleep(60)  # Cleanup a cada minuto
                except Exception as e:
                    logger.error(f"Erro na thread de cleanup: {e}")
        
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
            for thread_id, stats in list(self._connection_stats.items()):
                # Verificar se thread ainda existe
                thread_exists = any(t.ident == thread_id for t in threading.enumerate())
                
                # Verificar timeout
                age = (current_time - stats.last_used).total_seconds()
                is_old = age > self.connection_timeout
                
                if not thread_exists or is_old:
                    orphaned_threads.append(thread_id)
        
        # Fechar conexões órfãs
        for thread_id in orphaned_threads:
            self._close_orphaned_connection(thread_id)
            logger.info(f"Conexão órfã removida para thread {thread_id}")
    
    def _close_orphaned_connection(self, thread_id: int):
        """Fecha conexão órfã específica."""
        with self._lock:
            if thread_id in self._connections_pool:
                try:
                    self._connections_pool[thread_id].close()
                except Exception as e:
                    logger.warning(f"Erro ao fechar conexão órfã {thread_id}: {e}")
                finally:
                    del self._connections_pool[thread_id]
            
            if thread_id in self._connection_stats:
                self._connection_stats[thread_id].is_active = False
    
    def _save_metrics(self):
        """Salva métricas de performance no banco."""
        if not self._connection_stats:
            return
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for thread_id, stats in self._connection_stats.items():
                    if stats.is_active:
                        cursor.execute("""
                            INSERT INTO connection_metrics (
                                timestamp, thread_id, query_count, 
                                avg_execution_time, errors, connection_age
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().isoformat(),
                            thread_id,
                            stats.query_count,
                            stats.avg_execution_time,
                            stats.errors,
                            stats.age_seconds
                        ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das conexões."""
        with self._lock:
            active_connections = sum(1 for stats in self._connection_stats.values() if stats.is_active)
            total_queries = sum(stats.query_count for stats in self._connection_stats.values())
            total_errors = sum(stats.errors for stats in self._connection_stats.values())
            
            return {
                'active_connections': active_connections,
                'total_connections': len(self._connection_stats),
                'max_connections': self.max_connections,
                'total_queries': total_queries,
                'total_errors': total_errors,
                'error_rate': total_errors / max(1, total_queries) * 100,
                'active_threads': len(self._active_threads),
                'query_history_size': len(self._query_history),
                'error_history_size': len(self._error_history)
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gera relatório detalhado de performance."""
        stats = self.get_connection_stats()
        
        # Análise de queries recentes
        recent_queries = list(self._query_history)[-100:]  # Últimas 100 queries
        if recent_queries:
            execution_times = [q['execution_time'] for q in recent_queries]
            avg_execution_time = sum(execution_times) / len(execution_times)
            max_execution_time = max(execution_times)
            min_execution_time = min(execution_times)
        else:
            avg_execution_time = max_execution_time = min_execution_time = 0
        
        # Análise de erros recentes
        recent_errors = list(self._error_history)[-10:]  # Últimos 10 erros
        
        return {
            'connection_stats': stats,
            'performance_metrics': {
                'avg_execution_time': avg_execution_time,
                'max_execution_time': max_execution_time,
                'min_execution_time': min_execution_time,
                'recent_queries_analyzed': len(recent_queries)
            },
            'recent_errors': recent_errors,
            'database_path': str(self.db_path),
            'sqlite_config': self._sqlite_config,
            'report_timestamp': datetime.now().isoformat()
        }
    
    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Executa query de forma thread-safe com métricas."""
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def close_all_connections(self):
        """Fecha todas as conexões e para threads de cleanup."""
        self._running = False
        
        # Aguardar thread de cleanup
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Fechar todas as conexões
        with self._lock:
            for thread_id in list(self._connections_pool.keys()):
                self._close_orphaned_connection(thread_id)
        
        # Fechar conexão local se existir
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar conexão local: {e}")
            finally:
                self._local.connection = None
        
        logger.info("Todas as conexões foram fechadas")
    
    def __del__(self):
        """Cleanup automático."""
        try:
            self.close_all_connections()
        except Exception:
            pass


# Singleton global para o gerenciador de conexões
_connection_manager: Optional[ThreadSafeConnectionManager] = None
_manager_lock = threading.Lock()


def get_connection_manager(db_path: str = "monitoring.db") -> ThreadSafeConnectionManager:
    """Obtém instância singleton do gerenciador de conexões."""
    global _connection_manager
    
    if _connection_manager is None:
        with _manager_lock:
            if _connection_manager is None:
                _connection_manager = ThreadSafeConnectionManager(db_path)
    
    return _connection_manager


def close_connection_manager():
    """Fecha o gerenciador de conexões global."""
    global _connection_manager
    
    if _connection_manager is not None:
        with _manager_lock:
            if _connection_manager is not None:
                _connection_manager.close_all_connections()
                _connection_manager = None


if __name__ == "__main__":
    # Teste do gerenciador de conexões
    print("Testando ThreadSafeConnectionManager...")
    
    manager = get_connection_manager("test_connections.db")
    
    # Teste básico
    with manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)")
        cursor.execute("INSERT INTO test (value) VALUES (?)", ("teste",))
        conn.commit()
    
    # Testar estatísticas
    stats = manager.get_connection_stats()
    print(f"Estatísticas: {stats}")
    
    # Testar relatório de performance
    report = manager.get_performance_report()
    print(f"Relatório de performance: {json.dumps(report, indent=2)}")
    
    # Cleanup
    close_connection_manager()
    print("Teste concluído com sucesso!")
