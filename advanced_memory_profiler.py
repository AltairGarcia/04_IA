#!/usr/bin/env python3
"""
Sistema Avançado de Profiling de Memória para LangGraph 101

Implementa ferramentas robustas de análise de memória usando:
- tracemalloc (built-in Python)
- memory_profiler (external)
- psutil para métricas do sistema
- gc para análise de garbage collection
- objgraph para análise de objetos

Author: GitHub Copilot
Date: 2025-05-27
"""

import os
import gc
import sys
import time
import psutil
import threading
import logging
import sqlite3
import tracemalloc
import weakref
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import uuid
import platform

# Try to import platform-specific modules
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    logging.warning("resource module not available (Windows platform)")

# Try to import external libraries
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logging.warning("memory_profiler not available. Install with: pip install memory-profiler")

try:
    import objgraph
    OBJGRAPH_AVAILABLE = True
except ImportError:
    OBJGRAPH_AVAILABLE = False
    logging.warning("objgraph not available. Install with: pip install objgraph")

try:
    import pympler
    from pympler import tracker, muppy, summary
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False
    logging.warning("pympler not available. Install with: pip install pympler")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot de memória detalhado."""
    timestamp: datetime
    process_memory_mb: float
    system_memory_percent: float
    system_memory_available_gb: float
    gc_stats: Dict[str, int]
    tracemalloc_top: List[Dict[str, Any]]
    object_counts: Dict[str, int]
    thread_count: int
    file_descriptors: int
    stack_size_kb: int
    peak_memory_mb: float


@dataclass
class MemoryLeak:
    """Detecção de vazamento de memória."""
    object_type: str
    count_increase: int
    size_increase_mb: float
    first_seen: datetime
    last_seen: datetime
    severity: str  # low, medium, high, critical


@dataclass
class MemoryHotspot:
    """Hotspot de uso de memória."""
    filename: str
    line_number: int
    function_name: str
    size_mb: float
    count: int
    traceback: List[str]


class AdvancedMemoryProfiler:
    """
    Sistema avançado de profiling de memória.
    
    Combina múltiplas ferramentas de análise para detectar:
    - Vazamentos de memória
    - Hotspots de uso
    - Objetos não coletados pelo GC
    - Fragmentação de memória
    - Performance de alocação/desalocação
    """
    
    def __init__(self, 
                 db_path: str = "memory_profiling.db",
                 snapshot_interval: int = 30,
                 enable_tracemalloc: bool = True,
                 max_snapshots: int = 1000):
        
        self.db_path = Path(db_path)
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.conn: Optional[sqlite3.Connection] = None  # Initialize conn
        self.enable_tracemalloc = enable_tracemalloc
        
        # Estado do profiler
        self.is_running = False
        self.profiling_thread = None
        self._lock = threading.RLock()
        
        # Dados coletados
        self.snapshots = deque(maxlen=max_snapshots)
        self.leak_detections = {}
        self.hotspots = []
        self.baseline_snapshot = None
        
        # Trackers externos
        self.pympler_tracker = None
        if PYMPLER_AVAILABLE:
            self.pympler_tracker = tracker.SummaryTracker()
        
        # Configurar tracemalloc
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Mantém stack traces de até 25 frames
            logger.info("Tracemalloc iniciado em __init__")
        
        # Inicializar banco
        self._initialize_database()
        
        logger.info(f"AdvancedMemoryProfiler inicializado")
    
    def _initialize_database(self):
        """Initialize the database and schema."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            cursor = self.conn.cursor()
            # Example table creation (adjust as per actual schema)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    process_memory_mb REAL,
                    system_memory_percent REAL,
                    system_memory_available_gb REAL,
                    gc_stats_json TEXT,
                    tracemalloc_top_json TEXT,
                    object_counts_json TEXT,
                    thread_count INTEGER,
                    file_descriptors INTEGER,
                    stack_size_kb REAL,
                    peak_memory_mb REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_type TEXT,
                    count_increase INTEGER,
                    size_increase_mb REAL,
                    first_seen TEXT,
                    last_seen TEXT,
                    severity TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hotspots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    line_number INTEGER,
                    function_name TEXT,
                    size_mb REAL,
                    count INTEGER,
                    traceback_json TEXT
                )
            """)
            self.conn.commit()
            logger.info(f"Database initialized/schema verified at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
            # Re-raise or handle as appropriate for the application
            raise
    
    def start_profiling(self, baseline: bool = True):
        """Inicia o profiling de memória."""
        if self.is_running:
            logger.warning("Profiling já está ativo")
            return

        # Ensure tracemalloc is started if enabled and not already tracing.
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)
            logger.info("Tracemalloc iniciado em start_profiling")
        
        self.is_running = True
        
        # Capturar baseline se solicitado
        if baseline:
            self.baseline_snapshot = self._take_snapshot()
            logger.info("Baseline snapshot capturado")
        
        # Iniciar thread de profiling
        self.profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True,
            name="AdvancedMemoryProfiler"
        )
        self.profiling_thread.start()
        
        logger.info("Profiling de memória iniciado")
    
    def stop_profiling(self):
        """Para o profiling de memória."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.profiling_thread:
            self.profiling_thread.join(timeout=10)
        
        # Gerar relatório final
        final_report = self.generate_comprehensive_report()
        
        if self.conn: # Ensure connection is closed if profiling stops
            logger.info("Closing database connection during stop_profiling.")
            self.conn.close()
            self.conn = None
        
        logger.info("Profiling de memória parado")
        return final_report
    
    def cleanup(self):
        """Clean up resources, like closing the database connection and stopping tracemalloc."""
        logger.info(f"Iniciando cleanup do AdvancedMemoryProfiler ({id(self)})...")
        
        # Stop the profiling thread if it's running
        if self.is_running:
            self.is_running = False # Signal the loop to stop
            if self.profiling_thread and self.profiling_thread.is_alive():
                logger.info("Aguardando a thread de profiling terminar...")
                try:
                    self.profiling_thread.join(timeout=5.0) # Wait for 5 seconds
                    if self.profiling_thread.is_alive():
                        logger.warning("Thread de profiling não terminou a tempo.")
                    else:
                        logger.info("Thread de profiling terminada.")
                except Exception as e:
                    logger.error(f"Erro ao aguardar a thread de profiling: {e}")
            self.profiling_thread = None

        # Close the database connection
        if self.conn:
            logger.info(f"Fechando conexão com o banco de dados: {self.db_path}")
            try:
                self.conn.close()
                logger.info("Conexão com o banco de dados fechada com sucesso.")
            except sqlite3.Error as e:
                logger.error(f"Erro ao fechar conexão com o banco de dados: {e}")
            finally:
                self.conn = None # Ensure conn is None even if close fails
        else:
            logger.info("Nenhuma conexão com o banco de dados para fechar em cleanup.")

        # Stop tracemalloc
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("Tracemalloc parado via cleanup.")
        
        logger.info(f"Cleanup do AdvancedMemoryProfiler ({id(self)}) concluído.")

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        if hasattr(self, 'cleanup') and callable(self.cleanup):
            self.cleanup()

    def _profiling_loop(self):
        """Loop principal de profiling."""
        while self.is_running:
            try:
                # Capturar snapshot
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self.snapshots.append(snapshot)
                
                # Salvar no banco
                self._save_snapshot(snapshot)
                
                # Detectar vazamentos
                if len(self.snapshots) >= 5:  # Precisamos de histórico
                    leaks = self._detect_memory_leaks()
                    for leak in leaks:
                        self._save_leak(leak)
                
                # Detectar hotspots
                hotspots = self._detect_hotspots()
                for hotspot in hotspots:
                    self._save_hotspot(hotspot)
                
                # Log status periodicamente
                if len(self.snapshots) % 10 == 0:
                    self._log_profiling_status()
                
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de profiling: {e}")
                time.sleep(60)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Captura snapshot detalhado de memória."""
        timestamp = datetime.now()
        
        # Informações do processo
        process = psutil.Process()
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / 1024 / 1024
        
        # Informações do sistema
        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        system_memory_available_gb = system_memory.available / 1024 / 1024 / 1024
        
        # Estatísticas do GC
        gc_stats = {
            f"generation_{i}": len(gc.get_objects(i)) 
            for i in range(3)
        }
        gc_stats["total_objects"] = len(gc.get_objects())
        gc_stats["collected"] = gc.collect()
        
        # Tracemalloc top allocations
        tracemalloc_top = []
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_top.append({
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024
            })
            
            # Top 10 alocações
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                tracemalloc_top.append({
                    "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
        
        # Contagem de objetos por tipo
        object_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
        top_objects = dict(object_counts.most_common(20))
        
        # Informações do sistema
        thread_count = threading.active_count()
        
        # File descriptors (Unix only)
        file_descriptors = 0
        try:
            if hasattr(process, "num_fds"):
                file_descriptors = process.num_fds()
        except:
            pass
          # Stack size
        stack_size_kb = 0
        try:
            if RESOURCE_AVAILABLE:
                stack_size_kb = resource.getrlimit(resource.RLIMIT_STACK)[0] / 1024
        except:
            pass
        
        # Peak memory
        peak_memory_mb = process_memory_mb
        if hasattr(process_memory, 'peak_wset'):
            peak_memory_mb = process_memory.peak_wset / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=timestamp,
            process_memory_mb=process_memory_mb,
            system_memory_percent=system_memory_percent,
            system_memory_available_gb=system_memory_available_gb,
            gc_stats=gc_stats,
            tracemalloc_top=tracemalloc_top,
            object_counts=top_objects,
            thread_count=thread_count,
            file_descriptors=file_descriptors,
            stack_size_kb=stack_size_kb,
            peak_memory_mb=peak_memory_mb
        )
    
    def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detecta vazamentos de memória comparando snapshots."""
        if len(self.snapshots) < 5:
            return []
        
        leaks = []
        recent_snapshots = list(self.snapshots)[-5:]  # Últimos 5 snapshots
        
        # Comparar contagens de objetos
        baseline_objects = recent_snapshots[0].object_counts
        current_objects = recent_snapshots[-1].object_counts
        
        for obj_type, current_count in current_objects.items():
            baseline_count = baseline_objects.get(obj_type, 0)
            count_increase = current_count - baseline_count
            
            # Detectar aumento significativo
            if count_increase > 100 and count_increase > baseline_count * 0.5:
                # Estimar tamanho (aproximação)
                size_increase_mb = count_increase * 0.001  # Estimativa conservadora
                
                severity = "low"
                if count_increase > 1000:
                    severity = "high"
                elif count_increase > 500:
                    severity = "medium"
                
                leak = MemoryLeak(
                    object_type=obj_type,
                    count_increase=count_increase,
                    size_increase_mb=size_increase_mb,
                    first_seen=recent_snapshots[0].timestamp,
                    last_seen=recent_snapshots[-1].timestamp,
                    severity=severity
                )
                leaks.append(leak)
        
        return leaks
    
    def _detect_hotspots(self) -> List[MemoryHotspot]:
        """Detecta hotspots de uso de memória."""
        hotspots = []
        
        if not tracemalloc.is_tracing():
            return hotspots
        
        try:
            snapshot = tracemalloc.take_snapshot()
            # Filtrar por tamanho e agrupar por traceback para identificar hotspots reais
            # Aumentar o limite para top_stats para ter uma visão mais ampla
            top_stats = snapshot.statistics('traceback')[:50] 
            
            processed_hotspots = {}

            for stat in top_stats:
                if stat.size < 1024 * 1024:  # Ignorar < 1MB
                    continue
                
                # Usar o traceback formatado como chave para agrupar alocações similares
                traceback_key_list = stat.traceback.format()
                traceback_key = "\\n".join(traceback_key_list) # Chave mais robusta

                if traceback_key not in processed_hotspots:
                    frame = stat.traceback[-1] # Frame mais específico
                    filename = frame.filename
                    line_number = frame.lineno
                    # Tentar extrair o nome da função do traceback, se possível
                    # Isso pode ser complexo, então uma abordagem simples é usada aqui
                    function_name = "unknown" 
                    try:
                        # Tenta pegar o nome da função da última linha do traceback formatado
                        last_trace_line = traceback_key_list[-1].strip()
                        if " in " in last_trace_line:
                            function_name = last_trace_line.split(" in ")[-1]
                            if function_name.startswith("<"): # Ex: <module>
                                function_name = "unknown" 
                    except Exception:
                        pass # Mantém unknown se a extração falhar

                    processed_hotspots[traceback_key] = MemoryHotspot(
                        filename=filename,
                        line_number=line_number,
                        function_name=function_name, 
                        size_mb=0, # Será agregado
                        count=0,   # Será agregado
                        traceback=traceback_key_list 
                    )
                
                processed_hotspots[traceback_key].size_mb += stat.size / 1024 / 1024
                processed_hotspots[traceback_key].count += stat.count
            
            # Converter para lista e ordenar por tamanho
            hotspots = sorted(list(processed_hotspots.values()), key=lambda h: h.size_mb, reverse=True)[:20]

        except Exception as e:
            logger.error(f"Erro ao detectar hotspots: {e}")
        
        return hotspots
    
    def _save_snapshot(self, snapshot: MemorySnapshot):
        """Save a memory snapshot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save snapshot.")
            return

        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'snapshots'
            cursor.execute("""
                INSERT INTO snapshots (
                    timestamp, process_memory_mb, system_memory_percent,
                    system_memory_available_gb, gc_stats_json, tracemalloc_top_json,
                    object_counts_json, thread_count, file_descriptors,
                    stack_size_kb, peak_memory_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                snapshot.process_memory_mb,
                snapshot.system_memory_percent,
                snapshot.system_memory_available_gb,
                json.dumps(snapshot.gc_stats),
                json.dumps(snapshot.tracemalloc_top),
                json.dumps(snapshot.object_counts),
                snapshot.thread_count,
                snapshot.file_descriptors,
                snapshot.stack_size_kb,
                snapshot.peak_memory_mb
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def _save_leak(self, leak: MemoryLeak):
        """Save a detected memory leak to the database."""
        if not self.conn:
            logger.warning("No database connection available to save leak.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks' e campo detection_timestamp
            cursor.execute("""
                INSERT INTO leaks (
                    object_type, count_increase, size_increase_mb,
                    first_seen, last_seen, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                leak.object_type,
                leak.count_increase,
                leak.size_increase_mb,
                leak.first_seen.isoformat(),
                leak.last_seen.isoformat(),
                leak.severity
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving leak: {e}")
    
    def _save_hotspot(self, hotspot: MemoryHotspot):
        """Save a detected memory hotspot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save hotspot.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute("""
                INSERT INTO hotspots (
                    filename, line_number, function_name,
                    size_mb, count, traceback_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                hotspot.filename,
                hotspot.line_number,
                hotspot.function_name,
                hotspot.size_mb,
                hotspot.count,
                json.dumps(hotspot.traceback) # Salvar traceback como JSON
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving hotspot: {e}")

    def _analyze_leaks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored leak data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for leak analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks'
            cursor.execute(f"""
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_mb,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       GROUP_CONCAT(severity) as severities
                FROM leaks
                GROUP BY object_type
                ORDER BY total_size_mb DESC, total_increase DESC
                LIMIT {limit}
            """)
            leaks_data = []
            for row in cursor.fetchall():
                leaks_data.append({
                    "object_type": row[0],
                    "total_increase": row[1],
                    "total_size_mb": row[2],
                    "first_occurrence": row[3],
                    "last_occurrence": row[4],
                    "severities": list(set(row[5].split(','))) if row[5] else []
                })
            return leaks_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing leaks: {e}")
            return []

    def _analyze_hotspots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored hotspot data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for hotspot analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute(f"""
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size_mb, 
                       SUM(count) as total_count,
                       GROUP_CONCAT(traceback_json) as tracebacks_json_array
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size_mb DESC, total_count DESC
                LIMIT {limit}
            """)
            hotspots_data = []
            for row in cursor.fetchall():
                # Processar os tracebacks concatenados
                # Cada traceback_json é uma string JSON de uma lista.
                # GROUP_CONCAT junta essas strings com vírgula.
                # Precisamos parsear cada um individualmente se quisermos a lista de listas.
                # Ou, mais simples, pegar o primeiro ou o mais comum se for o caso.
                # Aqui, vamos apenas tentar carregar o primeiro se houver muitos.
                raw_tracebacks = row[5]
                tracebacks = []
                if raw_tracebacks:
                    # Tentativa de parsear o JSON concatenado.
                    # Isso pode ser problemático se os JSONs não estiverem bem formados
                    # ou se a concatenação criar um JSON inválido.
                    # Uma abordagem mais segura seria armazenar tracebacks de forma diferente
                    # ou processá-los antes do GROUP_CONCAT.
                    # Para este exemplo, vamos assumir que cada traceback é uma string JSON válida
                    # e tentar parsear a primeira.
                    try:
                        # Isso provavelmente não funcionará como esperado com GROUP_CONCAT
                        # de múltiplas listas JSON.
                        # tracebacks = json.loads(f'[{raw_tracebacks}]')
                        # Uma abordagem mais simples: pegar a primeira string JSON e parseá-la
                        first_traceback_json = raw_tracebacks.split('],[')[0] + ']' if '],[' in raw_tracebacks else raw_tracebacks
                        if not first_traceback_json.startswith('['):
                            first_traceback_json = '[' + first_traceback_json
                        if not first_traceback_json.endswith(']'):
                             # Pode já ter sido cortado, ou ser um único JSON
                            if not first_traceback_json.endswith('"}'): # fim de um objeto json
                                first_traceback_json = first_traceback_json + ']'
                        

                        # Sanitize common issues with concatenated JSON strings
                        # This is a common issue with GROUP_CONCAT of JSON strings.
                        # Example: "[...], [...]" needs to become "[[...], [...]]"
                        # Or, if they are distinct JSON objects: "{...}, {...}" needs to become "[{...}, {...}]"
                        # For now, we'll just try to parse the first one if it's a list of strings.
                        
                        # Simplificando: apenas pegamos a string bruta por enquanto
                        # A lógica de reconstrução de JSON a partir de GROUP_CONCAT é complexa
                        # e depende de como os dados foram inseridos.
                        # Se cada `traceback_json` era `json.dumps(["line1", "line2"])`
                        # então `GROUP_CONCAT` produz algo como:
                        # '["lineA1", "lineA2"],["lineB1", "lineB2"]'
                        # Para transformar isso em um JSON válido de lista de listas:
                        # `json.loads('[' + raw_tracebacks + ']')`
                        
                        # Se o traceback_json já é uma string representando uma lista de strings
                        # e o GROUP_CONCAT as une, precisamos de uma estratégia.
                        # Assumindo que cada traceback_json é uma lista de strings serializada:
                        # e.g., '["file:1", "file:2"]'
                        # GROUP_CONCAT resultaria em: '["file:1", "file:2"],["file:3", "file:4"]'
                        # Para parsear isso como uma lista de listas de strings:
                        parsed_tracebacks = []
                        current_json_str = ""
                        depth = 0
                        for char in raw_tracebacks:
                            current_json_str += char
                            if char == '[':
                                depth += 1
                            elif char == ']':
                                depth -= 1
                                if depth == 0 and current_json_str:
                                    try:
                                        parsed_tracebacks.append(json.loads(current_json_str))
                                        current_json_str = ""
                                    except json.JSONDecodeError:
                                        # Ignorar se uma sub-string não é JSON válido, pode acontecer com GROUP_CONCAT
                                        current_json_str = "" # Reset
                                        pass 
                        if parsed_tracebacks:
                             tracebacks = parsed_tracebacks[0] # Pegar o primeiro traceback completo

                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not parse tracebacks for hotspot {row[0]}:{row[1]}: {je}. Raw: {raw_tracebacks[:100]}")
                        tracebacks = [raw_tracebacks] # fallback to raw string if parsing fails

                hotspots_data.append({
                    "filename": row[0],
                    "line_number": row[1],
                    "function_name": row[2],
                    "total_size_mb": row[3],
                    "total_count": row[4],
                    "tracebacks": tracebacks # Idealmente, uma lista de strings (o primeiro traceback)
                })
            return hotspots_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing hotspots: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """Gera um resumo rápido do estado atual da memória."""
        if not self.snapshots:
            return {"error": "Nenhum snapshot disponível"}
        
        current = self.snapshots[-1]
        
        summary_report = {
            "timestamp": current.timestamp.isoformat(),
            "process_memory_mb": current.process_memory_mb,
            "system_memory_percent": current.system_memory_percent,
            "system_memory_available_gb": current.system_memory_available_gb,
            "total_objects": current.gc_stats["total_objects"],
            "active_threads": current.thread_count,
            "file_descriptors": current.file_descriptors,
            "peak_memory_mb": current.peak_memory_mb,
            "tracemalloc_enabled": tracemalloc.is_tracing(),
            "memory_profiler_available": MEMORY_PROFILER_AVAILABLE,
            "objgraph_available": OBJGRAPH_AVAILABLE,
            "pympler_available": PYMPLER_AVAILABLE
        }
        
        return summary_report

    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na análise de memória."""
        recommendations = []
        
        # Exemplo de recomendações baseadas em tendências de memória
        memory_trend = self._analyze_memory_trend()
        if memory_trend["trend_direction"] == "increasing":
            recommendations.append(
                "Atenção: O uso de memória está aumentando. Considere investigar objetos não coletados ou vazamentos de memória."
            )
        elif memory_trend["trend_direction"] == "decreasing":
            recommendations.append(
                "O uso de memória está diminuindo. Isso é bom, mas continue monitorando."
            )
        
        # Recomendações adicionais podem ser adicionadas aqui
        
        return recommendations

    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """Analisa tendência de uso de memória."""
        if len(self.snapshots) < 3:
            return {"error": "Insuficientes snapshots para análise de tendência"}
        
        memory_values = [s.process_memory_mb for s in self.snapshots]
        
        # Calcular tendência linear simples
        n = len(memory_values)
        x_values = list(range(n))
        
        # Coeficiente de correlação simples
        mean_x = sum(x_values) / n
        mean_y = sum(memory_values) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, memory_values))
        denominator_x = sum((x - mean_x) ** 2 for x in x_values)
        
        if denominator_x == 0:
            slope = 0
        else:
            slope = numerator / denominator_x
        
        trend_direction = "stable"
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        
        return {
            "trend_direction": trend_direction,
            "slope_mb_per_snapshot": slope,
            "min_memory_mb": min(memory_values),
            "max_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "memory_volatility": max(memory_values) - min(memory_values)
        }
    
    def _analyze_leaks(self, time_window_hours: int = 24) -> List[MemoryLeak]:
        """Analisa vazamentos de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_leaks.")
            return []

        leaks = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar vazamentos detectados
            # Esta é uma query de exemplo, ajuste conforme sua lógica de detecção
            query = """
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_increase,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       MAX(severity) as max_severity -- Pega a maior severidade
                FROM leaks
                WHERE datetime(last_seen) >= datetime('now', ?)
                GROUP BY object_type
                HAVING total_increase > 100 -- Exemplo de threshold
                ORDER BY total_size_increase DESC
            """
            
            # Calcula o tempo para a janela de análise
            time_threshold = f"-{time_window_hours} hours"
            
            cursor.execute(query, (time_threshold,))
            rows = cursor.fetchall()
            
            for row in rows:
                leak = MemoryLeak(
                    object_type=row[0],
                    count_increase=row[1],
                    size_increase_mb=row[2],
                    first_seen=datetime.fromisoformat(row[3]),
                    last_seen=datetime.fromisoformat(row[4]),
                    severity=row[5]
                )
                leaks.append(leak)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar vazamentos do banco: {e}")
        # Ensure leaks is always returned, even if an error occurs before its initialization in the try block.
        # However, it's initialized as [] before the try block, so this is fine.
        return leaks
    
    def _analyze_hotspots(self, top_n: int = 10) -> List[MemoryHotspot]:
        """Analisa hotspots de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_hotspots.")
            return []

        hotspots = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar hotspots
            # Esta é uma query de exemplo, ajuste conforme sua lógica
            query = """
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size, 
                       SUM(count) as total_count,
                       -- GROUP_CONCAT pode precisar de ajustes dependendo do dialeto SQL e do conteúdo
                       -- Para SQLite, GROUP_CONCAT(DISTINCT traceback_json) é uma abordagem.
                       -- Se traceback_json armazena uma lista JSON, pode ser necessário processamento posterior.
                       GROUP_CONCAT(traceback_json) as tracebacks_str 
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size DESC
                LIMIT ?
            """
            
            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()
            
            for row in rows:
                traceback_list = []
                if row[5]: # tracebacks_str
                    try:
                        # Tenta decodificar a string concatenada de JSONs
                        # Isso assume que cada traceback_json é uma string JSON válida de uma lista
                        # e que GROUP_CONCAT as une com vírgula.
                        # Se for uma única string JSON contendo todas as listas, ajuste o parse.
                        concatenated_tracebacks = row[5]
                        # Heurística para tentar separar JSONs concatenados se não for uma lista JSON válida
                        if concatenated_tracebacks.startswith('[') and concatenated_tracebacks.endswith(']') and concatenated_tracebacks.count('[') == 1:
                            traceback_list = json.loads(concatenated_tracebacks)
                        else:
                            # Tenta dividir e parsear individualmente se for algo como '[] []' ou '[][]'
                            # Isso é uma simplificação; uma solução robusta pode ser complexa
                            possible_jsons = concatenated_tracebacks.replace('][', ']#DELIM#[').split('#DELIM#')
                            for pj in possible_jsons:
                                traceback_list.extend(json.loads(pj))
                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not decode traceback JSON string: {row[5][:100]}... Error: {je}")
                        # Adiciona a string bruta se não puder ser parseada
                        traceback_list.append(str(row[5])) 

                hotspot = MemoryHotspot(
                    filename=row[0],
                    line_number=row[1],
                    function_name=row[2],
                    size_mb=row[3],
                    count=row[4],
                    traceback=traceback_list[:10] # Limita o traceback para exibição
                )
                hotspots.append(hotspot)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar hotspots do banco: {e}")
        # Ensure hotspots is always returned
        return hotspots

    def cleanup(self):
        """Clean up resources, like closing the database connection and stopping tracemalloc."""
        logger.info(f"Iniciando cleanup do AdvancedMemoryProfiler ({id(self)})...")
        
        # Stop the profiling thread if it's running
        if self.is_running:
            self.is_running = False # Signal the loop to stop
            if self.profiling_thread and self.profiling_thread.is_alive():
                logger.info("Aguardando a thread de profiling terminar...")
                try:
                    self.profiling_thread.join(timeout=5.0) # Wait for 5 seconds
                    if self.profiling_thread.is_alive():
                        logger.warning("Thread de profiling não terminou a tempo.")
                    else:
                        logger.info("Thread de profiling terminada.")
                except Exception as e:
                    logger.error(f"Erro ao aguardar a thread de profiling: {e}")
            self.profiling_thread = None

        # Close the database connection
        if self.conn:
            logger.info(f"Fechando conexão com o banco de dados: {self.db_path}")
            try:
                self.conn.close()
                logger.info("Conexão com o banco de dados fechada com sucesso.")
            except sqlite3.Error as e:
                logger.error(f"Erro ao fechar conexão com o banco de dados: {e}")
            finally:
                self.conn = None # Ensure conn is None even if close fails
        else:
            logger.info("Nenhuma conexão com o banco de dados para fechar em cleanup.")

        # Stop tracemalloc
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("Tracemalloc parado via cleanup.")
        
        logger.info(f"Cleanup do AdvancedMemoryProfiler ({id(self)}) concluído.")

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        if hasattr(self, 'cleanup') and callable(self.cleanup):
            self.cleanup()

    def _profiling_loop(self):
        """Loop principal de profiling."""
        while self.is_running:
            try:
                # Capturar snapshot
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self.snapshots.append(snapshot)
                
                # Salvar no banco
                self._save_snapshot(snapshot)
                
                # Detectar vazamentos
                if len(self.snapshots) >= 5:  # Precisamos de histórico
                    leaks = self._detect_memory_leaks()
                    for leak in leaks:
                        self._save_leak(leak)
                
                # Detectar hotspots
                hotspots = self._detect_hotspots()
                for hotspot in hotspots:
                    self._save_hotspot(hotspot)
                
                # Log status periodicamente
                if len(self.snapshots) % 10 == 0:
                    self._log_profiling_status()
                
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de profiling: {e}")
                time.sleep(60)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Captura snapshot detalhado de memória."""
        timestamp = datetime.now()
        
        # Informações do processo
        process = psutil.Process()
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / 1024 / 1024
        
        # Informações do sistema
        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        system_memory_available_gb = system_memory.available / 1024 / 1024 / 1024
        
        # Estatísticas do GC
        gc_stats = {
            f"generation_{i}": len(gc.get_objects(i)) 
            for i in range(3)
        }
        gc_stats["total_objects"] = len(gc.get_objects())
        gc_stats["collected"] = gc.collect()
        
        # Tracemalloc top allocations
        tracemalloc_top = []
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_top.append({
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024
            })
            
            # Top 10 alocações
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                tracemalloc_top.append({
                    "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
        
        # Contagem de objetos por tipo
        object_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
        top_objects = dict(object_counts.most_common(20))
        
        # Informações do sistema
        thread_count = threading.active_count()
        
        # File descriptors (Unix only)
        file_descriptors = 0
        try:
            if hasattr(process, "num_fds"):
                file_descriptors = process.num_fds()
        except:
            pass
          # Stack size
        stack_size_kb = 0
        try:
            if RESOURCE_AVAILABLE:
                stack_size_kb = resource.getrlimit(resource.RLIMIT_STACK)[0] / 1024
        except:
            pass
        
        # Peak memory
        peak_memory_mb = process_memory_mb
        if hasattr(process_memory, 'peak_wset'):
            peak_memory_mb = process_memory.peak_wset / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=timestamp,
            process_memory_mb=process_memory_mb,
            system_memory_percent=system_memory_percent,
            system_memory_available_gb=system_memory_available_gb,
            gc_stats=gc_stats,
            tracemalloc_top=tracemalloc_top,
            object_counts=top_objects,
            thread_count=thread_count,
            file_descriptors=file_descriptors,
            stack_size_kb=stack_size_kb,
            peak_memory_mb=peak_memory_mb
        )
    
    def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detecta vazamentos de memória comparando snapshots."""
        if len(self.snapshots) < 5:
            return []
        
        leaks = []
        recent_snapshots = list(self.snapshots)[-5:]  # Últimos 5 snapshots
        
        # Comparar contagens de objetos
        baseline_objects = recent_snapshots[0].object_counts
        current_objects = recent_snapshots[-1].object_counts
        
        for obj_type, current_count in current_objects.items():
            baseline_count = baseline_objects.get(obj_type, 0)
            count_increase = current_count - baseline_count
            
            # Detectar aumento significativo
            if count_increase > 100 and count_increase > baseline_count * 0.5:
                # Estimar tamanho (aproximação)
                size_increase_mb = count_increase * 0.001  # Estimativa conservadora
                
                severity = "low"
                if count_increase > 1000:
                    severity = "high"
                elif count_increase > 500:
                    severity = "medium"
                
                leak = MemoryLeak(
                    object_type=obj_type,
                    count_increase=count_increase,
                    size_increase_mb=size_increase_mb,
                    first_seen=recent_snapshots[0].timestamp,
                    last_seen=recent_snapshots[-1].timestamp,
                    severity=severity
                )
                leaks.append(leak)
        
        return leaks
    
    def _detect_hotspots(self) -> List[MemoryHotspot]:
        """Detecta hotspots de uso de memória."""
        hotspots = []
        
        if not tracemalloc.is_tracing():
            return hotspots
        
        try:
            snapshot = tracemalloc.take_snapshot()
            # Filtrar por tamanho e agrupar por traceback para identificar hotspots reais
            # Aumentar o limite para top_stats para ter uma visão mais ampla
            top_stats = snapshot.statistics('traceback')[:50] 
            
            processed_hotspots = {}

            for stat in top_stats:
                if stat.size < 1024 * 1024:  # Ignorar < 1MB
                    continue
                
                # Usar o traceback formatado como chave para agrupar alocações similares
                traceback_key_list = stat.traceback.format()
                traceback_key = "\\n".join(traceback_key_list) # Chave mais robusta

                if traceback_key not in processed_hotspots:
                    frame = stat.traceback[-1] # Frame mais específico
                    filename = frame.filename
                    line_number = frame.lineno
                    # Tentar extrair o nome da função do traceback, se possível
                    # Isso pode ser complexo, então uma abordagem simples é usada aqui
                    function_name = "unknown" 
                    try:
                        # Tenta pegar o nome da função da última linha do traceback formatado
                        last_trace_line = traceback_key_list[-1].strip()
                        if " in " in last_trace_line:
                            function_name = last_trace_line.split(" in ")[-1]
                            if function_name.startswith("<"): # Ex: <module>
                                function_name = "unknown" 
                    except Exception:
                        pass # Mantém unknown se a extração falhar

                    processed_hotspots[traceback_key] = MemoryHotspot(
                        filename=filename,
                        line_number=line_number,
                        function_name=function_name, 
                        size_mb=0, # Será agregado
                        count=0,   # Será agregado
                        traceback=traceback_key_list 
                    )
                
                processed_hotspots[traceback_key].size_mb += stat.size / 1024 / 1024
                processed_hotspots[traceback_key].count += stat.count
            
            # Converter para lista e ordenar por tamanho
            hotspots = sorted(list(processed_hotspots.values()), key=lambda h: h.size_mb, reverse=True)[:20]

        except Exception as e:
            logger.error(f"Erro ao detectar hotspots: {e}")
        
        return hotspots
    
    def _save_snapshot(self, snapshot: MemorySnapshot):
        """Save a memory snapshot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save snapshot.")
            return

        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'snapshots'
            cursor.execute("""
                INSERT INTO snapshots (
                    timestamp, process_memory_mb, system_memory_percent,
                    system_memory_available_gb, gc_stats_json, tracemalloc_top_json,
                    object_counts_json, thread_count, file_descriptors,
                    stack_size_kb, peak_memory_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                snapshot.process_memory_mb,
                snapshot.system_memory_percent,
                snapshot.system_memory_available_gb,
                json.dumps(snapshot.gc_stats),
                json.dumps(snapshot.tracemalloc_top),
                json.dumps(snapshot.object_counts),
                snapshot.thread_count,
                snapshot.file_descriptors,
                snapshot.stack_size_kb,
                snapshot.peak_memory_mb
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def _save_leak(self, leak: MemoryLeak):
        """Save a detected memory leak to the database."""
        if not self.conn:
            logger.warning("No database connection available to save leak.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks' e campo detection_timestamp
            cursor.execute("""
                INSERT INTO leaks (
                    object_type, count_increase, size_increase_mb,
                    first_seen, last_seen, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                leak.object_type,
                leak.count_increase,
                leak.size_increase_mb,
                leak.first_seen.isoformat(),
                leak.last_seen.isoformat(),
                leak.severity
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving leak: {e}")
    
    def _save_hotspot(self, hotspot: MemoryHotspot):
        """Save a detected memory hotspot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save hotspot.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute("""
                INSERT INTO hotspots (
                    filename, line_number, function_name,
                    size_mb, count, traceback_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                hotspot.filename,
                hotspot.line_number,
                hotspot.function_name,
                hotspot.size_mb,
                hotspot.count,
                json.dumps(hotspot.traceback) # Salvar traceback como JSON
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving hotspot: {e}")

    def _analyze_leaks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored leak data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for leak analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks'
            cursor.execute(f"""
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_mb,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       GROUP_CONCAT(severity) as severities
                FROM leaks
                GROUP BY object_type
                ORDER BY total_size_mb DESC, total_increase DESC
                LIMIT {limit}
            """)
            leaks_data = []
            for row in cursor.fetchall():
                leaks_data.append({
                    "object_type": row[0],
                    "total_increase": row[1],
                    "total_size_mb": row[2],
                    "first_occurrence": row[3],
                    "last_occurrence": row[4],
                    "severities": list(set(row[5].split(','))) if row[5] else []
                })
            return leaks_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing leaks: {e}")
            return []

    def _analyze_hotspots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored hotspot data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for hotspot analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute(f"""
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size_mb, 
                       SUM(count) as total_count,
                       GROUP_CONCAT(traceback_json) as tracebacks_json_array
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size_mb DESC, total_count DESC
                LIMIT {limit}
            """)
            hotspots_data = []
            for row in cursor.fetchall():
                # Processar os tracebacks concatenados
                # Cada traceback_json é uma string JSON de uma lista.
                # GROUP_CONCAT junta essas strings com vírgula.
                # Precisamos parsear cada um individualmente se quisermos a lista de listas.
                # Ou, mais simples, pegar o primeiro ou o mais comum se for o caso.
                # Aqui, vamos apenas tentar carregar o primeiro se houver muitos.
                raw_tracebacks = row[5]
                tracebacks = []
                if raw_tracebacks:
                    # Tentativa de parsear o JSON concatenado.
                    # Isso pode ser problemático se os JSONs não estiverem bem formados
                    # ou se a concatenação criar um JSON inválido.
                    # Uma abordagem mais segura seria armazenar tracebacks de forma diferente
                    # ou processá-los antes do GROUP_CONCAT.
                    # Para este exemplo, vamos assumir que cada traceback é uma string JSON válida
                    # e tentar parsear a primeira.
                    try:
                        # Isso provavelmente não funcionará como esperado com GROUP_CONCAT
                        # de múltiplas listas JSON.
                        # tracebacks = json.loads(f'[{raw_tracebacks}]')
                        # Uma abordagem mais simples: pegar a primeira string JSON e parseá-la
                        first_traceback_json = raw_tracebacks.split('],[')[0] + ']' if '],[' in raw_tracebacks else raw_tracebacks
                        if not first_traceback_json.startswith('['):
                            first_traceback_json = '[' + first_traceback_json
                        if not first_traceback_json.endswith(']'):
                             # Pode já ter sido cortado, ou ser um único JSON
                            if not first_traceback_json.endswith('"}'): # fim de um objeto json
                                first_traceback_json = first_traceback_json + ']'
                        

                        # Sanitize common issues with concatenated JSON strings
                        # This is a common issue with GROUP_CONCAT of JSON strings.
                        # Example: "[...], [...]" needs to become "[[...], [...]]"
                        # Or, if they are distinct JSON objects: "{...}, {...}" needs to become "[{...}, {...}]"
                        # For now, we'll just try to parse the first one if it's a list of strings.
                        
                        # Simplificando: apenas pegamos a string bruta por enquanto
                        # A lógica de reconstrução de JSON a partir de GROUP_CONCAT é complexa
                        # e depende de como os dados foram inseridos.
                        # Se cada `traceback_json` era `json.dumps(["line1", "line2"])`
                        # então `GROUP_CONCAT` produz algo como:
                        # '["lineA1", "lineA2"],["lineB1", "lineB2"]'
                        # Para transformar isso em um JSON válido de lista de listas:
                        # `json.loads('[' + raw_tracebacks + ']')`
                        
                        # Se o traceback_json já é uma string representando uma lista de strings
                        # e o GROUP_CONCAT as une, precisamos de uma estratégia.
                        # Assumindo que cada traceback_json é uma lista de strings serializada:
                        # e.g., '["file:1", "file:2"]'
                        # GROUP_CONCAT resultaria em: '["file:1", "file:2"],["file:3", "file:4"]'
                        # Para parsear isso como uma lista de listas de strings:
                        parsed_tracebacks = []
                        current_json_str = ""
                        depth = 0
                        for char in raw_tracebacks:
                            current_json_str += char
                            if char == '[':
                                depth += 1
                            elif char == ']':
                                depth -= 1
                                if depth == 0 and current_json_str:
                                    try:
                                        parsed_tracebacks.append(json.loads(current_json_str))
                                        current_json_str = ""
                                    except json.JSONDecodeError:
                                        # Ignorar se uma sub-string não é JSON válido, pode acontecer com GROUP_CONCAT
                                        current_json_str = "" # Reset
                                        pass 
                        if parsed_tracebacks:
                             tracebacks = parsed_tracebacks[0] # Pegar o primeiro traceback completo

                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not parse tracebacks for hotspot {row[0]}:{row[1]}: {je}. Raw: {raw_tracebacks[:100]}")
                        tracebacks = [raw_tracebacks] # fallback to raw string if parsing fails

                hotspots_data.append({
                    "filename": row[0],
                    "line_number": row[1],
                    "function_name": row[2],
                    "total_size_mb": row[3],
                    "total_count": row[4],
                    "tracebacks": tracebacks # Idealmente, uma lista de strings (o primeiro traceback)
                })
            return hotspots_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing hotspots: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """Gera um resumo rápido do estado atual da memória."""
        if not self.snapshots:
            return {"error": "Nenhum snapshot disponível"}
        
        current = self.snapshots[-1]
        
        summary_report = {
            "timestamp": current.timestamp.isoformat(),
            "process_memory_mb": current.process_memory_mb,
            "system_memory_percent": current.system_memory_percent,
            "system_memory_available_gb": current.system_memory_available_gb,
            "total_objects": current.gc_stats["total_objects"],
            "active_threads": current.thread_count,
            "file_descriptors": current.file_descriptors,
            "peak_memory_mb": current.peak_memory_mb,
            "tracemalloc_enabled": tracemalloc.is_tracing(),
            "memory_profiler_available": MEMORY_PROFILER_AVAILABLE,
            "objgraph_available": OBJGRAPH_AVAILABLE,
            "pympler_available": PYMPLER_AVAILABLE
        }
        
        return summary_report

    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na análise de memória."""
        recommendations = []
        
        # Exemplo de recomendações baseadas em tendências de memória
        memory_trend = self._analyze_memory_trend()
        if memory_trend["trend_direction"] == "increasing":
            recommendations.append(
                "Atenção: O uso de memória está aumentando. Considere investigar objetos não coletados ou vazamentos de memória."
            )
        elif memory_trend["trend_direction"] == "decreasing":
            recommendations.append(
                "O uso de memória está diminuindo. Isso é bom, mas continue monitorando."
            )
        
        # Recomendações adicionais podem ser adicionadas aqui
        
        return recommendations

    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """Analisa tendência de uso de memória."""
        if len(self.snapshots) < 3:
            return {"error": "Insuficientes snapshots para análise de tendência"}
        
        memory_values = [s.process_memory_mb for s in self.snapshots]
        
        # Calcular tendência linear simples
        n = len(memory_values)
        x_values = list(range(n))
        
        # Coeficiente de correlação simples
        mean_x = sum(x_values) / n
        mean_y = sum(memory_values) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, memory_values))
        denominator_x = sum((x - mean_x) ** 2 for x in x_values)
        
        if denominator_x == 0:
            slope = 0
        else:
            slope = numerator / denominator_x
        
        trend_direction = "stable"
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        
        return {
            "trend_direction": trend_direction,
            "slope_mb_per_snapshot": slope,
            "min_memory_mb": min(memory_values),
            "max_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "memory_volatility": max(memory_values) - min(memory_values)
        }
    
    def _analyze_leaks(self, time_window_hours: int = 24) -> List[MemoryLeak]:
        """Analisa vazamentos de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_leaks.")
            return []

        leaks = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar vazamentos detectados
            # Esta é uma query de exemplo, ajuste conforme sua lógica de detecção
            query = """
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_increase,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       MAX(severity) as max_severity -- Pega a maior severidade
                FROM leaks
                WHERE datetime(last_seen) >= datetime('now', ?)
                GROUP BY object_type
                HAVING total_increase > 100 -- Exemplo de threshold
                ORDER BY total_size_increase DESC
            """
            
            # Calcula o tempo para a janela de análise
            time_threshold = f"-{time_window_hours} hours"
            
            cursor.execute(query, (time_threshold,))
            rows = cursor.fetchall()
            
            for row in rows:
                leak = MemoryLeak(
                    object_type=row[0],
                    count_increase=row[1],
                    size_increase_mb=row[2],
                    first_seen=datetime.fromisoformat(row[3]),
                    last_seen=datetime.fromisoformat(row[4]),
                    severity=row[5]
                )
                leaks.append(leak)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar vazamentos do banco: {e}")
        # Ensure leaks is always returned, even if an error occurs before its initialization in the try block.
        # However, it's initialized as [] before the try block, so this is fine.
        return leaks
    
    def _analyze_hotspots(self, top_n: int = 10) -> List[MemoryHotspot]:
        """Analisa hotspots de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_hotspots.")
            return []

        hotspots = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar hotspots
            # Esta é uma query de exemplo, ajuste conforme sua lógica
            query = """
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size, 
                       SUM(count) as total_count,
                       -- GROUP_CONCAT pode precisar de ajustes dependendo do dialeto SQL e do conteúdo
                       -- Para SQLite, GROUP_CONCAT(DISTINCT traceback_json) é uma abordagem.
                       -- Se traceback_json armazena uma lista JSON, pode ser necessário processamento posterior.
                       GROUP_CONCAT(traceback_json) as tracebacks_str 
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size DESC
                LIMIT ?
            """
            
            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()
            
            for row in rows:
                traceback_list = []
                if row[5]: # tracebacks_str
                    try:
                        # Tenta decodificar a string concatenada de JSONs
                        # Isso assume que cada traceback_json é uma string JSON válida de uma lista
                        # e que GROUP_CONCAT as une com vírgula.
                        # Se for uma única string JSON contendo todas as listas, ajuste o parse.
                        concatenated_tracebacks = row[5]
                        # Heurística para tentar separar JSONs concatenados se não for uma lista JSON válida
                        if concatenated_tracebacks.startswith('[') and concatenated_tracebacks.endswith(']') and concatenated_tracebacks.count('[') == 1:
                            traceback_list = json.loads(concatenated_tracebacks)
                        else:
                            # Tenta dividir e parsear individualmente se for algo como '[] []' ou '[][]'
                            # Isso é uma simplificação; uma solução robusta pode ser complexa
                            possible_jsons = concatenated_tracebacks.replace('][', ']#DELIM#[').split('#DELIM#')
                            for pj in possible_jsons:
                                traceback_list.extend(json.loads(pj))
                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not decode traceback JSON string: {row[5][:100]}... Error: {je}")
                        # Adiciona a string bruta se não puder ser parseada
                        traceback_list.append(str(row[5])) 

                hotspot = MemoryHotspot(
                    filename=row[0],
                    line_number=row[1],
                    function_name=row[2],
                    size_mb=row[3],
                    count=row[4],
                    traceback=traceback_list[:10] # Limita o traceback para exibição
                )
                hotspots.append(hotspot)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar hotspots do banco: {e}")
        # Ensure hotspots is always returned
        return hotspots

    def cleanup(self):
        """Clean up resources, like closing the database connection and stopping tracemalloc."""
        logger.info(f"Iniciando cleanup do AdvancedMemoryProfiler ({id(self)})...")
        
        # Stop the profiling thread if it's running
        if self.is_running:
            self.is_running = False # Signal the loop to stop
            if self.profiling_thread and self.profiling_thread.is_alive():
                logger.info("Aguardando a thread de profiling terminar...")
                try:
                    self.profiling_thread.join(timeout=5.0) # Wait for 5 seconds
                    if self.profiling_thread.is_alive():
                        logger.warning("Thread de profiling não terminou a tempo.")
                    else:
                        logger.info("Thread de profiling terminada.")
                except Exception as e:
                    logger.error(f"Erro ao aguardar a thread de profiling: {e}")
            self.profiling_thread = None

        # Close the database connection
        if self.conn:
            logger.info(f"Fechando conexão com o banco de dados: {self.db_path}")
            try:
                self.conn.close()
                logger.info("Conexão com o banco de dados fechada com sucesso.")
            except sqlite3.Error as e:
                logger.error(f"Erro ao fechar conexão com o banco de dados: {e}")
            finally:
                self.conn = None # Ensure conn is None even if close fails
        else:
            logger.info("Nenhuma conexão com o banco de dados para fechar em cleanup.")

        # Stop tracemalloc
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("Tracemalloc parado via cleanup.")
        
        logger.info(f"Cleanup do AdvancedMemoryProfiler ({id(self)}) concluído.")

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        if hasattr(self, 'cleanup') and callable(self.cleanup):
            self.cleanup()

    def _profiling_loop(self):
        """Loop principal de profiling."""
        while self.is_running:
            try:
                # Capturar snapshot
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self.snapshots.append(snapshot)
                
                # Salvar no banco
                self._save_snapshot(snapshot)
                
                # Detectar vazamentos
                if len(self.snapshots) >= 5:  # Precisamos de histórico
                    leaks = self._detect_memory_leaks()
                    for leak in leaks:
                        self._save_leak(leak)
                
                # Detectar hotspots
                hotspots = self._detect_hotspots()
                for hotspot in hotspots:
                    self._save_hotspot(hotspot)
                
                # Log status periodicamente
                if len(self.snapshots) % 10 == 0:
                    self._log_profiling_status()
                
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de profiling: {e}")
                time.sleep(60)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Captura snapshot detalhado de memória."""
        timestamp = datetime.now()
        
        # Informações do processo
        process = psutil.Process()
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / 1024 / 1024
        
        # Informações do sistema
        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        system_memory_available_gb = system_memory.available / 1024 / 1024 / 1024
        
        # Estatísticas do GC
        gc_stats = {
            f"generation_{i}": len(gc.get_objects(i)) 
            for i in range(3)
        }
        gc_stats["total_objects"] = len(gc.get_objects())
        gc_stats["collected"] = gc.collect()
        
        # Tracemalloc top allocations
        tracemalloc_top = []
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_top.append({
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024
            })
            
            # Top 10 alocações
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                tracemalloc_top.append({
                    "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
        
        # Contagem de objetos por tipo
        object_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
        top_objects = dict(object_counts.most_common(20))
        
        # Informações do sistema
        thread_count = threading.active_count()
        
        # File descriptors (Unix only)
        file_descriptors = 0
        try:
            if hasattr(process, "num_fds"):
                file_descriptors = process.num_fds()
        except:
            pass
          # Stack size
        stack_size_kb = 0
        try:
            if RESOURCE_AVAILABLE:
                stack_size_kb = resource.getrlimit(resource.RLIMIT_STACK)[0] / 1024
        except:
            pass
        
        # Peak memory
        peak_memory_mb = process_memory_mb
        if hasattr(process_memory, 'peak_wset'):
            peak_memory_mb = process_memory.peak_wset / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=timestamp,
            process_memory_mb=process_memory_mb,
            system_memory_percent=system_memory_percent,
            system_memory_available_gb=system_memory_available_gb,
            gc_stats=gc_stats,
            tracemalloc_top=tracemalloc_top,
            object_counts=top_objects,
            thread_count=thread_count,
            file_descriptors=file_descriptors,
            stack_size_kb=stack_size_kb,
            peak_memory_mb=peak_memory_mb
        )
    
    def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detecta vazamentos de memória comparando snapshots."""
        if len(self.snapshots) < 5:
            return []
        
        leaks = []
        recent_snapshots = list(self.snapshots)[-5:]  # Últimos 5 snapshots
        
        # Comparar contagens de objetos
        baseline_objects = recent_snapshots[0].object_counts
        current_objects = recent_snapshots[-1].object_counts
        
        for obj_type, current_count in current_objects.items():
            baseline_count = baseline_objects.get(obj_type, 0)
            count_increase = current_count - baseline_count
            
            # Detectar aumento significativo
            if count_increase > 100 and count_increase > baseline_count * 0.5:
                # Estimar tamanho (aproximação)
                size_increase_mb = count_increase * 0.001  # Estimativa conservadora
                
                severity = "low"
                if count_increase > 1000:
                    severity = "high"
                elif count_increase > 500:
                    severity = "medium"
                
                leak = MemoryLeak(
                    object_type=obj_type,
                    count_increase=count_increase,
                    size_increase_mb=size_increase_mb,
                    first_seen=recent_snapshots[0].timestamp,
                    last_seen=recent_snapshots[-1].timestamp,
                    severity=severity
                )
                leaks.append(leak)
        
        return leaks
    
    def _detect_hotspots(self) -> List[MemoryHotspot]:
        """Detecta hotspots de uso de memória."""
        hotspots = []
        
        if not tracemalloc.is_tracing():
            return hotspots
        
        try:
            snapshot = tracemalloc.take_snapshot()
            # Filtrar por tamanho e agrupar por traceback para identificar hotspots reais
            # Aumentar o limite para top_stats para ter uma visão mais ampla
            top_stats = snapshot.statistics('traceback')[:50] 
            
            processed_hotspots = {}

            for stat in top_stats:
                if stat.size < 1024 * 1024:  # Ignorar < 1MB
                    continue
                
                # Usar o traceback formatado como chave para agrupar alocações similares
                traceback_key_list = stat.traceback.format()
                traceback_key = "\\n".join(traceback_key_list) # Chave mais robusta

                if traceback_key not in processed_hotspots:
                    frame = stat.traceback[-1] # Frame mais específico
                    filename = frame.filename
                    line_number = frame.lineno
                    # Tentar extrair o nome da função do traceback, se possível
                    # Isso pode ser complexo, então uma abordagem simples é usada aqui
                    function_name = "unknown" 
                    try:
                        # Tenta pegar o nome da função da última linha do traceback formatado
                        last_trace_line = traceback_key_list[-1].strip()
                        if " in " in last_trace_line:
                            function_name = last_trace_line.split(" in ")[-1]
                            if function_name.startswith("<"): # Ex: <module>
                                function_name = "unknown" 
                    except Exception:
                        pass # Mantém unknown se a extração falhar

                    processed_hotspots[traceback_key] = MemoryHotspot(
                        filename=filename,
                        line_number=line_number,
                        function_name=function_name, 
                        size_mb=0, # Será agregado
                        count=0,   # Será agregado
                        traceback=traceback_key_list 
                    )
                
                processed_hotspots[traceback_key].size_mb += stat.size / 1024 / 1024
                processed_hotspots[traceback_key].count += stat.count
            
            # Converter para lista e ordenar por tamanho
            hotspots = sorted(list(processed_hotspots.values()), key=lambda h: h.size_mb, reverse=True)[:20]

        except Exception as e:
            logger.error(f"Erro ao detectar hotspots: {e}")
        
        return hotspots
    
    def _save_snapshot(self, snapshot: MemorySnapshot):
        """Save a memory snapshot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save snapshot.")
            return

        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'snapshots'
            cursor.execute("""
                INSERT INTO snapshots (
                    timestamp, process_memory_mb, system_memory_percent,
                    system_memory_available_gb, gc_stats_json, tracemalloc_top_json,
                    object_counts_json, thread_count, file_descriptors,
                    stack_size_kb, peak_memory_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                snapshot.process_memory_mb,
                snapshot.system_memory_percent,
                snapshot.system_memory_available_gb,
                json.dumps(snapshot.gc_stats),
                json.dumps(snapshot.tracemalloc_top),
                json.dumps(snapshot.object_counts),
                snapshot.thread_count,
                snapshot.file_descriptors,
                snapshot.stack_size_kb,
                snapshot.peak_memory_mb
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def _save_leak(self, leak: MemoryLeak):
        """Save a detected memory leak to the database."""
        if not self.conn:
            logger.warning("No database connection available to save leak.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks' e campo detection_timestamp
            cursor.execute("""
                INSERT INTO leaks (
                    object_type, count_increase, size_increase_mb,
                    first_seen, last_seen, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                leak.object_type,
                leak.count_increase,
                leak.size_increase_mb,
                leak.first_seen.isoformat(),
                leak.last_seen.isoformat(),
                leak.severity
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving leak: {e}")
    
    def _save_hotspot(self, hotspot: MemoryHotspot):
        """Save a detected memory hotspot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save hotspot.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute("""
                INSERT INTO hotspots (
                    filename, line_number, function_name,
                    size_mb, count, traceback_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                hotspot.filename,
                hotspot.line_number,
                hotspot.function_name,
                hotspot.size_mb,
                hotspot.count,
                json.dumps(hotspot.traceback) # Salvar traceback como JSON
            ))
            logger.error(f"Error saving hotspot: {e}")

    def _analyze_leaks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored leak data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for leak analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks'
            cursor.execute(f"""
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_mb,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       GROUP_CONCAT(severity) as severities
                FROM leaks
                GROUP BY object_type
                ORDER BY total_size_mb DESC, total_increase DESC
                LIMIT {limit}
            """)
            leaks_data = []
            for row in cursor.fetchall():
                leaks_data.append({
                    "object_type": row[0],
                    "total_increase": row[1],
                    "total_size_mb": row[2],
                    "first_occurrence": row[3],
                    "last_occurrence": row[4],
                    "severities": list(set(row[5].split(','))) if row[5] else []
                })
            return leaks_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing leaks: {e}")
            return []

    def _analyze_hotspots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored hotspot data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for hotspot analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute(f"""
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size_mb, 
                       SUM(count) as total_count,
                       GROUP_CONCAT(traceback_json) as tracebacks_json_array
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size_mb DESC, total_count DESC
                LIMIT {limit}
            """)
            hotspots_data = []
            for row in cursor.fetchall():
                # Processar os tracebacks concatenados
                # Cada traceback_json é uma string JSON de uma lista.
                # GROUP_CONCAT junta essas strings com vírgula.
                # Precisamos parsear cada um individualmente se quisermos a lista de listas.
                # Ou, mais simples, pegar o primeiro ou o mais comum se for o caso.
                # Aqui, vamos apenas tentar carregar o primeiro se houver muitos.
                raw_tracebacks = row[5]
                tracebacks = []
                if raw_tracebacks:
                    # Tentativa de parsear o JSON concatenado.
                    # Isso pode ser problemático se os JSONs não estiverem bem formados
                    # ou se a concatenação criar um JSON inválido.
                    # Uma abordagem mais segura seria armazenar tracebacks de forma diferente
                    # ou processá-los antes do GROUP_CONCAT.
                    # Para este exemplo, vamos assumir que cada traceback é uma string JSON válida
                    # e tentar parsear a primeira.
                    try:
                        # Isso provavelmente não funcionará como esperado com GROUP_CONCAT
                        # de múltiplas listas JSON.
                        # tracebacks = json.loads(f'[{raw_tracebacks}]')
                        # Uma abordagem mais simples: pegar a primeira string JSON e parseá-la
                        first_traceback_json = raw_tracebacks.split('],[')[0] + ']' if '],[' in raw_tracebacks else raw_tracebacks
                        if not first_traceback_json.startswith('['):
                            first_traceback_json = '[' + first_traceback_json
                        if not first_traceback_json.endswith(']'):
                             # Pode já ter sido cortado, ou ser um único JSON
                            if not first_traceback_json.endswith('"}'): # fim de um objeto json
                                first_traceback_json = first_traceback_json + ']'
                        

                        # Sanitize common issues with concatenated JSON strings
                        # This is a common issue with GROUP_CONCAT of JSON strings.
                        # Example: "[...], [...]" needs to become "[[...], [...]]"
                        # Or, if they are distinct JSON objects: "{...}, {...}" needs to become "[{...}, {...}]"
                        # For now, we'll just try to parse the first one if it's a list of strings.
                        
                        # Simplificando: apenas pegamos a string bruta por enquanto
                        # A lógica de reconstrução de JSON a partir de GROUP_CONCAT é complexa
                        # e depende de como os dados foram inseridos.
                        # Se cada `traceback_json` era `json.dumps(["line1", "line2"])`
                        # então `GROUP_CONCAT` produz algo como:
                        # '["lineA1", "lineA2"],["lineB1", "lineB2"]'
                        # Para transformar isso em um JSON válido de lista de listas:
                        # `json.loads('[' + raw_tracebacks + ']')`
                        
                        # Se o traceback_json já é uma string representando uma lista de strings
                        # e o GROUP_CONCAT as une, precisamos de uma estratégia.
                        # Assumindo que cada traceback_json é uma lista de strings serializada:
                        # e.g., '["file:1", "file:2"]'
                        # GROUP_CONCAT resultaria em: '["file:1", "file:2"],["file:3", "file:4"]'
                        # Para parsear isso como uma lista de listas de strings:
                        parsed_tracebacks = []
                        current_json_str = ""
                        depth = 0
                        for char in raw_tracebacks:
                            current_json_str += char
                            if char == '[':
                                depth += 1
                            elif char == ']':
                                depth -= 1
                                if depth == 0 and current_json_str:
                                    try:
                                        parsed_tracebacks.append(json.loads(current_json_str))
                                        current_json_str = ""
                                    except json.JSONDecodeError:
                                        # Ignorar se uma sub-string não é JSON válido, pode acontecer com GROUP_CONCAT
                                        current_json_str = "" # Reset
                                        pass 
                        if parsed_tracebacks:
                             tracebacks = parsed_tracebacks[0] # Pegar o primeiro traceback completo

                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not parse tracebacks for hotspot {row[0]}:{row[1]}: {je}. Raw: {raw_tracebacks[:100]}")
                        tracebacks = [raw_tracebacks] # fallback to raw string if parsing fails

                hotspots_data.append({
                    "filename": row[0],
                    "line_number": row[1],
                    "function_name": row[2],
                    "total_size_mb": row[3],
                    "total_count": row[4],
                    "tracebacks": tracebacks # Idealmente, uma lista de strings (o primeiro traceback)
                })
            return hotspots_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing hotspots: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """Gera um resumo rápido do estado atual da memória."""
        if not self.snapshots:
            return {"error": "Nenhum snapshot disponível"}
        
        current = self.snapshots[-1]
        
        summary_report = {
            "timestamp": current.timestamp.isoformat(),
            "process_memory_mb": current.process_memory_mb,
            "system_memory_percent": current.system_memory_percent,
            "system_memory_available_gb": current.system_memory_available_gb,
            "total_objects": current.gc_stats["total_objects"],
            "active_threads": current.thread_count,
            "file_descriptors": current.file_descriptors,
            "peak_memory_mb": current.peak_memory_mb,
            "tracemalloc_enabled": tracemalloc.is_tracing(),
            "memory_profiler_available": MEMORY_PROFILER_AVAILABLE,
            "objgraph_available": OBJGRAPH_AVAILABLE,
            "pympler_available": PYMPLER_AVAILABLE
        }
        
        return summary_report

    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na análise de memória."""
        recommendations = []
        
        # Exemplo de recomendações baseadas em tendências de memória
        memory_trend = self._analyze_memory_trend()
        if memory_trend["trend_direction"] == "increasing":
            recommendations.append(
                "Atenção: O uso de memória está aumentando. Considere investigar objetos não coletados ou vazamentos de memória."
            )
        elif memory_trend["trend_direction"] == "decreasing":
            recommendations.append(
                "O uso de memória está diminuindo. Isso é bom, mas continue monitorando."
            )
        
        # Recomendações adicionais podem ser adicionadas aqui
        
        return recommendations

    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """Analisa tendência de uso de memória."""
        if len(self.snapshots) < 3:
            return {"error": "Insuficientes snapshots para análise de tendência"}
        
        memory_values = [s.process_memory_mb for s in self.snapshots]
        
        # Calcular tendência linear simples
        n = len(memory_values)
        x_values = list(range(n))
        
        # Coeficiente de correlação simples
        mean_x = sum(x_values) / n
        mean_y = sum(memory_values) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, memory_values))
        denominator_x = sum((x - mean_x) ** 2 for x in x_values)
        
        if denominator_x == 0:
            slope = 0
        else:
            slope = numerator / denominator_x
        
        trend_direction = "stable"
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        
        return {
            "trend_direction": trend_direction,
            "slope_mb_per_snapshot": slope,
            "min_memory_mb": min(memory_values),
            "max_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "memory_volatility": max(memory_values) - min(memory_values)
        }
    
    def _analyze_leaks(self, time_window_hours: int = 24) -> List[MemoryLeak]:
        """Analisa vazamentos de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_leaks.")
            return []

        leaks = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar vazamentos detectados
            # Esta é uma query de exemplo, ajuste conforme sua lógica de detecção
            query = """
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_increase,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       MAX(severity) as max_severity -- Pega a maior severidade
                FROM leaks
                WHERE datetime(last_seen) >= datetime('now', ?)
                GROUP BY object_type
                HAVING total_increase > 100 -- Exemplo de threshold
                ORDER BY total_size_increase DESC
            """
            
            # Calcula o tempo para a janela de análise
            time_threshold = f"-{time_window_hours} hours"
            
            cursor.execute(query, (time_threshold,))
            rows = cursor.fetchall()
            
            for row in rows:
                leak = MemoryLeak(
                    object_type=row[0],
                    count_increase=row[1],
                    size_increase_mb=row[2],
                    first_seen=datetime.fromisoformat(row[3]),
                    last_seen=datetime.fromisoformat(row[4]),
                    severity=row[5]
                )
                leaks.append(leak)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar vazamentos do banco: {e}")
        
        return leaks
    
    def _analyze_hotspots(self, top_n: int = 10) -> List[MemoryHotspot]:
        """Analisa hotspots de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_hotspots.")
            return []

        hotspots = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar hotspots
            # Esta é uma query de exemplo, ajuste conforme sua lógica
            query = """
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size, 
                       SUM(count) as total_count,
                       -- GROUP_CONCAT pode precisar de ajustes dependendo do dialeto SQL e do conteúdo
                       -- Para SQLite, GROUP_CONCAT(DISTINCT traceback_json) é uma abordagem.
                       -- Se traceback_json armazena uma lista JSON, pode ser necessário processamento posterior.
                       GROUP_CONCAT(traceback_json) as tracebacks_str 
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size DESC
                LIMIT ?
            """
            
            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()
            
            for row in rows:
                traceback_list = []
                if row[5]: # tracebacks_str
                    try:
                        # Tenta decodificar a string concatenada de JSONs
                        # Isso assume que cada traceback_json é uma string JSON válida de uma lista
                        # e que GROUP_CONCAT as une com vírgula.
                        # Se for uma única string JSON contendo todas as listas, ajuste o parse.
                        concatenated_tracebacks = row[5]
                        # Heurística para tentar separar JSONs concatenados se não for uma lista JSON válida
                        if concatenated_tracebacks.startswith('[') and concatenated_tracebacks.endswith(']') and concatenated_tracebacks.count('[') == 1:
                            traceback_list = json.loads(concatenated_tracebacks)
                        else:
                            # Tenta dividir e parsear individualmente se for algo como '[] []' ou '[][]'
                            # Isso é uma simplificação; uma solução robusta pode ser complexa
                            possible_jsons = concatenated_tracebacks.replace('][', ']#DELIM#[').split('#DELIM#')
                            for pj in possible_jsons:
                                traceback_list.extend(json.loads(pj))
                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not decode traceback JSON string: {row[5][:100]}... Error: {je}")
                        # Adiciona a string bruta se não puder ser parseada
                        traceback_list.append(str(row[5])) 

                hotspot = MemoryHotspot(
                    filename=row[0],
                    line_number=row[1],
                    function_name=row[2],
                    size_mb=row[3],
                    count=row[4],
                    traceback=traceback_list[:10] # Limita o traceback para exibição
                )
                hotspots.append(hotspot)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar hotspots do banco: {e}")
            
        return hotspots

    def cleanup(self):
        """Clean up resources, like closing the database connection and stopping tracemalloc."""
        logger.info(f"Iniciando cleanup do AdvancedMemoryProfiler ({id(self)})...")
        
        # Stop the profiling thread if it's running
        if self.is_running:
            self.is_running = False # Signal the loop to stop
            if self.profiling_thread and self.profiling_thread.is_alive():
                logger.info("Aguardando a thread de profiling terminar...")
                try:
                    self.profiling_thread.join(timeout=5.0) # Wait for 5 seconds
                    if self.profiling_thread.is_alive():
                        logger.warning("Thread de profiling não terminou a tempo.")
                    else:
                        logger.info("Thread de profiling terminada.")
                except Exception as e:
                    logger.error(f"Erro ao aguardar a thread de profiling: {e}")
            self.profiling_thread = None

        # Close the database connection
        if self.conn:
            logger.info(f"Fechando conexão com o banco de dados: {self.db_path}")
            try:
                self.conn.close()
                logger.info("Conexão com o banco de dados fechada com sucesso.")
            except sqlite3.Error as e:
                logger.error(f"Erro ao fechar conexão com o banco de dados: {e}")
            finally:
                self.conn = None # Ensure conn is None even if close fails
        else:
            logger.info("Nenhuma conexão com o banco de dados para fechar em cleanup.")

        # Stop tracemalloc
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("Tracemalloc parado via cleanup.")
        
        logger.info(f"Cleanup do AdvancedMemoryProfiler ({id(self)}) concluído.")

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        if hasattr(self, 'cleanup') and callable(self.cleanup):
            self.cleanup()

    def _profiling_loop(self):
        """Loop principal de profiling."""
        while self.is_running:
            try:
                # Capturar snapshot
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self.snapshots.append(snapshot)
                
                # Salvar no banco
                self._save_snapshot(snapshot)
                
                # Detectar vazamentos
                if len(self.snapshots) >= 5:  # Precisamos de histórico
                    leaks = self._detect_memory_leaks()
                    for leak in leaks:
                        self._save_leak(leak)
                
                # Detectar hotspots
                hotspots = self._detect_hotspots()
                for hotspot in hotspots:
                    self._save_hotspot(hotspot)
                
                # Log status periodicamente
                if len(self.snapshots) % 10 == 0:
                    self._log_profiling_status()
                
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de profiling: {e}")
                time.sleep(60)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Captura snapshot detalhado de memória."""
        timestamp = datetime.now()
        
        # Informações do processo
        process = psutil.Process()
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / 1024 / 1024
        
        # Informações do sistema
        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        system_memory_available_gb = system_memory.available / 1024 / 1024 / 1024
        
        # Estatísticas do GC
        gc_stats = {
            f"generation_{i}": len(gc.get_objects(i)) 
            for i in range(3)
        }
        gc_stats["total_objects"] = len(gc.get_objects())
        gc_stats["collected"] = gc.collect()
        
        # Tracemalloc top allocations
        tracemalloc_top = []
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_top.append({
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024
            })
            
            # Top 10 alocações
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                tracemalloc_top.append({
                    "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
        
        # Contagem de objetos por tipo
        object_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
        top_objects = dict(object_counts.most_common(20))
        
        # Informações do sistema
        thread_count = threading.active_count()
        
        # File descriptors (Unix only)
        file_descriptors = 0
        try:
            if hasattr(process, "num_fds"):
                file_descriptors = process.num_fds()
        except:
            pass
          # Stack size
        stack_size_kb = 0
        try:
            if RESOURCE_AVAILABLE:
                stack_size_kb = resource.getrlimit(resource.RLIMIT_STACK)[0] / 1024
        except:
            pass
        
        # Peak memory
        peak_memory_mb = process_memory_mb
        if hasattr(process_memory, 'peak_wset'):
            peak_memory_mb = process_memory.peak_wset / 1024 / 1024
        
        return MemorySnapshot(
            timestamp=timestamp,
            process_memory_mb=process_memory_mb,
            system_memory_percent=system_memory_percent,
            system_memory_available_gb=system_memory_available_gb,
            gc_stats=gc_stats,
            tracemalloc_top=tracemalloc_top,
            object_counts=top_objects,
            thread_count=thread_count,
            file_descriptors=file_descriptors,
            stack_size_kb=stack_size_kb,
            peak_memory_mb=peak_memory_mb
        )
    
    def _detect_memory_leaks(self) -> List[MemoryLeak]:
        """Detecta vazamentos de memória comparando snapshots."""
        if len(self.snapshots) < 5:
            return []
        
        leaks = []
        recent_snapshots = list(self.snapshots)[-5:]  # Últimos 5 snapshots
        
        # Comparar contagens de objetos
        baseline_objects = recent_snapshots[0].object_counts
        current_objects = recent_snapshots[-1].object_counts
        
        for obj_type, current_count in current_objects.items():
            baseline_count = baseline_objects.get(obj_type, 0)
            count_increase = current_count - baseline_count
            
            # Detectar aumento significativo
            if count_increase > 100 and count_increase > baseline_count * 0.5:
                # Estimar tamanho (aproximação)
                size_increase_mb = count_increase * 0.001  # Estimativa conservadora
                
                severity = "low"
                if count_increase > 1000:
                    severity = "high"
                elif count_increase > 500:
                    severity = "medium"
                
                leak = MemoryLeak(
                    object_type=obj_type,
                    count_increase=count_increase,
                    size_increase_mb=size_increase_mb,
                    first_seen=recent_snapshots[0].timestamp,
                    last_seen=recent_snapshots[-1].timestamp,
                    severity=severity
                )
                leaks.append(leak)
        
        return leaks
    
    def _detect_hotspots(self) -> List[MemoryHotspot]:
        """Detecta hotspots de uso de memória."""
        hotspots = []
        
        if not tracemalloc.is_tracing():
            return hotspots
        
        try:
            snapshot = tracemalloc.take_snapshot()
            # Filtrar por tamanho e agrupar por traceback para identificar hotspots reais
            # Aumentar o limite para top_stats para ter uma visão mais ampla
            top_stats = snapshot.statistics('traceback')[:50] 
            
            processed_hotspots = {}

            for stat in top_stats:
                if stat.size < 1024 * 1024:  # Ignorar < 1MB
                    continue
                
                # Usar o traceback formatado como chave para agrupar alocações similares
                traceback_key_list = stat.traceback.format()
                traceback_key = "\\n".join(traceback_key_list) # Chave mais robusta

                if traceback_key not in processed_hotspots:
                    frame = stat.traceback[-1] # Frame mais específico
                    filename = frame.filename
                    line_number = frame.lineno
                    # Tentar extrair o nome da função do traceback, se possível
                    # Isso pode ser complexo, então uma abordagem simples é usada aqui
                    function_name = "unknown" 
                    try:
                        # Tenta pegar o nome da função da última linha do traceback formatado
                        last_trace_line = traceback_key_list[-1].strip()
                        if " in " in last_trace_line:
                            function_name = last_trace_line.split(" in ")[-1]
                            if function_name.startswith("<"): # Ex: <module>
                                function_name = "unknown" 
                    except Exception:
                        pass # Mantém unknown se a extração falhar

                    processed_hotspots[traceback_key] = MemoryHotspot(
                        filename=filename,
                        line_number=line_number,
                        function_name=function_name, 
                        size_mb=0, # Será agregado
                        count=0,   # Será agregado
                        traceback=traceback_key_list 
                    )
                
                processed_hotspots[traceback_key].size_mb += stat.size / 1024 / 1024
                processed_hotspots[traceback_key].count += stat.count
            
            # Converter para lista e ordenar por tamanho
            hotspots = sorted(list(processed_hotspots.values()), key=lambda h: h.size_mb, reverse=True)[:20]

        except Exception as e:
            logger.error(f"Erro ao detectar hotspots: {e}")
        
        return hotspots
    
    def _save_snapshot(self, snapshot: MemorySnapshot):
        """Save a memory snapshot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save snapshot.")
            return

        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'snapshots'
            cursor.execute("""
                INSERT INTO snapshots (
                    timestamp, process_memory_mb, system_memory_percent,
                    system_memory_available_gb, gc_stats_json, tracemalloc_top_json,
                    object_counts_json, thread_count, file_descriptors,
                    stack_size_kb, peak_memory_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                snapshot.process_memory_mb,
                snapshot.system_memory_percent,
                snapshot.system_memory_available_gb,
                json.dumps(snapshot.gc_stats),
                json.dumps(snapshot.tracemalloc_top),
                json.dumps(snapshot.object_counts),
                snapshot.thread_count,
                snapshot.file_descriptors,
                snapshot.stack_size_kb,
                snapshot.peak_memory_mb
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def _save_leak(self, leak: MemoryLeak):
        """Save a detected memory leak to the database."""
        if not self.conn:
            logger.warning("No database connection available to save leak.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks' e campo detection_timestamp
            cursor.execute("""
                INSERT INTO leaks (
                    object_type, count_increase, size_increase_mb,
                    first_seen, last_seen, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                leak.object_type,
                leak.count_increase,
                leak.size_increase_mb,
                leak.first_seen.isoformat(),
                leak.last_seen.isoformat(),
                leak.severity
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving leak: {e}")
    
    def _save_hotspot(self, hotspot: MemoryHotspot):
        """Save a detected memory hotspot to the database."""
        if not self.conn:
            logger.warning("No database connection available to save hotspot.")
            return
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute("""
                INSERT INTO hotspots (
                    filename, line_number, function_name,
                    size_mb, count, traceback_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                hotspot.filename,
                hotspot.line_number,
                hotspot.function_name,
                hotspot.size_mb,
                hotspot.count,
                json.dumps(hotspot.traceback) # Salvar traceback como JSON
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving hotspot: {e}")

    def _analyze_leaks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored leak data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for leak analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'leaks'
            cursor.execute(f"""
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_mb,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       GROUP_CONCAT(severity) as severities
                FROM leaks
                GROUP BY object_type
                ORDER BY total_size_mb DESC, total_increase DESC
                LIMIT {limit}
            """)
            leaks_data = []
            for row in cursor.fetchall():
                leaks_data.append({
                    "object_type": row[0],
                    "total_increase": row[1],
                    "total_size_mb": row[2],
                    "first_occurrence": row[3],
                    "last_occurrence": row[4],
                    "severities": list(set(row[5].split(','))) if row[5] else []
                })
            return leaks_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing leaks: {e}")
            return []

    def _analyze_hotspots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze stored hotspot data from the database."""
        if not self.conn:
            logger.warning("Database connection not available for hotspot analysis.")
            return []
        try:
            cursor = self.conn.cursor()
            # Corrigido o nome da tabela para 'hotspots' e campo traceback_json
            cursor.execute(f"""
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size_mb, 
                       SUM(count) as total_count,
                       GROUP_CONCAT(traceback_json) as tracebacks_json_array
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size_mb DESC, total_count DESC
                LIMIT {limit}
            """)
            hotspots_data = []
            for row in cursor.fetchall():
                # Processar os tracebacks concatenados
                # Cada traceback_json é uma string JSON de uma lista.
                # GROUP_CONCAT junta essas strings com vírgula.
                # Precisamos parsear cada um individualmente se quisermos a lista de listas.
                # Ou, mais simples, pegar o primeiro ou o mais comum se for o caso.
                # Aqui, vamos apenas tentar carregar o primeiro se houver muitos.
                raw_tracebacks = row[5]
                tracebacks = []
                if raw_tracebacks:
                    # Tentativa de parsear o JSON concatenado.
                    # Isso pode ser problemático se os JSONs não estiverem bem formados
                    # ou se a concatenação criar um JSON inválido.
                    # Uma abordagem mais segura seria armazenar tracebacks de forma diferente
                    # ou processá-los antes do GROUP_CONCAT.
                    # Para este exemplo, vamos assumir que cada traceback é uma string JSON válida
                    # e tentar parsear a primeira.
                    try:
                        # Isso provavelmente não funcionará como esperado com GROUP_CONCAT
                        # de múltiplas listas JSON.
                        # tracebacks = json.loads(f'[{raw_tracebacks}]')
                        # Uma abordagem mais simples: pegar a primeira string JSON e parseá-la
                        first_traceback_json = raw_tracebacks.split('],[')[0] + ']' if '],[' in raw_tracebacks else raw_tracebacks
                        if not first_traceback_json.startswith('['):
                            first_traceback_json = '[' + first_traceback_json
                        if not first_traceback_json.endswith(']'):
                             # Pode já ter sido cortado, ou ser um único JSON
                            if not first_traceback_json.endswith('"}'): # fim de um objeto json
                                first_traceback_json = first_traceback_json + ']'
                        

                        # Sanitize common issues with concatenated JSON strings
                        # This is a common issue with GROUP_CONCAT of JSON strings.
                        # Example: "[...], [...]" needs to become "[[...], [...]]"
                        # Or, if they are distinct JSON objects: "{...}, {...}" needs to become "[{...}, {...}]"
                        # For now, we'll just try to parse the first one if it's a list of strings.
                        
                        # Simplificando: apenas pegamos a string bruta por enquanto
                        # A lógica de reconstrução de JSON a partir de GROUP_CONCAT é complexa
                        # e depende de como os dados foram inseridos.
                        # Se cada `traceback_json` era `json.dumps(["line1", "line2"])`
                        # então `GROUP_CONCAT` produz algo como:
                        # '["lineA1", "lineA2"],["lineB1", "lineB2"]'
                        # Para transformar isso em um JSON válido de lista de listas:
                        # `json.loads('[' + raw_tracebacks + ']')`
                        
                        # Se o traceback_json já é uma string representando uma lista de strings
                        # e o GROUP_CONCAT as une, precisamos de uma estratégia.
                        # Assumindo que cada traceback_json é uma lista de strings serializada:
                        # e.g., '["file:1", "file:2"]'
                        # GROUP_CONCAT resultaria em: '["file:1", "file:2"],["file:3", "file:4"]'
                        # Para parsear isso como uma lista de listas de strings:
                        parsed_tracebacks = []
                        current_json_str = ""
                        depth = 0
                        for char in raw_tracebacks:
                            current_json_str += char
                            if char == '[':
                                depth += 1
                            elif char == ']':
                                depth -= 1
                                if depth == 0 and current_json_str:
                                    try:
                                        parsed_tracebacks.append(json.loads(current_json_str))
                                        current_json_str = ""
                                    except json.JSONDecodeError:
                                        # Ignorar se uma sub-string não é JSON válido, pode acontecer com GROUP_CONCAT
                                        current_json_str = "" # Reset
                                        pass 
                        if parsed_tracebacks:
                             tracebacks = parsed_tracebacks[0] # Pegar o primeiro traceback completo

                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not parse tracebacks for hotspot {row[0]}:{row[1]}: {je}. Raw: {raw_tracebacks[:100]}")
                        tracebacks = [raw_tracebacks] # fallback to raw string if parsing fails

                hotspots_data.append({
                    "filename": row[0],
                    "line_number": row[1],
                    "function_name": row[2],
                    "total_size_mb": row[3],
                    "total_count": row[4],
                    "tracebacks": tracebacks # Idealmente, uma lista de strings (o primeiro traceback)
                })
            return hotspots_data
        except sqlite3.Error as e:
            logger.error(f"Error analyzing hotspots: {e}")
            return []

    def get_summary(self) -> Dict[str, Any]:
        """Gera um resumo rápido do estado atual da memória."""
        if not self.snapshots:
            return {"error": "Nenhum snapshot disponível"}
        
        current = self.snapshots[-1]
        
        summary_report = {
            "timestamp": current.timestamp.isoformat(),
            "process_memory_mb": current.process_memory_mb,
            "system_memory_percent": current.system_memory_percent,
            "system_memory_available_gb": current.system_memory_available_gb,
            "total_objects": current.gc_stats["total_objects"],
            "active_threads": current.thread_count,
            "file_descriptors": current.file_descriptors,
            "peak_memory_mb": current.peak_memory_mb,
            "tracemalloc_enabled": tracemalloc.is_tracing(),
            "memory_profiler_available": MEMORY_PROFILER_AVAILABLE,
            "objgraph_available": OBJGRAPH_AVAILABLE,
            "pympler_available": PYMPLER_AVAILABLE
        }
        
        return summary_report

    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na análise de memória."""
        recommendations = []
        
        # Exemplo de recomendações baseadas em tendências de memória
        memory_trend = self._analyze_memory_trend()
        if memory_trend["trend_direction"] == "increasing":
            recommendations.append(
                "Atenção: O uso de memória está aumentando. Considere investigar objetos não coletados ou vazamentos de memória."
            )
        elif memory_trend["trend_direction"] == "decreasing":
            recommendations.append(
                "O uso de memória está diminuindo. Isso é bom, mas continue monitorando."
            )
        
        # Recomendações adicionais podem ser adicionadas aqui
        
        return recommendations

    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """Analisa tendência de uso de memória."""
        if len(self.snapshots) < 3:
            return {"error": "Insuficientes snapshots para análise de tendência"}
        
        memory_values = [s.process_memory_mb for s in self.snapshots]
        
        # Calcular tendência linear simples
        n = len(memory_values)
        x_values = list(range(n))
        
        # Coeficiente de correlação simples
        mean_x = sum(x_values) / n
        mean_y = sum(memory_values) / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, memory_values))
        denominator_x = sum((x - mean_x) ** 2 for x in x_values)
        
        if denominator_x == 0:
            slope = 0
        else:
            slope = numerator / denominator_x
        
        trend_direction = "stable"
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        
        return {
            "trend_direction": trend_direction,
            "slope_mb_per_snapshot": slope,
            "min_memory_mb": min(memory_values),
            "max_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "memory_volatility": max(memory_values) - min(memory_values)
        }
    
    def _analyze_leaks(self, time_window_hours: int = 24) -> List[MemoryLeak]:
        """Analisa vazamentos de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_leaks.")
            return []

        leaks = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar vazamentos detectados
            # Esta é uma query de exemplo, ajuste conforme sua lógica de detecção
            query = """
                SELECT object_type, SUM(count_increase) as total_increase, 
                       SUM(size_increase_mb) as total_size_increase,
                       MIN(first_seen) as first_occurrence,
                       MAX(last_seen) as last_occurrence,
                       MAX(severity) as max_severity -- Pega a maior severidade
                FROM leaks
                WHERE datetime(last_seen) >= datetime('now', ?)
                GROUP BY object_type
                HAVING total_increase > 100 -- Exemplo de threshold
                ORDER BY total_size_increase DESC
            """
            
            # Calcula o tempo para a janela de análise
            time_threshold = f"-{time_window_hours} hours"
            
            cursor.execute(query, (time_threshold,))
            rows = cursor.fetchall()
            
            for row in rows:
                leak = MemoryLeak(
                    object_type=row[0],
                    count_increase=row[1],
                    size_increase_mb=row[2],
                    first_seen=datetime.fromisoformat(row[3]),
                    last_seen=datetime.fromisoformat(row[4]),
                    severity=row[5]
                )
                leaks.append(leak)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar vazamentos do banco: {e}")
        
        return leaks
    
    def _analyze_hotspots(self, top_n: int = 10) -> List[MemoryHotspot]:
        """Analisa hotspots de memória do banco de dados."""
        if not self.conn:
            logger.warning("Database connection not available for _analyze_hotspots.")
            return []

        hotspots = []
        try:
            cursor = self.conn.cursor()
            
            # Query para buscar hotspots
            # Esta é uma query de exemplo, ajuste conforme sua lógica
            query = """
                SELECT filename, line_number, function_name, 
                       SUM(size_mb) as total_size, 
                       SUM(count) as total_count,
                       -- GROUP_CONCAT pode precisar de ajustes dependendo do dialeto SQL e do conteúdo
                       -- Para SQLite, GROUP_CONCAT(DISTINCT traceback_json) é uma abordagem.
                       -- Se traceback_json armazena uma lista JSON, pode ser necessário processamento posterior.
                       GROUP_CONCAT(traceback_json) as tracebacks_str 
                FROM hotspots
                GROUP BY filename, line_number, function_name
                ORDER BY total_size DESC
                LIMIT ?
            """
            
            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()
            
            for row in rows:
                traceback_list = []
                if row[5]: # tracebacks_str
                    try:
                        # Tenta decodificar a string concatenada de JSONs
                        # Isso assume que cada traceback_json é uma string JSON válida de uma lista
                        # e que GROUP_CONCAT as une com vírgula.
                        # Se for uma única string JSON contendo todas as listas, ajuste o parse.
                        concatenated_tracebacks = row[5]
                        # Heurística para tentar separar JSONs concatenados se não for uma lista JSON válida
                        if concatenated_tracebacks.startswith('[') and concatenated_tracebacks.endswith(']') and concatenated_tracebacks.count('[') == 1:
                            traceback_list = json.loads(concatenated_tracebacks)
                        else:
                            # Tenta dividir e parsear individualmente se for algo como '[] []' ou '[][]'
                            # Isso é uma simplificação; uma solução robusta pode ser complexa
                            possible_jsons = concatenated_tracebacks.replace('][', ']#DELIM#[').split('#DELIM#')
                            for pj in possible_jsons:
                                traceback_list.extend(json.loads(pj))
                    except json.JSONDecodeError as je:
                        logger.warning(f"Could not decode traceback JSON string: {row[5][:100]}... Error: {je}")
                        # Adiciona a string bruta se não puder ser parseada
                        traceback_list.append(str(row[5])) 

                hotspot = MemoryHotspot(
                    filename=row[0],
                    line_number=row[1],
                    function_name=row[2],
                    size_mb=row[3],
                    count=row[4],
                    traceback=traceback_list[:10] # Limita o traceback para exibição
                )
                hotspots.append(hotspot)
                
        except sqlite3.Error as e:
            logger.error(f"Erro ao analisar hotspots do banco: {e}")
            
        return hotspots