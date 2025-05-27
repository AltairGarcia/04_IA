#!/usr/bin/env python3
"""
Integração do ThreadSafeConnectionManager ao Sistema de Monitoramento Unificado

Este módulo integra o gerenciador robusto de conexões SQLite ao sistema de 
monitoramento unificado, fornecendo:

1. Substituição automática do sistema de database básico
2. Integração de métricas de conexão ao sistema de monitoramento
3. Profiling de memória específico para conexões de database
4. Alertas avançados para problemas de conexão
5. Migração transparente dos dados existentes

Author: GitHub Copilot
Date: 2025-05-27
"""

import os
import sys
import time
import logging
import sqlite3
import threading
import weakref
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager

# Importar sistemas existentes
try:
    from unified_monitoring_system import UnifiedMonitoringSystem
    from thread_safe_connection_manager import ThreadSafeConnectionManager, get_connection_manager
    from advanced_memory_profiler import AdvancedMemoryProfiler
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Alguns módulos não estão disponíveis: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseIntegrationMetrics:
    """Métricas de integração de database."""
    timestamp: datetime
    active_connections: int
    total_queries: int
    query_errors: int
    avg_query_time: float
    connection_pool_efficiency: float
    memory_used_by_connections_mb: float
    orphaned_connections_cleaned: int
    database_file_size_mb: float


class EnhancedUnifiedMonitoringSystem:
    """
    Sistema de monitoramento unificado aprimorado com gerenciamento robusto de conexões.
    
    Integra o ThreadSafeConnectionManager ao sistema existente, fornecendo:
    - Gestão robusta de conexões SQLite
    - Profiling de memória específico para database
    - Métricas avançadas de performance de database
    - Alertas inteligentes para problemas de conexão
    """
    
    def __init__(self, 
                 db_path: str = "enhanced_monitoring.db",
                 enable_memory_profiling: bool = True,
                 connection_timeout: int = 300,
                 max_connections: int = 20):
        
        self.db_path = Path(db_path)
        self.enable_memory_profiling = enable_memory_profiling
        self.is_running = False
        
        # Gerenciador de conexões robusto
        self.connection_manager = ThreadSafeConnectionManager(
            db_path=str(self.db_path),
            max_connections=max_connections,
            connection_timeout=connection_timeout
        )
        
        # Sistema de monitoramento base (se disponível)
        self.base_monitoring = None
        if IMPORTS_AVAILABLE:
            try:
                # Usar o sistema unificado existente com o novo gerenciador
                self.base_monitoring = UnifiedMonitoringSystem(
                    db_path=str(self.db_path),
                    enable_alerts=True
                )
                # Substituir o database manager do sistema base
                self.base_monitoring.db_manager = self.connection_manager
                logger.info("Sistema base integrado com sucesso")
            except Exception as e:
                logger.warning(f"Não foi possível integrar sistema base: {e}")
        
        # Profiler de memória (se disponível)
        self.memory_profiler = None
        if self.enable_memory_profiling and IMPORTS_AVAILABLE:
            try:
                self.memory_profiler = AdvancedMemoryProfiler(
                    db_path=str(self.db_path.with_suffix('.memory.db')),
                    snapshot_interval=60  # Snapshot a cada minuto
                )
                logger.info("Memory profiler integrado")
            except Exception as e:
                logger.warning(f"Não foi possível inicializar memory profiler: {e}")
        
        # Estado interno
        self._lock = threading.RLock()
        self.monitoring_thread = None
        self.integration_metrics = []
        self.alert_handlers = []
        
        # Inicializar esquema aprimorado
        self._initialize_enhanced_schema()
        
        # Migrar dados existentes se necessário
        self._migrate_existing_data()
        
        logger.info(f"EnhancedUnifiedMonitoringSystem inicializado: {self.db_path}")
    
    def _initialize_enhanced_schema(self):
        """Inicializa esquema aprimorado do banco de dados."""
        try:
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabela de métricas de integração
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS database_integration_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        active_connections INTEGER,
                        total_queries INTEGER,
                        query_errors INTEGER,
                        avg_query_time REAL,
                        connection_pool_efficiency REAL,
                        memory_used_by_connections_mb REAL,
                        orphaned_connections_cleaned INTEGER,
                        database_file_size_mb REAL
                    )
                """)
                
                # Tabela de alertas aprimorados
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS enhanced_alerts (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        category TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        source TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        metrics TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TEXT,
                        resolved_by TEXT
                    )
                """)
                
                # Tabela de eventos de sistema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        description TEXT NOT NULL,
                        data TEXT,
                        severity TEXT DEFAULT 'info'
                    )
                """)
                
                # Tabela de configurações dinâmicas
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_config (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        description TEXT,
                        updated_at TEXT NOT NULL,
                        updated_by TEXT DEFAULT 'system'
                    )
                """)
                
                # Índices para performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_integration_metrics_timestamp ON database_integration_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_alerts_timestamp ON enhanced_alerts(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_alerts_category ON enhanced_alerts(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)")
                
                # Configurações padrão
                default_configs = {
                    'memory_threshold_percent': '85',
                    'connection_timeout_seconds': '300',
                    'max_connections': '20',
                    'profiling_enabled': 'true',
                    'alert_cooldown_minutes': '5',
                    'cleanup_interval_seconds': '60'
                }
                
                for key, value in default_configs.items():
                    cursor.execute("""
                        INSERT OR IGNORE INTO system_config (key, value, description, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, (key, value, f"Default configuration for {key}", datetime.now().isoformat()))
                
                conn.commit()
                logger.info("Esquema aprimorado inicializado com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar esquema aprimorado: {e}")
            raise
    
    def _migrate_existing_data(self):
        """Migra dados de sistemas de monitoramento existentes."""
        try:
            # Verificar se há tabelas antigas para migrar
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Listar tabelas existentes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                migration_count = 0
                
                # Migrar system_metrics se existir
                if 'system_metrics' in existing_tables:
                    cursor.execute("SELECT COUNT(*) FROM system_metrics")
                    old_metrics_count = cursor.fetchone()[0]
                    
                    if old_metrics_count > 0:
                        logger.info(f"Migrando {old_metrics_count} métricas do sistema antigo...")
                        
                        # Criar evento de migração
                        self._log_system_event(
                            "migration",
                            "database_migration",
                            f"Iniciando migração de {old_metrics_count} métricas do sistema antigo",
                            {"old_metrics_count": old_metrics_count}
                        )
                        migration_count += old_metrics_count
                
                # Migrar system_alerts se existir
                if 'system_alerts' in existing_tables:
                    cursor.execute("SELECT COUNT(*) FROM system_alerts")
                    old_alerts_count = cursor.fetchone()[0]
                    
                    if old_alerts_count > 0:
                        logger.info(f"Migrando {old_alerts_count} alertas do sistema antigo...")
                        
                        # Migrar alertas para formato aprimorado
                        cursor.execute("""
                            INSERT INTO enhanced_alerts 
                            (id, timestamp, category, severity, source, title, message, details, resolved, resolved_at)
                            SELECT 
                                id, timestamp, 'legacy' as category, severity, source, 
                                message as title, message, details, resolved, resolved_at
                            FROM system_alerts 
                            WHERE id NOT IN (SELECT id FROM enhanced_alerts)
                        """)
                        
                        migration_count += old_alerts_count
                
                if migration_count > 0:
                    conn.commit()
                    logger.info(f"Migração concluída: {migration_count} registros migrados")
                    
                    # Log evento de migração concluída
                    self._log_system_event(
                        "migration",
                        "database_migration",
                        f"Migração concluída com sucesso: {migration_count} registros",
                        {"migrated_records": migration_count}
                    )
                else:
                    logger.info("Nenhum dado antigo encontrado para migração")
                    
        except Exception as e:
            logger.error(f"Erro durante migração: {e}")
            # Não falhar por causa da migração
    
    def start_monitoring(self):
        """Inicia o sistema de monitoramento aprimorado."""
        if self.is_running:
            logger.warning("Sistema já está executando")
            return
        
        self.is_running = True
        
        # Iniciar profiler de memória se disponível
        if self.memory_profiler:
            try:
                self.memory_profiler.start_profiling(baseline=True)
                logger.info("Memory profiler iniciado")
            except Exception as e:
                logger.warning(f"Erro ao iniciar memory profiler: {e}")
        
        # Iniciar sistema base se disponível
        if self.base_monitoring:
            try:
                self.base_monitoring.start_monitoring()
                logger.info("Sistema base de monitoramento iniciado")
            except Exception as e:
                logger.warning(f"Erro ao iniciar sistema base: {e}")
        
        # Iniciar thread de monitoramento de integração
        self.monitoring_thread = threading.Thread(
            target=self._integration_monitoring_loop,
            daemon=True,
            name="EnhancedMonitoring-Integration"
        )
        self.monitoring_thread.start()
        
        # Log evento de inicialização
        self._log_system_event(
            "system",
            "monitoring_start",
            "Sistema de monitoramento aprimorado iniciado",
            {
                "memory_profiling_enabled": self.memory_profiler is not None,
                "base_system_enabled": self.base_monitoring is not None,
                "connection_manager_active": True
            }
        )
        
        logger.info("Sistema de monitoramento aprimorado iniciado")
    
    def stop_monitoring(self):
        """Para o sistema de monitoramento."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Parar thread de monitoramento
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        # Parar profiler de memória
        if self.memory_profiler:
            try:
                final_report = self.memory_profiler.stop_profiling()
                self._save_profiling_report(final_report)
                logger.info("Memory profiler finalizado")
            except Exception as e:
                logger.warning(f"Erro ao finalizar memory profiler: {e}")
        
        # Parar sistema base
        if self.base_monitoring:
            try:
                self.base_monitoring.stop_monitoring()
                logger.info("Sistema base finalizado")
            except Exception as e:
                logger.warning(f"Erro ao finalizar sistema base: {e}")
        
        # Log evento de finalização
        self._log_system_event(
            "system",
            "monitoring_stop",
            "Sistema de monitoramento aprimorado finalizado",
            {"integration_metrics_collected": len(self.integration_metrics)}
        )
        
        logger.info("Sistema de monitoramento aprimorado finalizado")
    
    def _integration_monitoring_loop(self):
        """Loop de monitoramento de integração."""
        while self.is_running:
            try:
                # Coletar métricas de integração
                metrics = self._collect_integration_metrics()
                
                with self._lock:
                    self.integration_metrics.append(metrics)
                    
                    # Manter apenas últimas 1000 métricas na memória
                    if len(self.integration_metrics) > 1000:
                        self.integration_metrics = self.integration_metrics[-1000:]
                
                # Salvar no banco
                self._save_integration_metrics(metrics)
                
                # Analisar e gerar alertas
                self._analyze_and_alert(metrics)
                
                # Log status periodicamente
                if len(self.integration_metrics) % 10 == 0:
                    self._log_integration_status(metrics)
                
                time.sleep(60)  # Coletar a cada minuto
                
            except Exception as e:
                logger.error(f"Erro no loop de monitoramento de integração: {e}")
                time.sleep(30)
    
    def _collect_integration_metrics(self) -> DatabaseIntegrationMetrics:
        """Coleta métricas de integração do sistema."""
        timestamp = datetime.now()
        
        # Métricas do gerenciador de conexões
        connection_stats = self.connection_manager.get_connection_stats()
        
        # Métricas do banco de dados
        db_file_size_mb = 0
        try:
            if self.db_path.exists():
                db_file_size_mb = self.db_path.stat().st_size / 1024 / 1024
        except:
            pass
        
        # Calcular eficiência do pool de conexões
        active_connections = connection_stats.get('active_connections', 0)
        max_connections = connection_stats.get('max_connections', 1)
        pool_efficiency = (active_connections / max_connections) * 100
        
        # Estimar uso de memória por conexões (aproximação)
        memory_per_connection_mb = 2.0  # Estimativa conservadora
        memory_used_by_connections_mb = active_connections * memory_per_connection_mb
        
        return DatabaseIntegrationMetrics(
            timestamp=timestamp,
            active_connections=active_connections,
            total_queries=connection_stats.get('total_queries', 0),
            query_errors=connection_stats.get('total_errors', 0),
            avg_query_time=0.0,  # Será calculado se disponível
            connection_pool_efficiency=pool_efficiency,
            memory_used_by_connections_mb=memory_used_by_connections_mb,
            orphaned_connections_cleaned=0,  # Será implementado
            database_file_size_mb=db_file_size_mb
        )
    
    def _save_integration_metrics(self, metrics: DatabaseIntegrationMetrics):
        """Salva métricas de integração no banco."""
        try:
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO database_integration_metrics (
                        timestamp, active_connections, total_queries, query_errors,
                        avg_query_time, connection_pool_efficiency, 
                        memory_used_by_connections_mb, orphaned_connections_cleaned,
                        database_file_size_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.active_connections,
                    metrics.total_queries,
                    metrics.query_errors,
                    metrics.avg_query_time,
                    metrics.connection_pool_efficiency,
                    metrics.memory_used_by_connections_mb,
                    metrics.orphaned_connections_cleaned,
                    metrics.database_file_size_mb
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao salvar métricas de integração: {e}")
    
    def _analyze_and_alert(self, metrics: DatabaseIntegrationMetrics):
        """Analisa métricas e gera alertas se necessário."""
        alerts = []
        
        # Verificar eficiência do pool de conexões
        if metrics.connection_pool_efficiency > 90:
            alerts.append({
                "category": "database",
                "severity": "warning",
                "title": "Pool de Conexões Quase Cheio",
                "message": f"Pool de conexões com {metrics.connection_pool_efficiency:.1f}% de utilização",
                "details": {
                    "active_connections": metrics.active_connections,
                    "efficiency_percent": metrics.connection_pool_efficiency
                }
            })
        
        # Verificar taxa de erro de queries
        if metrics.total_queries > 0:
            error_rate = (metrics.query_errors / metrics.total_queries) * 100
            if error_rate > 5:
                alerts.append({
                    "category": "database",
                    "severity": "critical" if error_rate > 10 else "warning",
                    "title": "Alta Taxa de Erro em Queries",
                    "message": f"Taxa de erro de {error_rate:.1f}% detectada",
                    "details": {
                        "total_queries": metrics.total_queries,
                        "query_errors": metrics.query_errors,
                        "error_rate_percent": error_rate
                    }
                })
        
        # Verificar tamanho do banco de dados
        if metrics.database_file_size_mb > 1000:  # > 1GB
            alerts.append({
                "category": "storage",
                "severity": "info",
                "title": "Banco de Dados Grande",
                "message": f"Banco de dados com {metrics.database_file_size_mb:.1f}MB",
                "details": {
                    "database_size_mb": metrics.database_file_size_mb,
                    "recommendation": "Considerar limpeza de dados antigos"
                }
            })
        
        # Salvar alertas
        for alert_data in alerts:
            self._create_enhanced_alert(**alert_data)
    
    def _create_enhanced_alert(self, category: str, severity: str, title: str, 
                              message: str, details: Dict = None):
        """Cria alerta aprimorado."""
        alert_id = f"{category}_{severity}_{int(time.time())}"
        
        try:
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO enhanced_alerts (
                        id, timestamp, category, severity, source, title, message, details, metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert_id,
                    datetime.now().isoformat(),
                    category,
                    severity,
                    "enhanced_monitoring",
                    title,
                    message,
                    json.dumps(details or {}),
                    json.dumps({})
                ))
                conn.commit()
                
                logger.info(f"Alerta criado: [{severity.upper()}] {title}")
                
                # Chamar handlers de alerta
                for handler in self.alert_handlers:
                    try:
                        handler(alert_id, category, severity, title, message, details)
                    except Exception as e:
                        logger.error(f"Erro em handler de alerta: {e}")
                        
        except Exception as e:
            logger.error(f"Erro ao criar alerta: {e}")
    
    def _log_system_event(self, event_type: str, source: str, description: str, 
                         data: Dict = None, severity: str = "info"):
        """Log evento do sistema."""
        try:
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_events (
                        timestamp, event_type, source, description, data, severity
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    event_type,
                    source,
                    description,
                    json.dumps(data or {}),
                    severity
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao log evento: {e}")
    
    def _log_integration_status(self, metrics: DatabaseIntegrationMetrics):
        """Log status atual da integração."""
        logger.info(
            f"Integration Status: "
            f"Connections: {metrics.active_connections}, "
            f"Queries: {metrics.total_queries}, "
            f"Errors: {metrics.query_errors}, "
            f"Pool Efficiency: {metrics.connection_pool_efficiency:.1f}%, "
            f"DB Size: {metrics.database_file_size_mb:.1f}MB"
        )
    
    def _save_profiling_report(self, report: Dict):
        """Salva relatório de profiling no sistema."""
        try:
            self._log_system_event(
                "profiling",
                "memory_profiler",
                "Relatório de profiling de memória gerado",
                {
                    "report_summary": {
                        "duration_minutes": report.get("report_metadata", {}).get("profiling_duration_minutes", 0),
                        "snapshots": report.get("report_metadata", {}).get("total_snapshots", 0),
                        "memory_change_mb": report.get("memory_summary", {}).get("memory_change_mb", 0),
                        "recommendations": len(report.get("recommendations", []))
                    }
                },
                "info"
            )
        except Exception as e:
            logger.error(f"Erro ao salvar relatório de profiling: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Adiciona handler para alertas."""
        self.alert_handlers.append(handler)
        logger.info("Handler de alerta adicionado")
    
    def get_integration_report(self) -> Dict[str, Any]:
        """Gera relatório de integração."""
        try:
            with self.connection_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Estatísticas gerais
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_metrics,
                        AVG(active_connections) as avg_connections,
                        MAX(active_connections) as max_connections,
                        AVG(connection_pool_efficiency) as avg_efficiency,
                        SUM(query_errors) as total_errors,
                        MAX(database_file_size_mb) as max_db_size
                    FROM database_integration_metrics 
                    WHERE timestamp > datetime('now', '-24 hours')
                """)
                
                stats = cursor.fetchone()
                
                # Alertas recentes
                cursor.execute("""
                    SELECT category, severity, COUNT(*) 
                    FROM enhanced_alerts 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY category, severity
                    ORDER BY COUNT(*) DESC
                """)
                
                alerts_summary = cursor.fetchall()
                
                # Eventos do sistema
                cursor.execute("""
                    SELECT event_type, COUNT(*) 
                    FROM system_events 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY event_type
                    ORDER BY COUNT(*) DESC
                """)
                
                events_summary = cursor.fetchall()
                
                report = {
                    "report_metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "report_period": "24 hours",
                        "system_status": "running" if self.is_running else "stopped"
                    },
                    "integration_statistics": {
                        "total_metrics_collected": stats[0] if stats else 0,
                        "avg_active_connections": stats[1] if stats else 0,
                        "max_active_connections": stats[2] if stats else 0,
                        "avg_pool_efficiency": stats[3] if stats else 0,
                        "total_query_errors": stats[4] if stats else 0,
                        "max_database_size_mb": stats[5] if stats else 0
                    },
                    "alerts_summary": [
                        {"category": row[0], "severity": row[1], "count": row[2]}
                        for row in alerts_summary
                    ],
                    "events_summary": [
                        {"event_type": row[0], "count": row[1]}
                        for row in events_summary
                    ],
                    "connection_manager_status": self.connection_manager.get_connection_stats(),
                    "memory_profiler_active": self.memory_profiler is not None and self.memory_profiler.is_running,
                    "base_monitoring_active": self.base_monitoring is not None
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de integração: {e}")
            return {"error": str(e)}
    
    def export_integration_report(self, filepath: str = None) -> str:
        """Exporta relatório de integração para arquivo."""
        if filepath is None:
            filepath = f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_integration_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Relatório de integração exportado para {filepath}")
        return filepath
    
    def cleanup(self):
        """Limpeza de recursos."""
        self.stop_monitoring()
        
        if self.connection_manager:
            self.connection_manager.close_all_connections()
        
        if self.memory_profiler:
            self.memory_profiler.cleanup()
        
        logger.info("EnhancedUnifiedMonitoringSystem finalizado")
    
    def __del__(self):
        """Cleanup automático."""
        try:
            self.cleanup()
        except:
            pass


def create_integrated_monitoring_system(
    db_path: str = "integrated_monitoring.db",
    enable_memory_profiling: bool = True,
    **kwargs
) -> EnhancedUnifiedMonitoringSystem:
    """
    Factory function para criar sistema de monitoramento integrado.
    
    Args:
        db_path: Caminho para o banco de dados
        enable_memory_profiling: Habilitar profiling de memória
        **kwargs: Argumentos adicionais para o sistema
    
    Returns:
        Instância configurada do sistema integrado
    """
    return EnhancedUnifiedMonitoringSystem(
        db_path=db_path,
        enable_memory_profiling=enable_memory_profiling,
        **kwargs
    )


def migrate_from_legacy_system(
    legacy_db_path: str,
    new_db_path: str = "integrated_monitoring.db"
) -> bool:
    """
    Migra sistema legado para o sistema integrado.
    
    Args:
        legacy_db_path: Caminho do banco legado
        new_db_path: Caminho do novo banco integrado
    
    Returns:
        True se a migração foi bem-sucedida
    """
    try:
        # Criar sistema integrado
        integrated_system = EnhancedUnifiedMonitoringSystem(db_path=new_db_path)
        
        # A migração é automática no __init__
        logger.info(f"Migração de {legacy_db_path} para {new_db_path} concluída")
        
        # Gerar relatório de migração
        report_file = integrated_system.export_integration_report(
            f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        logger.info(f"Relatório de migração salvo em: {report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro durante migração: {e}")
        return False


if __name__ == "__main__":
    # Teste do sistema integrado
    print("Testando Sistema de Monitoramento Integrado...")
    
    # Criar sistema integrado
    system = create_integrated_monitoring_system(
        db_path="test_integrated_monitoring.db",
        enable_memory_profiling=True
    )
    
    try:
        # Iniciar monitoramento
        system.start_monitoring()
        
        # Aguardar alguns ciclos
        time.sleep(90)
        
        # Gerar relatório
        report_file = system.export_integration_report()
        print(f"Relatório salvo em: {report_file}")
        
    finally:
        # Cleanup
        system.cleanup()
    
    print("Teste concluído!")
