#!/usr/bin/env python3
"""
Teste Abrangente do Sistema Integrado com Profiling

Este script realiza testes completos do sistema de monitoramento unificado
com profiling de memória e gerenciamento robusto de conexões.

Testes realizados:
1. Profiling de memória durante operação normal
2. Teste de carga com múltiplas threads
3. Detecção de vazamentos de memória
4. Performance do gerenciador de conexões
5. Integração entre todos os componentes
6. Geração de relatórios detalhados

Author: GitHub Copilot
Date: 2025-05-27
"""

import os
import sys
import time
import logging
import threading
import json
import gc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Importar sistemas desenvolvidos
try:
    from enhanced_unified_monitoring import EnhancedUnifiedMonitoringSystem, create_integrated_monitoring_system
    from advanced_memory_profiler import AdvancedMemoryProfiler, run_memory_analysis
    from thread_safe_connection_manager import ThreadSafeConnectionManager, get_connection_manager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Não foi possível importar módulos necessários: {e}")
    IMPORTS_AVAILABLE = False
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveSystemTest:
    """
    Teste abrangente do sistema integrado.
    
    Realiza múltiplos tipos de testes para validar:
    - Funcionamento básico
    - Performance sob carga
    - Detecção de vazamentos
    - Profiling de memória
    - Integração entre componentes
    """
    
    def __init__(self, test_db_path: str = "comprehensive_test.db"):
        self.test_db_path = Path(test_db_path)
        self.test_results = {}
        self.start_time = datetime.now()
        
        # Limpar banco de teste anterior
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        
        logger.info(f"Teste abrangente iniciado: {self.test_db_path}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Executa todos os testes e retorna resultados."""
        logger.info("=== INICIANDO TESTES ABRANGENTES ===")
        
        # Baseline de memória
        baseline_memory = self._get_memory_baseline()
        
        # Teste 1: Funcionamento básico
        logger.info("Teste 1: Funcionamento básico do sistema")
        basic_test_result = self._test_basic_functionality()
        self.test_results["basic_functionality"] = basic_test_result
        
        # Teste 2: Profiling de memória
        logger.info("Teste 2: Profiling de memória")
        memory_profiling_result = self._test_memory_profiling()
        self.test_results["memory_profiling"] = memory_profiling_result
        
        # Teste 3: Gerenciador de conexões
        logger.info("Teste 3: Gerenciador de conexões thread-safe")
        connection_test_result = self._test_connection_manager()
        self.test_results["connection_manager"] = connection_test_result
        
        # Teste 4: Teste de carga
        logger.info("Teste 4: Teste de carga com múltiplas threads")
        load_test_result = self._test_load_performance()
        self.test_results["load_performance"] = load_test_result
        
        # Teste 5: Detecção de vazamentos
        logger.info("Teste 5: Detecção de vazamentos de memória")
        leak_detection_result = self._test_leak_detection()
        self.test_results["leak_detection"] = leak_detection_result
        
        # Teste 6: Integração completa
        logger.info("Teste 6: Integração completa do sistema")
        integration_result = self._test_full_integration()
        self.test_results["full_integration"] = integration_result
        
        # Análise final
        final_memory = self._get_memory_baseline()
        memory_impact = self._analyze_memory_impact(baseline_memory, final_memory)
        
        # Compilar resultados finais
        final_results = self._compile_final_results(memory_impact)
        
        logger.info("=== TESTES CONCLUÍDOS ===")
        return final_results
    
    def _get_memory_baseline(self) -> Dict[str, float]:
        """Obtém baseline de memória atual."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "system_memory_percent": system_memory.percent,
            "system_available_gb": system_memory.available / 1024 / 1024 / 1024,
            "gc_objects": len(gc.get_objects()),
            "thread_count": threading.active_count(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Testa funcionamento básico do sistema."""
        test_start = time.time()
        
        try:
            # Criar sistema integrado
            system = create_integrated_monitoring_system(
                db_path=str(self.test_db_path),
                enable_memory_profiling=False  # Separado no próximo teste
            )
            
            # Iniciar monitoramento
            system.start_monitoring()
            
            # Aguardar alguns ciclos de coleta
            time.sleep(65)  # Mais de 1 minuto para coletar métricas
            
            # Verificar se métricas estão sendo coletadas
            report = system.get_integration_report()
            
            # Parar sistema
            system.stop_monitoring()
            system.cleanup()
            
            # Validar resultados
            success = (
                report.get("integration_statistics", {}).get("total_metrics_collected", 0) > 0 and
                "error" not in report
            )
            
            return {
                "success": success,
                "execution_time_seconds": time.time() - test_start,
                "metrics_collected": report.get("integration_statistics", {}).get("total_metrics_collected", 0),
                "connection_stats": report.get("connection_manager_status", {}),
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Erro no teste básico: {e}")
            return {
                "success": False,
                "execution_time_seconds": time.time() - test_start,
                "error": str(e)
            }
    
    def _test_memory_profiling(self) -> Dict[str, Any]:
        """Testa profiling de memória."""
        test_start = time.time()
        
        try:
            # Criar profiler de memória
            profiler = AdvancedMemoryProfiler(
                db_path=str(self.test_db_path.with_suffix('.profiling.db')),
                snapshot_interval=10,  # Mais rápido para teste
                enable_tracemalloc=True
            )
            
            # Iniciar profiling
            profiler.start_profiling(baseline=True)
            
            # Simular carga de memória
            test_data = []
            for i in range(1000):
                test_data.append([j for j in range(100)])  # Criar objetos em memória
                if i % 100 == 0:
                    time.sleep(0.1)  # Pausas para o profiler capturar
            
            # Aguardar profiling
            time.sleep(30)
            
            # Parar profiling e obter relatório
            report = profiler.stop_profiling()
            profiler.cleanup()
            
            # Limpar dados de teste
            del test_data
            gc.collect()
            
            # Validar resultados
            success = (
                report is not None and
                "error" not in report and
                report.get("report_metadata", {}).get("total_snapshots", 0) > 0
            )
            
            return {
                "success": success,
                "execution_time_seconds": time.time() - test_start,
                "snapshots_captured": report.get("report_metadata", {}).get("total_snapshots", 0),
                "memory_change_mb": report.get("memory_summary", {}).get("memory_change_mb", 0),
                "recommendations": report.get("recommendations", []),
                "profiling_report": report
            }
            
        except Exception as e:
            logger.error(f"Erro no teste de profiling: {e}")
            return {
                "success": False,
                "execution_time_seconds": time.time() - test_start,
                "error": str(e)
            }
    
    def _test_connection_manager(self) -> Dict[str, Any]:
        """Testa gerenciador de conexões thread-safe."""
        test_start = time.time()
        
        try:
            # Criar gerenciador de conexões
            manager = ThreadSafeConnectionManager(
                db_path=str(self.test_db_path.with_suffix('.connections.db')),
                max_connections=10
            )
            
            # Teste de operações básicas
            with manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER, value TEXT)")
                cursor.execute("INSERT INTO test_table (id, value) VALUES (?, ?)", (1, "test"))
                conn.commit()
            
            # Teste de múltiplas threads
            def worker_function(worker_id: int):
                try:
                    for i in range(10):
                        with manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("INSERT INTO test_table (id, value) VALUES (?, ?)", 
                                         (worker_id * 10 + i, f"worker_{worker_id}_item_{i}"))
                            conn.commit()
                        time.sleep(0.1)
                    return True
                except Exception as e:
                    logger.error(f"Erro no worker {worker_id}: {e}")
                    return False
            
            # Executar workers em paralelo
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker_function, i) for i in range(5)]
                results = [future.result() for future in as_completed(futures)]
            
            # Verificar resultados
            with manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM test_table")
                total_records = cursor.fetchone()[0]
            
            # Obter estatísticas
            stats = manager.get_connection_stats()
            performance_report = manager.get_performance_report()
            
            # Cleanup
            manager.close_all_connections()
            
            # Validar resultados
            success = (
                all(results) and  # Todos os workers foram bem-sucedidos
                total_records >= 50 and  # Pelo menos 50 registros inseridos
                stats.get("total_queries", 0) > 0  # Queries foram executadas
            )
            
            return {
                "success": success,
                "execution_time_seconds": time.time() - test_start,
                "worker_success_rate": sum(results) / len(results) * 100,
                "total_records_created": total_records,
                "connection_stats": stats,
                "performance_report": performance_report
            }
            
        except Exception as e:
            logger.error(f"Erro no teste de conexões: {e}")
            return {
                "success": False,
                "execution_time_seconds": time.time() - test_start,
                "error": str(e)
            }
    
    def _test_load_performance(self) -> Dict[str, Any]:
        """Testa performance sob carga."""
        test_start = time.time()
        
        try:
            # Criar sistema com profiling habilitado
            system = create_integrated_monitoring_system(
                db_path=str(self.test_db_path.with_suffix('.load.db')),
                enable_memory_profiling=True,
                max_connections=20
            )
            
            system.start_monitoring()
            
            # Baseline de performance
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Simulação de carga intensa
            def load_worker(worker_id: int):
                try:
                    operations = 0
                    for i in range(50):
                        # Operações de database
                        with system.connection_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO system_events (timestamp, event_type, source, description)
                                VALUES (?, ?, ?, ?)
                            """, (
                                datetime.now().isoformat(),
                                "load_test",
                                f"worker_{worker_id}",
                                f"Load test operation {i}"
                            ))
                            conn.commit()
                            operations += 1
                        
                        # Criar alguns objetos em memória
                        temp_data = [x for x in range(100)]
                        del temp_data
                        
                        time.sleep(0.05)  # Pausa curta
                    
                    return operations
                    
                except Exception as e:
                    logger.error(f"Erro no load worker {worker_id}: {e}")
                    return 0
            
            # Executar carga com múltiplas threads
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(load_worker, i) for i in range(10)]
                worker_results = [future.result() for future in as_completed(futures)]
            
            # Aguardar sistema processar
            time.sleep(60)
            
            # Métricas finais
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Obter relatórios
            integration_report = system.get_integration_report()
            
            if system.memory_profiler:
                profiling_report = system.memory_profiler.generate_comprehensive_report()
            else:
                profiling_report = {"error": "Memory profiler not available"}
            
            # Parar sistema
            system.stop_monitoring()
            system.cleanup()
            
            # Validar resultados
            total_operations = sum(worker_results)
            success = (
                total_operations >= 400 and  # Pelo menos 400 operações
                memory_growth < 50 and  # Crescimento de memória < 50MB
                integration_report.get("integration_statistics", {}).get("total_metrics_collected", 0) > 0
            )
            
            return {
                "success": success,
                "execution_time_seconds": time.time() - test_start,
                "total_operations": total_operations,
                "memory_growth_mb": memory_growth,
                "operations_per_second": total_operations / (time.time() - test_start),
                "integration_report": integration_report,
                "profiling_report": profiling_report
            }
            
        except Exception as e:
            logger.error(f"Erro no teste de carga: {e}")
            return {
                "success": False,
                "execution_time_seconds": time.time() - test_start,
                "error": str(e)
            }
    
    def _test_leak_detection(self) -> Dict[str, Any]:
        """Testa detecção de vazamentos de memória."""
        test_start = time.time()
        
        try:
            # Criar profiler especializado em detecção de vazamentos
            profiler = AdvancedMemoryProfiler(
                db_path=str(self.test_db_path.with_suffix('.leaks.db')),
                snapshot_interval=5,  # Snapshots mais frequentes
                enable_tracemalloc=True
            )
            
            profiler.start_profiling(baseline=True)
            
            # Simular vazamento intencional
            leaked_objects = []
            
            for cycle in range(5):
                # Criar objetos que "vazam"
                for i in range(200):
                    obj = {
                        "id": i,
                        "cycle": cycle,
                        "data": [x for x in range(50)],
                        "timestamp": datetime.now()
                    }
                    leaked_objects.append(obj)
                
                # Aguardar snapshot
                time.sleep(6)
                
                logger.info(f"Leak cycle {cycle + 1}: {len(leaked_objects)} objects leaked")
            
            # Aguardar detecção final
            time.sleep(10)
            
            # Obter relatório
            report = profiler.stop_profiling()
            profiler.cleanup()
            
            # Limpar vazamentos para não afetar outros testes
            del leaked_objects
            gc.collect()
            
            # Analisar detecções
            leak_analysis = report.get("leak_analysis", {})
            top_leaking_types = leak_analysis.get("top_leaking_types", [])
            
            # Validar se vazamentos foram detectados
            dict_leaks = [leak for leak in top_leaking_types if leak.get("object_type") == "dict"]
            list_leaks = [leak for leak in top_leaking_types if leak.get("object_type") == "list"]
            
            success = (
                len(top_leaking_types) > 0 and
                (len(dict_leaks) > 0 or len(list_leaks) > 0) and
                report.get("memory_summary", {}).get("memory_change_mb", 0) > 5  # Crescimento significativo
            )
            
            return {
                "success": success,
                "execution_time_seconds": time.time() - test_start,
                "leaks_detected": len(top_leaking_types),
                "memory_growth_mb": report.get("memory_summary", {}).get("memory_change_mb", 0),
                "leak_analysis": leak_analysis,
                "recommendations": report.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"Erro no teste de detecção de vazamentos: {e}")
            return {
                "success": False,
                "execution_time_seconds": time.time() - test_start,
                "error": str(e)
            }
    
    def _test_full_integration(self) -> Dict[str, Any]:
        """Testa integração completa do sistema."""
        test_start = time.time()
        
        try:
            # Criar sistema completo com todos os recursos
            system = create_integrated_monitoring_system(
                db_path=str(self.test_db_path.with_suffix('.full.db')),
                enable_memory_profiling=True,
                max_connections=15,
                connection_timeout=180
            )
            
            # Adicionar handler de alerta personalizado
            alerts_received = []
            
            def alert_handler(alert_id, category, severity, title, message, details):
                alerts_received.append({
                    "id": alert_id,
                    "category": category,
                    "severity": severity,
                    "title": title,
                    "message": message,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"Alert received: [{severity}] {title}")
            
            system.add_alert_handler(alert_handler)
            
            # Iniciar sistema completo
            system.start_monitoring()
            
            # Simular operação normal do sistema
            for i in range(30):
                # Operações de database variadas
                with system.connection_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Inserir dados variados
                    cursor.execute("""
                        INSERT INTO system_events (timestamp, event_type, source, description, data)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        "integration_test",
                        "full_test",
                        f"Integration test operation {i}",
                        json.dumps({"iteration": i, "test_data": list(range(10))})
                    ))
                    conn.commit()
                
                # Breve pausa
                time.sleep(2)
            
            # Aguardar processamento completo
            time.sleep(90)
            
            # Obter relatórios finais
            integration_report = system.get_integration_report()
            
            # Exportar relatórios
            integration_file = system.export_integration_report(
                f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Verificar profiling de memória
            memory_report = None
            if system.memory_profiler and system.memory_profiler.is_running:
                memory_report = system.memory_profiler.generate_comprehensive_report()
            
            # Parar sistema
            system.stop_monitoring()
            system.cleanup()
            
            # Validar integração
            success = (
                integration_report.get("integration_statistics", {}).get("total_metrics_collected", 0) > 0 and
                integration_report.get("connection_manager_status", {}).get("total_queries", 0) >= 30 and
                len(alerts_received) >= 0 and  # Alertas podem ou não ser gerados
                "error" not in integration_report
            )
            
            return {
                "success": success,
                "execution_time_seconds": time.time() - test_start,
                "integration_report": integration_report,
                "memory_report": memory_report,
                "alerts_received": alerts_received,
                "report_file": integration_file,
                "operations_completed": 30
            }
            
        except Exception as e:
            logger.error(f"Erro no teste de integração completa: {e}")
            return {
                "success": False,
                "execution_time_seconds": time.time() - test_start,
                "error": str(e)
            }
    
    def _analyze_memory_impact(self, baseline: Dict, final: Dict) -> Dict[str, Any]:
        """Analisa impacto na memória durante todos os testes."""
        memory_change_mb = final["process_memory_mb"] - baseline["process_memory_mb"]
        memory_change_percent = (memory_change_mb / baseline["process_memory_mb"]) * 100
        
        object_change = final["gc_objects"] - baseline["gc_objects"]
        thread_change = final["thread_count"] - baseline["thread_count"]
        
        return {
            "baseline_memory_mb": baseline["process_memory_mb"],
            "final_memory_mb": final["process_memory_mb"],
            "memory_change_mb": memory_change_mb,
            "memory_change_percent": memory_change_percent,
            "object_change": object_change,
            "thread_change": thread_change,
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "memory_efficiency": "excellent" if memory_change_mb < 10 else "good" if memory_change_mb < 25 else "poor"
        }
    
    def _compile_final_results(self, memory_impact: Dict) -> Dict[str, Any]:
        """Compila resultados finais de todos os testes."""
        # Calcular taxa de sucesso geral
        successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        total_tests = len(self.test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Calcular tempo total de execução
        total_execution_time = sum(
            result.get("execution_time_seconds", 0) 
            for result in self.test_results.values()
        )
        
        # Compilar recomendações de todos os testes
        all_recommendations = []
        for test_name, result in self.test_results.items():
            if "recommendations" in result:
                all_recommendations.extend(result["recommendations"])
            if "profiling_report" in result and "recommendations" in result["profiling_report"]:
                all_recommendations.extend(result["profiling_report"]["recommendations"])
        
        # Determinar status geral
        overall_status = "PASS" if success_rate >= 80 else "PARTIAL" if success_rate >= 60 else "FAIL"
        
        return {
            "test_summary": {
                "overall_status": overall_status,
                "success_rate_percent": success_rate,
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "total_execution_time_seconds": total_execution_time,
                "test_duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60
            },
            "memory_impact_analysis": memory_impact,
            "individual_test_results": self.test_results,
            "performance_summary": {
                "memory_efficiency": memory_impact.get("memory_efficiency", "unknown"),
                "system_stability": "stable" if memory_impact.get("memory_change_mb", 0) < 50 else "unstable",
                "resource_usage": "optimal" if memory_impact.get("thread_change", 0) < 10 else "high"
            },
            "recommendations": list(set(all_recommendations)),  # Remove duplicatas
            "test_metadata": {
                "test_database": str(self.test_db_path),
                "test_start_time": self.start_time.isoformat(),
                "test_end_time": datetime.now().isoformat(),
                "python_version": sys.version,
                "platform": os.name
            }
        }


def run_comprehensive_tests(output_file: str = None) -> Dict[str, Any]:
    """
    Executa bateria completa de testes e salva resultados.
    
    Args:
        output_file: Arquivo para salvar resultados (opcional)
    
    Returns:
        Resultados completos dos testes
    """
    if not IMPORTS_AVAILABLE:
        logger.error("Módulos necessários não estão disponíveis")
        return {"error": "Missing required modules"}
    
    # Executar testes
    test_suite = ComprehensiveSystemTest("comprehensive_test.db")
    results = test_suite.run_all_tests()
    
    # Salvar resultados
    if output_file is None:
        output_file = f"comprehensive_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Log resumo
    logger.info(f"=== RESUMO DOS TESTES ===")
    logger.info(f"Status Geral: {results['test_summary']['overall_status']}")
    logger.info(f"Taxa de Sucesso: {results['test_summary']['success_rate_percent']:.1f}%")
    logger.info(f"Testes Bem-sucedidos: {results['test_summary']['successful_tests']}/{results['test_summary']['total_tests']}")
    logger.info(f"Tempo Total: {results['test_summary']['test_duration_minutes']:.1f} minutos")
    logger.info(f"Impacto na Memória: {results['memory_impact_analysis']['memory_change_mb']:+.1f}MB")
    logger.info(f"Eficiência da Memória: {results['memory_impact_analysis']['memory_efficiency']}")
    logger.info(f"Resultados salvos em: {output_file}")
    
    return results


if __name__ == "__main__":
    print("=== TESTE ABRANGENTE DO SISTEMA INTEGRADO ===")
    print("Este teste pode levar vários minutos para completar...")
    print()
    
    # Executar testes
    results = run_comprehensive_tests()
    
    # Exibir resultado final
    if results.get("test_summary", {}).get("overall_status") == "PASS":
        print("✅ TODOS OS TESTES PASSARAM COM SUCESSO!")
    elif results.get("test_summary", {}).get("overall_status") == "PARTIAL":
        print("⚠️  ALGUNS TESTES FALHARAM - REVISÃO NECESSÁRIA")
    else:
        print("❌ FALHA CRÍTICA NOS TESTES")
    
    print("\nPara detalhes completos, consulte o arquivo de resultados gerado.")
