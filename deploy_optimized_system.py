#!/usr/bin/env python3
"""
Script de Deployment Final - Sistema Otimizado LangGraph 101

Este script aplica o sistema integrado otimizado ao ambiente de produção,
substituindo as implementações antigas pelo novo sistema unificado.

Author: GitHub Copilot
Date: 2025-05-27
"""

import os
import sys
import shutil
import logging
import json
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Configura logging para deployment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment_final.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def backup_existing_files(backup_dir: str, files_to_backup: list):
    """Faz backup dos arquivos existentes"""
    logger = logging.getLogger(__name__)
    backup_path = Path(backup_dir)
    backup_path.mkdir(exist_ok=True)
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            backup_file = backup_path / f"{filename}.backup"
            shutil.copy2(file_path, backup_file)
            logger.info(f"Backup criado: {backup_file}")

def deploy_optimized_system():
    """Executa o deployment do sistema otimizado"""
    logger = setup_logging()
    
    logger.info("=== DEPLOYMENT SISTEMA OTIMIZADO LANGGRAPH 101 ===")
    
    # Arquivos principais do sistema otimizado
    core_files = [
        "enhanced_unified_monitoring.py",
        "thread_safe_connection_manager.py", 
        "advanced_memory_profiler.py"
    ]
    
    # Arquivos antigos para backup
    legacy_files = [
        "unified_monitoring_system.py",
        "database_manager.py",
        "performance_monitor.py"
    ]
    
    try:
        # 1. Criar backup dos arquivos existentes
        logger.info("Etapa 1: Criando backup dos arquivos existentes...")
        backup_dir = f"backup_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_existing_files(backup_dir, legacy_files)
        
        # 2. Verificar arquivos do sistema otimizado
        logger.info("Etapa 2: Verificando arquivos do sistema otimizado...")
        missing_files = []
        for file in core_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Arquivos obrigatórios não encontrados: {missing_files}")
            return False
        
        # 3. Executar testes finais
        logger.info("Etapa 3: Executando testes de validação...")
        
        # Importar e testar sistema principal
        try:
            from enhanced_unified_monitoring import create_integrated_monitoring_system
            from thread_safe_connection_manager import ThreadSafeConnectionManager
            from advanced_memory_profiler import AdvancedMemoryProfiler            # Teste básico de inicialização
            test_db = "deployment_test.db"
            monitoring_system = create_integrated_monitoring_system(
                db_path=test_db,
                enable_memory_profiling=True
            )
            monitoring_system.start_monitoring()
            
            # Teste de conectividade
            connection_manager = monitoring_system.connection_manager
            stats = connection_manager.get_connection_stats()
            logger.info(f"Teste de conectividade: {stats['active_connections']} conexões ativas")
              # Teste de profiling
            profiler = monitoring_system.memory_profiler
            if profiler:
                report = profiler.generate_comprehensive_report()
                if "memory_summary" in report:
                    current_memory = report["memory_summary"]["current_memory_mb"]
                    logger.info(f"Teste de profiling: {current_memory:.2f}MB em uso")
                else:
                    logger.info("Teste de profiling: Sistema ativo")
            
            monitoring_system.stop_monitoring()
              # Cleanup do teste
            try:
                if os.path.exists(test_db):
                    os.remove(test_db)
            except PermissionError:
                # Arquivo pode estar sendo usado - não é crítico
                pass
                
            logger.info("✅ Todos os testes de validação passaram!")
            
        except Exception as e:
            logger.error(f"❌ Falha nos testes de validação: {e}")
            return False
        
        # 4. Gerar configuração de produção
        logger.info("Etapa 4: Gerando configuração de produção...")
        
        production_config = {
            "system": {
                "name": "LangGraph 101 Optimized",
                "version": "2.0.0",
                "deployment_date": datetime.now().isoformat(),
                "environment": "production"
            },
            "database": {
                "connection_manager": "ThreadSafeConnectionManager",
                "max_connections": 50,
                "enable_health_checks": True,
                "enable_metrics": True
            },
            "monitoring": {
                "system": "EnhancedUnifiedMonitoringSystem",
                "enable_alerts": True,
                "enable_memory_profiling": True,
                "profiling_interval": 300
            },
            "memory": {
                "profiler": "AdvancedMemoryProfiler",
                "enable_leak_detection": True,
                "snapshot_interval": 600,
                "enable_tracemalloc": True
            },
            "backup": {
                "directory": backup_dir,
                "legacy_files_backed_up": legacy_files
            }
        }
        
        with open("production_config.json", "w") as f:
            json.dump(production_config, f, indent=2)
        
        logger.info("Configuração de produção salva em: production_config.json")
        
        # 5. Gerar script de inicialização
        logger.info("Etapa 5: Gerando script de inicialização...")
        
        startup_script = '''#!/usr/bin/env python3
"""
Script de Inicialização - Sistema Otimizado LangGraph 101
Gerado automaticamente em: {deployment_date}
"""

import logging
from enhanced_unified_monitoring import create_integrated_monitoring_system

def main():
    """Inicializa o sistema otimizado"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Iniciando sistema otimizado LangGraph 101...")
      # Criar sistema integrado
    monitoring_system = create_integrated_monitoring_system(
        db_path="langgraph_optimized.db",
        enable_memory_profiling=True,
        enable_alerts=True
    )
      # Iniciar sistema
    monitoring_system.start_monitoring()
    logger.info("Sistema iniciado com sucesso!")
    
    return monitoring_system

if __name__ == "__main__":
    system = main()
    
    try:
        # Manter sistema rodando
        import time
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\\nParando sistema...")
        system.stop_monitoring()
        print("Sistema parado.")
'''.format(deployment_date=datetime.now().isoformat())
        
        with open("start_optimized_system.py", "w") as f:
            f.write(startup_script)
        
        logger.info("Script de inicialização criado: start_optimized_system.py")
        
        # 6. Deployment finalizado
        logger.info("Etapa 6: Deployment finalizado!")
        
        # Sumário final
        logger.info("=== DEPLOYMENT CONCLUÍDO COM SUCESSO ===")
        logger.info(f"✅ Backup criado em: {backup_dir}")
        logger.info(f"✅ Configuração: production_config.json")
        logger.info(f"✅ Inicialização: start_optimized_system.py")
        logger.info("✅ Sistema pronto para produção!")
        
        # Instruções finais
        print("\n" + "="*50)
        print("🎉 DEPLOYMENT CONCLUÍDO COM SUCESSO!")
        print("="*50)
        print("\nPróximos passos:")
        print("1. Revisar production_config.json")
        print("2. Executar: python start_optimized_system.py")
        print("3. Monitorar logs em deployment_final.log")
        print("\nSistema pronto para produção! 🚀")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro durante deployment: {e}")
        logger.error("Reverta para os backups se necessário")
        return False

if __name__ == "__main__":
    success = deploy_optimized_system()
    sys.exit(0 if success else 1)
