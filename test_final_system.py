#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Final do Sistema Otimizado LangGraph 101
"""

import sys
import time
import logging
from enhanced_unified_monitoring import create_integrated_monitoring_system

def test_system():
    """Testa o sistema otimizado"""
    print("=== TESTE FINAL DO SISTEMA OTIMIZADO ===")
    
    try:
        # Configurar logging
        logging.basicConfig(level=logging.WARNING)  # Apenas warnings e erros
        print("1. Criando sistema integrado...")
        system = create_integrated_monitoring_system(
            db_path="test_final.db",
            enable_memory_profiling=True
        )
        
        print("2. Iniciando monitoramento...")
        system.start_monitoring()
        
        print("3. Sistema ativo por 10 segundos...")
        time.sleep(10)
        
        print("4. Coletando estatísticas...")
        conn_stats = system.connection_manager.get_connection_stats()
        print(f"   - Conexões ativas: {conn_stats['active_connections']}")
        print(f"   - Total de queries: {conn_stats['total_queries']}")
        print(f"   - Taxa de erro: {conn_stats['error_rate']:.1%}")
        
        if system.memory_profiler:
            report = system.memory_profiler.generate_comprehensive_report()
            if "memory_summary" in report:
                memory_mb = report["memory_summary"]["current_memory_mb"]
                print(f"   - Memória atual: {memory_mb:.1f}MB")
            else:
                print("   - Profiling de memória: Ativo")
        
        print("5. Parando sistema...")
        system.stop_monitoring()
        
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("Sistema otimizado está funcionando perfeitamente!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        return False
    
    finally:
        try:
            system.cleanup()
        except:
            pass

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
