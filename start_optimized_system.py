#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inicialização - Sistema Otimizado LangGraph 101
Gerado automaticamente em: 2025-05-27T10:03:06.869334
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
        print("\nParando sistema...")
        system.stop_monitoring()
        print("Sistema parado.")
