#!/usr/bin/env python3
"""
LangGraph 101 - CLI Integration Patch
====================================

This file patches the existing CLI application (langgraph-101.py) to integrate
with the new infrastructure while maintaining full backward compatibility.

The patch:
- Adds infrastructure integration without breaking existing CLI functionality
- Provides enhanced features when infrastructure is available
- Falls back gracefully to original behavior when infrastructure is unavailable
- Adds monitoring and performance tracking to CLI
- Enhances the CLI with infrastructure status commands

Author: GitHub Copilot
Date: 2024
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import integration wrapper
try:
    from app_integration_wrapper import cli_wrapper, get_enhanced_app, get_integration_status
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    print("⚠️  Integration wrapper not available - running in original mode")

def patch_cli_app():
    """Apply integration patches to CLI application"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    # Return enhanced CLI wrapper
    return cli_wrapper

def enhance_help_menu():
    """Enhance the help menu with infrastructure commands"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    def enhanced_help():
        """Enhanced help menu with infrastructure commands"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            # Fallback to print
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = RED = GREEN = None
        
        print_colored("\\n--- Comandos Básicos ---", Colors.YELLOW, bold=True)
        print_colored("sair       - Encerra a conversa", Colors.WHITE)
        print_colored("ajuda      - Mostra esta mensagem de ajuda", Colors.WHITE)
        print_colored("limpar     - Limpa o histórico da conversa", Colors.WHITE)
        print_colored("cls/clear  - Limpa a tela", Colors.WHITE)
        print_colored("salvar     - Salva o histórico da conversa", Colors.WHITE)
        print_colored("personas   - Lista todas as personas disponíveis", Colors.WHITE)
        print_colored("persona X  - Muda para a persona X (ex: persona Yoda)", Colors.WHITE)
        
        print_colored("\\n--- Comandos de Memória ---", Colors.YELLOW, bold=True)
        print_colored("memória    - Mostra as memórias mais importantes", Colors.WHITE)
        print_colored("esquece    - Limpa todas as memórias armazenadas", Colors.WHITE)
        print_colored("lembra X   - Adiciona manualmente um fato importante à memória", Colors.WHITE)
        
        print_colored("\\n--- Comandos de Exportação ---", Colors.YELLOW, bold=True)
        print_colored("exportar   - Lista os formatos de exportação disponíveis", Colors.WHITE)
        print_colored("exportar X - Exporta a conversa no formato X (ex: exportar html)", Colors.WHITE)
        print_colored("enviar     - Envia a conversa para um email", Colors.WHITE)
        
        # Enhanced infrastructure commands
        print_colored("\\n--- Comandos de Infraestrutura ---", Colors.CYAN, bold=True)
        print_colored("status     - Mostra o status da infraestrutura", Colors.WHITE)
        print_colored("performance- Mostra métricas de desempenho", Colors.WHITE)
        print_colored("health     - Verifica a saúde do sistema", Colors.WHITE)
        print_colored("cache      - Mostra informações do cache", Colors.WHITE)
        print_colored("workers    - Mostra status dos workers", Colors.WHITE)
        print_colored("config     - Mostra configuração atual", Colors.WHITE)
        
        print_colored("------------------------\\n", Colors.YELLOW)
    
    return enhanced_help

def enhance_welcome_message():
    """Enhance the welcome message with infrastructure information"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    def enhanced_welcome():
        """Enhanced welcome message"""
        try:
            from ui import print_colored, Colors, print_welcome
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            def print_welcome():
                print("Welcome to LangGraph 101!")
            
            class Colors:
                CYAN = GREEN = YELLOW = WHITE = None
        
        # Original welcome
        print_welcome()
        
        # Enhanced information
        print_colored("\\n🚀 Enhanced Edition - Infrastructure Integration", Colors.CYAN, bold=True)
        
        integration_status = get_integration_status()
        if integration_status['infrastructure_available']:
            print_colored("✅ Infrastructure Mode: Enhanced features enabled", Colors.GREEN)
            
            # Show available components
            enabled_components = [name for name, loaded in integration_status['components_loaded'].items() if loaded]
            if enabled_components:
                print_colored(f"📦 Active Components: {', '.join(enabled_components)}", Colors.WHITE)
        else:
            print_colored("⚠️  Fallback Mode: Basic features only", Colors.YELLOW)
        
        print_colored("💡 Type 'status' for infrastructure information", Colors.WHITE)
        print_colored("💡 Type 'ajuda' for all available commands\\n", Colors.WHITE)
    
    return enhanced_welcome

def enhance_command_processor():
    """Enhance the command processor with infrastructure commands"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    def process_infrastructure_command(command: str) -> bool:
        """Process infrastructure-specific commands"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = RED = None
        
        command = command.lower().strip()
        
        if command == 'status':
            show_infrastructure_status()
            return True
        elif command == 'performance':
            show_performance_metrics()
            return True
        elif command == 'health':
            show_health_check()
            return True
        elif command == 'cache':
            show_cache_info()
            return True
        elif command == 'workers':
            show_worker_status()
            return True
        elif command == 'config':
            show_current_config()
            return True
        
        return False
    
    def show_infrastructure_status():
        """Show infrastructure status"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = RED = None
        
        print_colored("\\n🏗️ Status da Infraestrutura:", Colors.YELLOW, bold=True)
        
        integration_status = get_integration_status()
        
        # Overall status
        if integration_status['infrastructure_available']:
            print_colored("✅ Infraestrutura: Disponível", Colors.GREEN)
        else:
            print_colored("❌ Infraestrutura: Indisponível", Colors.RED)
        
        if integration_status['fallback_mode']:
            print_colored("⚠️  Modo: Fallback (funcionalidade básica)", Colors.YELLOW)
        else:
            print_colored("🚀 Modo: Completo (todas as funcionalidades)", Colors.GREEN)
        
        # Component status
        print_colored("\\n📦 Componentes:", Colors.CYAN)
        for component, loaded in integration_status['components_loaded'].items():
            icon = "✅" if loaded else "❌"
            status = "Carregado" if loaded else "Não disponível"
            component_name = component.replace('_', ' ').title()
            print_colored(f"  {icon} {component_name}: {status}", Colors.WHITE)
        
        print()
    
    def show_performance_metrics():
        """Show performance metrics"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = None
        
        print_colored("\\n📊 Métricas de Desempenho:", Colors.YELLOW, bold=True)
        
        integration_status = get_integration_status()
        perf_metrics = integration_status['performance_metrics']
        
        if perf_metrics:
            for metric, stats in perf_metrics.items():
                metric_name = metric.replace('_', ' ').title()
                print_colored(f"\\n{metric_name}:", Colors.CYAN)
                print_colored(f"  Média: {stats['avg']:.3f}s", Colors.WHITE)
                print_colored(f"  Mínimo: {stats['min']:.3f}s", Colors.WHITE)
                print_colored(f"  Máximo: {stats['max']:.3f}s", Colors.WHITE)
                print_colored(f"  Último: {stats['last']:.3f}s", Colors.WHITE)
                print_colored(f"  Total: {stats['count']} medições", Colors.WHITE)
        else:
            print_colored("Nenhuma métrica disponível ainda.", Colors.YELLOW)
        
        print()
    
    def show_health_check():
        """Show system health check"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = RED = None
        
        print_colored("\\n🏥 Verificação de Saúde do Sistema:", Colors.YELLOW, bold=True)
        
        enhanced_app = get_enhanced_app()
        
        # Check infrastructure hub
        if enhanced_app.infrastructure_hub:
            print_colored("✅ Infrastructure Hub: Ativo", Colors.GREEN)
            
            # Try to get health status
            try:
                import asyncio
                health_status = asyncio.run(enhanced_app.infrastructure_hub.get_health_status())
                
                for component, status in health_status.items():
                    if isinstance(status, dict) and 'status' in status:
                        component_status = status['status']
                        icon = "✅" if component_status == 'healthy' else "❌"
                        print_colored(f"  {icon} {component}: {component_status}", Colors.WHITE)
                    else:
                        print_colored(f"  📊 {component}: {status}", Colors.WHITE)
            except Exception as e:
                print_colored(f"  ⚠️  Erro ao verificar saúde: {e}", Colors.YELLOW)
        else:
            print_colored("❌ Infrastructure Hub: Inativo", Colors.RED)
        
        # System resources
        try:
            import psutil
            
            print_colored("\\n💻 Recursos do Sistema:", Colors.CYAN)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_color = Colors.GREEN if cpu_percent < 70 else Colors.YELLOW if cpu_percent < 90 else Colors.RED
            print_colored(f"  CPU: {cpu_percent}%", cpu_color)
            
            # Memory usage
            memory = psutil.virtual_memory()
            mem_color = Colors.GREEN if memory.percent < 70 else Colors.YELLOW if memory.percent < 90 else Colors.RED
            print_colored(f"  Memória: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)", mem_color)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_color = Colors.GREEN if disk_percent < 80 else Colors.YELLOW if disk_percent < 95 else Colors.RED
            print_colored(f"  Disco: {disk_percent:.1f}% ({disk.free // (1024**3)}GB livres)", disk_color)
            
        except ImportError:
            print_colored("  ⚠️  psutil não disponível para métricas de sistema", Colors.YELLOW)
        except Exception as e:
            print_colored(f"  ⚠️  Erro ao obter métricas de sistema: {e}", Colors.YELLOW)
        
        print()
    
    def show_cache_info():
        """Show cache information"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = RED = None
        
        print_colored("\\n🗄️ Informações do Cache:", Colors.YELLOW, bold=True)
        
        enhanced_app = get_enhanced_app()
        
        if enhanced_app.infrastructure_hub and enhanced_app.infrastructure_hub.cache_manager:
            try:
                cache_manager = enhanced_app.infrastructure_hub.cache_manager
                
                # Get cache stats
                cache_stats = cache_manager.get_stats()
                
                print_colored("Status do Cache:", Colors.CYAN)
                print_colored(f"  Hits: {cache_stats.get('hits', 0)}", Colors.WHITE)
                print_colored(f"  Misses: {cache_stats.get('misses', 0)}", Colors.WHITE)
                
                hit_rate = cache_stats.get('hit_rate', 0) * 100
                hit_color = Colors.GREEN if hit_rate > 80 else Colors.YELLOW if hit_rate > 50 else Colors.RED
                print_colored(f"  Taxa de Acerto: {hit_rate:.1f}%", hit_color)
                
                print_colored(f"  Chaves Ativas: {cache_stats.get('keys', 0)}", Colors.WHITE)
                
            except Exception as e:
                print_colored(f"  ⚠️  Erro ao obter stats do cache: {e}", Colors.YELLOW)
        else:
            print_colored("❌ Cache Manager não disponível", Colors.RED)
        
        print()
    
    def show_worker_status():
        """Show worker status"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = RED = None
        
        print_colored("\\n👷 Status dos Workers:", Colors.YELLOW, bold=True)
        
        enhanced_app = get_enhanced_app()
        
        if enhanced_app.infrastructure_hub and enhanced_app.infrastructure_hub.message_queue:
            try:
                # Get worker stats from message queue
                queue_system = enhanced_app.infrastructure_hub.message_queue
                
                print_colored("Message Queue:", Colors.CYAN)
                print_colored("  Status: Ativo", Colors.GREEN)
                
                # Try to get Celery inspection info
                try:
                    from celery import current_app
                    inspect = current_app.control.inspect()
                    
                    active_tasks = inspect.active()
                    if active_tasks:
                        total_active = sum(len(tasks) for tasks in active_tasks.values())
                        print_colored(f"  Tarefas Ativas: {total_active}", Colors.WHITE)
                    else:
                        print_colored("  Tarefas Ativas: 0", Colors.WHITE)
                    
                except Exception:
                    print_colored("  ⚠️  Informações detalhadas não disponíveis", Colors.YELLOW)
                
            except Exception as e:
                print_colored(f"  ⚠️  Erro ao obter status dos workers: {e}", Colors.YELLOW)
        else:
            print_colored("❌ Message Queue não disponível", Colors.RED)
        
        print()
    
    def show_current_config():
        """Show current configuration"""
        try:
            from ui import print_colored, Colors
        except ImportError:
            def print_colored(text, color=None, bold=False):
                print(text)
            
            class Colors:
                YELLOW = CYAN = WHITE = GREEN = None
        
        print_colored("\\n⚙️ Configuração Atual:", Colors.YELLOW, bold=True)
        
        enhanced_app = get_enhanced_app()
        
        if enhanced_app.config:
            config = enhanced_app.config
            
            print_colored("Ambiente:", Colors.CYAN)
            print_colored(f"  Ambiente: {config.environment}", Colors.WHITE)
            print_colored(f"  Debug: {'Sim' if config.debug else 'Não'}", Colors.WHITE)
            print_colored(f"  Versão: {config.config_version}", Colors.WHITE)
            
            print_colored("\\nServiços:", Colors.CYAN)
            print_colored(f"  Porta do Adapter: {config.services.adapter_port}", Colors.WHITE)
            print_colored(f"  Porta do Streamlit: {config.services.streamlit_port}", Colors.WHITE)
            print_colored(f"  Porta do Gateway: {config.services.gateway_port}", Colors.WHITE)
            
            print_colored("\\nComponentes:", Colors.CYAN)
            components = config.components
            print_colored(f"  API Gateway: {'Ativo' if components.enable_api_gateway else 'Inativo'}", Colors.WHITE)
            print_colored(f"  Message Queue: {'Ativo' if components.enable_message_queue else 'Inativo'}", Colors.WHITE)
            print_colored(f"  Cache Manager: {'Ativo' if components.enable_cache_manager else 'Inativo'}", Colors.WHITE)
            print_colored(f"  Rate Limiting: {'Ativo' if components.enable_rate_limiting else 'Inativo'}", Colors.WHITE)
            
        else:
            print_colored("❌ Configuração integrada não disponível", Colors.RED)
        
        print()
    
    return process_infrastructure_command

def patch_main_loop():
    """Patch the main interaction loop"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    def enhanced_main_loop():
        """Enhanced main loop with infrastructure features"""
        
        # Import necessary modules
        try:
            from ui import get_user_input, print_colored, Colors, clear_screen
        except ImportError:
            def get_user_input():
                return input(">>> ")
            def print_colored(text, color=None, bold=False):
                print(text)
            def clear_screen():
                os.system('cls' if os.name == 'nt' else 'clear')
            
            class Colors:
                CYAN = GREEN = YELLOW = WHITE = RED = None
        
        # Get enhanced components
        enhanced_app = get_enhanced_app()
        infrastructure_processor = enhance_command_processor()
        
        # Main interaction loop
        while True:
            try:
                user_input = get_user_input()
                
                if not user_input.strip():
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    print_colored("👋 Encerrando LangGraph 101...", Colors.CYAN)
                    break
                
                # Check for clear commands
                if user_input.lower() in ['cls', 'clear']:
                    clear_screen()
                    continue
                
                # Check for help commands
                if user_input.lower() in ['ajuda', 'help']:
                    enhanced_help = enhance_help_menu()
                    if enhanced_help:
                        enhanced_help()
                    continue
                
                # Check for infrastructure commands
                if infrastructure_processor and infrastructure_processor(user_input):
                    continue
                
                # Process regular chat message with enhanced features
                try:
                    result = enhanced_app.process_message(user_input)
                    
                    if result.get('status') == 'success':
                        response = result.get('response', '')
                        print_colored(response, Colors.WHITE)
                        
                        # Show mode indicator
                        mode = result.get('mode', 'enhanced')
                        if mode == 'fallback':
                            print_colored("(modo fallback)", Colors.YELLOW)
                        elif mode == 'enhanced':
                            print_colored("(modo aprimorado)", Colors.GREEN)
                    else:
                        print_colored(f"Erro: {result.get('response', 'Erro desconhecido')}", Colors.RED)
                
                except Exception as e:
                    print_colored(f"Erro no processamento: {e}", Colors.RED)
                
            except KeyboardInterrupt:
                print_colored("\\n👋 Encerrando LangGraph 101...", Colors.CYAN)
                break
            except Exception as e:
                print_colored(f"Erro inesperado: {e}", Colors.RED)
    
    return enhanced_main_loop

def apply_cli_patches():
    """Apply all CLI patches"""
    
    if not INTEGRATION_AVAILABLE:
        return {
            'cli_wrapper': None,
            'enhanced_help': None,
            'enhanced_welcome': None,
            'enhanced_main_loop': None,
            'infrastructure_processor': None
        }
    
    return {
        'cli_wrapper': patch_cli_app(),
        'enhanced_help': enhance_help_menu(),
        'enhanced_welcome': enhance_welcome_message(),
        'enhanced_main_loop': patch_main_loop(),
        'infrastructure_processor': enhance_command_processor()
    }

# Export for use in main CLI application
__all__ = [
    'patch_cli_app',
    'enhance_help_menu',
    'enhance_welcome_message', 
    'enhance_command_processor',
    'patch_main_loop',
    'apply_cli_patches'
]

# Auto-apply patches when imported
if __name__ != "__main__":
    patches = apply_cli_patches()
    
    # Make patches available globally
    globals().update(patches)
