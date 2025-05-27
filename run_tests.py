"""
Script para executar todos os testes unitários do projeto LangGraph 101.

Este script executa todos os testes e gera um relatório de cobertura.
Para executar: python run_tests.py
"""

import os
import sys
import subprocess
import argparse


def run_tests(verbose=False, with_coverage=True):
    """Executar todos os testes unitários.

    Args:
        verbose: Se True, exibe informações detalhadas sobre os testes.
        with_coverage: Se True, gera relatório de cobertura.

    Returns:
        Código de saída do pytest (0 para sucesso, diferente de 0 para falha).
    """
    print("=" * 50)
    print("Executando testes para o projeto LangGraph 101")
    print("=" * 50)

    # Verificar se o diretório de testes existe
    if not os.path.exists("tests"):
        print("Diretório de testes não encontrado!")
        return 1

    # Construir o comando de teste
    cmd = ["pytest", "tests/"]

    if verbose:
        cmd.append("-v")

    if with_coverage:
        cmd.extend(["--cov=.", "--cov-report=term", "--cov-report=html:coverage_report"])

    # Executar o comando
    result = subprocess.run(cmd)

    if with_coverage:
        coverage_report_path = "coverage_report/index.html"
        if os.path.exists(coverage_report_path):
            print(f"\nRelatório de cobertura HTML gerado em: {os.path.abspath(coverage_report_path)}")
        else:
            print(f"\n⚠️  Falha ao gerar o relatório de cobertura HTML. Verifique a saída do pytest-cov.")

    if result.returncode == 0:
        print("\n✅ Todos os testes passaram com sucesso!")
    else:
        print("\n❌ Alguns testes falharam. Verifique os erros acima.")

    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa os testes unitários do LangGraph 101")
    parser.add_argument("-v", "--verbose", action="store_true", help="Exibe informações detalhadas")
    parser.add_argument("--no-coverage", action="store_true", help="Não gera relatório de cobertura")

    args = parser.parse_args()

    exit_code = run_tests(verbose=args.verbose, with_coverage=not args.no_coverage)
    sys.exit(exit_code)
