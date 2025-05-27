"""
Testes unitários para o módulo personas.py
"""

import pytest
from personas import (
    Persona, get_all_personas, get_persona_by_name, get_default_persona,
    don_corleone, sherlock_holmes, yoda, marie_curie, shakespeare
)


class TestPersona:
    """Testes para a classe Persona."""

    def test_persona_initialization(self):
        """Testa a inicialização básica de uma persona."""
        persona = Persona(
            name="Test Persona",
            description="Uma persona de teste",
            system_prompt="Você é uma persona de teste. Aja como um robô."
        )

        # Verificar se os atributos foram definidos corretamente
        assert persona.name == "Test Persona"
        assert persona.description == "Uma persona de teste"
        assert persona.system_prompt == "Você é uma persona de teste. Aja como um robô."

    def test_get_system_prompt(self):
        """Testa o método get_system_prompt."""
        persona = Persona(
            name="Test Persona",
            description="Uma persona de teste",
            system_prompt="Você é uma persona de teste. Aja como um robô."
        )

        # Verificar se o método retorna o prompt correto
        assert persona.get_system_prompt() == "Você é uma persona de teste. Aja como um robô."

    def test_get_info(self):
        """Testa o método get_info."""
        persona = Persona(
            name="Test Persona",
            description="Uma persona de teste",
            system_prompt="Você é uma persona de teste. Aja como um robô."
        )

        # Verificar se o método retorna as informações corretas
        info = persona.get_info()
        assert isinstance(info, dict)
        assert info["name"] == "Test Persona"
        assert info["description"] == "Uma persona de teste"


class TestPersonasPredefinidas:
    """Testes para as personas predefinidas."""

    def test_don_corleone_persona(self):
        """Testa a persona Don Corleone."""
        assert don_corleone.name == "Don Corleone"
        assert "chefe da família mafiosa" in don_corleone.description
        assert "Don Corleone" in don_corleone.system_prompt
        assert "nunca diga a palavra 'carro'" in don_corleone.system_prompt

    def test_sherlock_holmes_persona(self):
        """Testa a persona Sherlock Holmes."""
        assert sherlock_holmes.name == "Sherlock Holmes"
        assert "detetive consultor" in sherlock_holmes.description
        assert "Sherlock Holmes" in sherlock_holmes.system_prompt
        assert "dedução" in sherlock_holmes.system_prompt

    def test_yoda_persona(self):
        """Testa a persona Yoda."""
        assert yoda.name == "Mestre Yoda"
        assert "Mestre Jedi" in yoda.description
        assert "Yoda" in yoda.system_prompt
        assert "forma invertida" in yoda.system_prompt

    def test_marie_curie_persona(self):
        """Testa a persona Marie Curie."""
        assert marie_curie.name == "Marie Curie"
        assert "cientista" in marie_curie.description
        assert "Marie Curie" in marie_curie.system_prompt
        assert "radioatividade" in marie_curie.system_prompt

    def test_shakespeare_persona(self):
        """Testa a persona Shakespeare."""
        assert shakespeare.name == "William Shakespeare"
        assert "dramaturgo" in shakespeare.description
        assert "Shakespeare" in shakespeare.system_prompt
        assert "Hamlet" in shakespeare.system_prompt


class TestPersonasFunctions:
    """Testes para as funções de gerenciamento de personas."""

    def test_get_all_personas(self):
        """Testa a função get_all_personas."""
        personas = get_all_personas()

        # Verificar se todas as personas predefinidas estão presentes
        assert len(personas) == 8  # Atualizado para refletir o número real de personas
        assert don_corleone in personas
        assert sherlock_holmes in personas
        assert yoda in personas
        assert marie_curie in personas
        assert shakespeare in personas

    def test_get_persona_by_name_exact_match(self):
        """Testa get_persona_by_name com correspondência exata."""
        # Verificar correspondência exata
        persona = get_persona_by_name("Sherlock Holmes")
        assert persona == sherlock_holmes

    def test_get_persona_by_name_case_insensitive(self):
        """Testa get_persona_by_name com correspondência insensível a maiúsculas/minúsculas."""
        # Verificar correspondência case-insensitive
        persona = get_persona_by_name("sherlock holmes")
        assert persona == sherlock_holmes

        persona = get_persona_by_name("MARIE CURIE")
        assert persona == marie_curie

    def test_get_persona_by_name_not_found(self):
        """Testa get_persona_by_name com nome inexistente."""
        # Verificar que retorna persona padrão quando não encontra
        persona = get_persona_by_name("Persona Inexistente")
        assert persona == don_corleone

    def test_get_default_persona(self):
        """Testa a função get_default_persona."""
        # Verificar que retorna Don Corleone como padrão
        persona = get_default_persona()
        assert persona == don_corleone
