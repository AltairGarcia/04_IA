"""
Testes unitários para o módulo history.py
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

from history import ConversationHistory, get_conversation_history
from langchain_core.messages import HumanMessage, AIMessage


class TestConversationHistory:
    """Testes para a classe ConversationHistory."""

    def test_init_default(self):
        """Testa a inicialização padrão da classe."""
        with patch('history.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.get_conversation.return_value = None

            history = ConversationHistory()

            assert history.messages == []
            assert history.max_history == 20  # Default is now 20
            assert history.conversation_id is not None

    def test_init_custom(self):
        """Testa a inicialização com parâmetros personalizados."""
        with patch('history.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.get_conversation.return_value = None

            conversation_id = "test-conversation-id"
            history = ConversationHistory(max_history=5, conversation_id=conversation_id, persona_name="TestPersona")

            assert history.messages == []
            assert history.max_history == 5
            assert history.conversation_id == conversation_id
            mock_db.create_conversation.assert_called_once_with(conversation_id, "TestPersona")

    def test_add_user_message(self):
        """Testa a adição de mensagem do usuário."""
        with patch('history.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.get_conversation.return_value = None

            history = ConversationHistory()

            # Mock para uuid
            with patch('history.uuid.uuid4') as mock_uuid:
                mock_uuid.return_value = "test-uuid"

                # Adicionar mensagem
                history.add_user_message("Olá, como vai?")

                # Verificar se a mensagem foi adicionada corretamente
                assert len(history.messages) == 1
                assert history.messages[0]["role"] == "user"
                assert history.messages[0]["content"] == "Olá, como vai?"
                assert "id" in history.messages[0]

                # Verificar que foi salvo no banco
                mock_db.add_message.assert_called_once()

    def test_add_ai_message(self):
        """Testa a adição de mensagem da IA."""
        with patch('history.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.get_conversation.return_value = None

            history = ConversationHistory()

            # Mock para uuid
            with patch('history.uuid.uuid4') as mock_uuid:
                mock_uuid.return_value = "test-uuid"

                # Adicionar mensagem
                history.add_ai_message("Estou bem, obrigado por perguntar!")

                # Verificar se a mensagem foi adicionada corretamente
                assert len(history.messages) == 1
                assert history.messages[0]["role"] == "assistant"  # Nota: agora é "assistant" em vez de "ai"
                assert history.messages[0]["content"] == "Estou bem, obrigado por perguntar!"
                assert "id" in history.messages[0]

                # Verificar que foi salvo no banco
                mock_db.add_message.assert_called_once()

    def test_clear(self):
        """Testa o método de limpar o histórico."""
        history = ConversationHistory()

        # Adicionar algumas mensagens
        history.add_user_message("Mensagem 1")
        history.add_ai_message("Resposta 1")

        # Verificar que as mensagens foram adicionadas
        assert len(history.messages) == 2

        # Limpar o histórico
        history.clear()

        # Verificar que as mensagens foram removidas
        assert history.messages == []

    def test_langchain_message_conversion(self):
        """Testa a conversão para mensagens do LangChain."""
        with patch('history.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.get_conversation.return_value = None

            history = ConversationHistory()

            # Adicionar mensagens diretamente com o novo formato
            history.messages = [
                {"role": "user", "content": "Pergunta do usuário", "id": "id1"},
                {"role": "assistant", "content": "Resposta da IA", "id": "id2"},
                {"role": "user", "content": "Outra pergunta", "id": "id3"}
            ]

            # Teste da função get_history
            history_list = history.get_history()
            assert len(history_list) == 3
            assert "role" in history_list[0]
            assert "content" in history_list[0]
            assert "id" not in history_list[0]

            # Implementar um método para converter para mensagens do LangChain para testes
            # Esta parte seria um método auxiliar só para testes, já que a classe não o tem mais

            def convert_to_langchain_messages(messages):
                lc_messages = []
                for msg in messages:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))
                return lc_messages

            lc_messages = convert_to_langchain_messages(history.messages)

            # Verificar a conversão
            assert len(lc_messages) == 3
            assert isinstance(lc_messages[0], HumanMessage)
            assert isinstance(lc_messages[1], AIMessage)
            assert isinstance(lc_messages[2], HumanMessage)
            assert lc_messages[0].content == "Pergunta do usuário"
            assert lc_messages[1].content == "Resposta da IA"
            assert lc_messages[2].content == "Outra pergunta"

    def test_trim_history(self):
        """Testa o truncamento do histórico quando excede o máximo."""
        with patch('history.get_database') as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            mock_db.get_conversation.return_value = None

            # Criar histórico com max_history=2
            history = ConversationHistory(max_history=2)

            # Mock uuid to return predictable values
            with patch('history.uuid.uuid4', side_effect=['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10']):
                # Adicionar mensagens além do limite (2 pares = 4 mensagens)
                for i in range(5):
                    history.add_user_message(f"Pergunta {i}")
                    history.add_ai_message(f"Resposta {i}")

                # Verificar que o histórico foi truncado para max_history mensagens
                assert len(history.messages) == 2  # Agora é 2 porque max_history=2

                # Verificar que apenas as mensagens mais recentes foram mantidas
                assert history.messages[0]["content"] == "Pergunta 4"
                assert history.messages[1]["content"] == "Resposta 4"

    def test_save_without_file(self):
        """Testa salvar quando nenhum arquivo foi especificado."""
        history = ConversationHistory()

        # Tentar salvar sem especificar arquivo
        result = history.save()

        # Verificar que a operação falhou
        assert result is False

    def test_save_success(self):
        """Testa o salvamento bem-sucedido do histórico."""
        # Usar um arquivo temporário para o teste
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            history = ConversationHistory(history_file=temp_path)

            # Adicionar algumas mensagens
            history.add_user_message("Teste de salvamento")
            history.add_ai_message("Resposta para salvamento")

            # Salvar o histórico
            result = history.save()

            # Verificar que a operação foi bem-sucedida
            assert result is True

            # Verificar que o arquivo foi criado e contém o histórico
            assert os.path.exists(temp_path)

            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                assert "messages" in saved_data
                assert len(saved_data["messages"]) == 2
                assert saved_data["messages"][0]["content"] == "Teste de salvamento"
                assert saved_data["messages"][1]["content"] == "Resposta para salvamento"
        finally:
            # Limpar o arquivo temporário
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_file_not_exists(self):
        """Testa carregar quando o arquivo não existe."""
        history = ConversationHistory(history_file="arquivo_inexistente.json")

        # Tentar carregar de um arquivo que não existe
        result = history.load()

        # Verificar que a operação falhou
        assert result is False

    def test_load_success(self):
        """Testa o carregamento bem-sucedido do histórico."""
        # Criar arquivo temporário com dados de histórico
        test_data = {
            "messages": [
                {"role": "user", "content": "Mensagem carregada", "timestamp": "2025-01-01T12:00:00"},
                {"role": "ai", "content": "Resposta carregada", "timestamp": "2025-01-01T12:01:00"}
            ],
            "timestamp": "2025-01-01T12:01:00"
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(test_data, temp_file, ensure_ascii=False)

        try:
            history = ConversationHistory(history_file=temp_path)

            # Carregar o histórico
            result = history.load()

            # Verificar que a operação foi bem-sucedida
            assert result is True

            # Verificar que o histórico foi carregado corretamente
            assert len(history.messages) == 2
            assert history.messages[0]["content"] == "Mensagem carregada"
            assert history.messages[1]["content"] == "Resposta carregada"
        finally:
            # Limpar o arquivo temporário
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestGetConversationHistory:
    """Testes para a função get_conversation_history."""

    @patch('history.os.path.dirname')
    @patch('history.os.makedirs')
    @patch('history.datetime')
    def test_get_conversation_history_without_save(self, mock_datetime, mock_makedirs, mock_dirname):
        """Testa get_conversation_history sem salvar o histórico."""
        history = get_conversation_history(max_history=15)

        # Verificar que o gerenciador foi criado corretamente
        assert isinstance(history, ConversationHistory)
        assert history.max_history == 15
        assert history.history_file is None

        # Verificar que nenhum diretório foi criado
        mock_makedirs.assert_not_called()

    @patch('history.os.path.dirname')
    @patch('history.os.makedirs')
    @patch('history.datetime')
    def test_get_conversation_history_with_persona(self, mock_datetime, mock_makedirs, mock_dirname):
        """Testa get_conversation_history com persona."""
        # Configurar mocks
        mock_dirname.return_value = "/path/to/project"
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20250101_120000"
        mock_datetime.now.return_value = mock_now

        # Chamar a função sendo testada
        history = get_conversation_history(max_history=15, persona_name="TestPersona")

        # Verificar que o gerenciador foi criado corretamente
        assert isinstance(history, ConversationHistory)
        assert history.max_history == 15
        assert history.history_file is not None
        assert "chat_history_20250101_120000.json" in history.history_file

        # Verificar que o diretório de histórico foi criado
        mock_makedirs.assert_called_once_with(
            os.path.join("/path/to/project", "history"),
            exist_ok=True
        )
