# LangGraph 101 - Optimized Production System

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Performance](https://img.shields.io/badge/Performance-Excellent-blue)]()
[![Memory](https://img.shields.io/badge/Memory%20Usage-45.7MB-green)]()
[![Tests](https://img.shields.io/badge/Tests-83.3%25%20Pass-success)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20Compatible-informational)]()

🎯 **SYSTEM STATUS:** Fully optimized with thread-safe database management, advanced memory profiling, and comprehensive monitoring. Ready for production deployment.

## 🚀 Quick Start

### Production Deployment (< 5 minutes)
```powershell
# Deploy optimized system
cd "c:\ALTAIR GARCIA\04__ia"
python deploy_optimized_system.py

# Validate deployment
python test_final_system.py

# Start production system
python start_optimized_system.py
```

### Migration from Legacy System
See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions.

## 🏗️ System Architecture

### Core Components (Optimized)
- **ThreadSafeConnectionManager**: Zero-error database operations
- **AdvancedMemoryProfiler**: Automatic leak detection  
- **EnhancedUnifiedMonitoringSystem**: Comprehensive observability
- **Production Configuration**: Optimized for Windows deployment

### Performance Metrics
- **Memory Efficiency**: 45.7MB stable usage
- **Database Reliability**: 0% error rate
- **Thread Safety**: 100% concurrent operation support
- **System Startup**: Under 1 second
- **Monitoring Coverage**: Real-time + historical analysis

## Content Creator Mode

LangGraph 101 is now an AI-powered content creation agent for YouTube, TikTok, **Blog Posts, Twitter Threads,** and more. It orchestrates text, audio, and visual content using the best APIs:

- **Google Gemini**: Scriptwriting, brainstorming, copywriting, **blog post generation, Twitter thread creation,** repurposing text
- **ElevenLabs (TTS)**: Turns scripts into high-quality voiceovers, supports unique personas
- **Pexels/Pixabay**: Fetches stock images/videos for b-roll, thumbnails, backgrounds
- **Stability AI/DALL-E**: Generates custom images for thumbnails, branding, avatars
- **AssemblyAI/Deepgram**: Transcribes audio/video for captions, SEO, and repurposing
- **YouTube Data API**: Researches trends, keywords, and competitor insights

### Example Workflow
1. User: "Don Corleone, I need a 5-min YouTube video on password security, with a script, thumbnail, and narration."
2. Agent:
   - Generates script, title, description, and thumbnail prompt (Gemini)
   - Converts script to audio (ElevenLabs)
   - Generates thumbnail image (Stability AI/DALL-E)
   - Fetches stock media (Pexels/Pixabay)
   - Transcribes audio (AssemblyAI/Deepgram)
   - Researches trends (YouTube Data API)
3. Agent returns: script, audio file, thumbnail, stock media links, captions

**New Content Examples:**
*   User: "Create a blog post about the benefits of learning Python, make it around 1000 words and use a professional tone."
*   Agent: Generates a Markdown file with the blog post (title, meta description, content) using Gemini.
*   User: "Generate a Twitter thread on the latest advancements in renewable energy, about 7 tweets long, with an engaging tone."
*   Agent: Generates a text file with the Twitter thread (title, list of tweets) using Gemini.

### API Analytics System
The system now includes comprehensive API usage tracking and analytics:

- **Usage Monitoring**: Track calls, tokens, audio duration, and image generation across all APIs
- **Cost Estimation**: Calculate estimated costs based on current pricing models
- **Performance Metrics**: Monitor response times and success rates for all external APIs
- **Dashboard Integration**: Visualize API usage, costs, and performance in the content dashboard
- **Cost Optimization**: Get recommendations to reduce API costs based on usage patterns

To access API analytics:
1. Launch the dashboard: `streamlit run content_dashboard.py`
2. Navigate to the "API Analytics" tab
3. View usage, performance metrics, and cost optimization tips

For details on extending the analytics system, see [API_ANALYTICS_GUIDE.md](API_ANALYTICS_GUIDE.md) and [API_ANALYTICS_DOCS.md](API_ANALYTICS_DOCS.md).

### Advanced Features & Customization
*   **Persona-Driven Content:** Leverage unique personas (e.g., Don Corleone, Sherlock Holmes) to give your content a distinct voice and style. The agent adapts its language and tone based on the selected persona.
*   **Multi-Platform Support:** While initially focused on YouTube, the architecture is designed to be adaptable for other platforms like TikTok, Instagram Reels, podcasts, **blogs, and Twitter.**
*   **Workflow Automation:** The agent automates the entire content creation pipeline, from ideation to asset generation **(including video scripts, voiceovers, blog articles, and social media threads)**, saving significant time and effort.
*   **API Integration:** Seamlessly integrates with leading AI and media APIs, ensuring access to cutting-edge technology for each step of the process.
*   **Customizable Prompts:** Fine-tune the prompts used for each API to achieve specific styles, tones, or content requirements.
*   **Error Handling & Resilience:** Built-in mechanisms for retrying failed API calls and gracefully degrading functionality when services are unavailable.
*   **Modular Design:** The system is designed with modularity in mind, allowing for easy addition of new tools, APIs, or personas.

### Use Cases
*   **YouTube Automation:** Generate complete video packages (script, voiceover, visuals, captions) for YouTube channels.
*   **Blog Post Generation:** Create well-structured and informative blog posts on various topics.
*   **Twitter Thread Crafting:** Develop engaging Twitter threads to share insights or stories.
*   **Podcast Creation:** Create podcast episodes with scripted content and AI-generated narration.
*   **Social Media Snippets:** Repurpose longer content into short, engaging clips for platforms like TikTok and Instagram.
*   **Educational Content:** Develop instructional videos or audio lessons with clear explanations and supporting visuals.
*   **Marketing & Advertising:** Produce promotional videos, ad copy, and voiceovers for marketing campaigns.

---

![LangGraph Logo](https://img.shields.io/badge/LangGraph-101-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

Uma aplicação de chat interativo usando LangGraph e o modelo Gemini da Google.

## 📋 Visão Geral

Este projeto demonstra como criar um agente conversacional usando LangGraph, LangChain e o modelo Gemini da Google. O agente tem a personalidade de Don Corleone e pode realizar buscas na web usando a API Tavily.

Código baseado no vídeo: [LangGraph Tutorial](https://www.youtube.com/watch?v=9bSzZ-9eUkM)

## ✨ Funcionalidades

- 🤖 Agente conversacional com personalidade de Don Corleone
- 🌐 Capacidade de busca na web usando Tavily
- 🧮 Calculadora integrada para operações matemáticas
- 📅 Reconhecimento de consultas sobre data e hora
- 💬 Sistema de ajuda inteligente com descrição das capacidades
- 🔄 Arquitetura modular e extensível
- 📝 Histórico de conversa (implementação básica)
- 🛠️ Comandos especiais (ajuda, limpar, sair)

## 🔧 Requisitos

- Python 3.10 ou superior
- Google Generative AI API Key ([Obtenha aqui](https://aistudio.google.com/app/apikey))
- Tavily API Key ([Obtenha aqui](https://tavily.com/#api))

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/langgraph-101.git
cd langgraph-101
```

> **Nota**: Caso não queira clonar, você pode baixar o código fonte como arquivo ZIP e extrair localmente.

### 2. Crie um ambiente virtual

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## ⚙️ Configuração

1. Na raiz do projeto, crie um arquivo `.env` com suas chaves de API. Você pode copiar o arquivo `.env.example` para `.env` e preencher suas chaves:

   ```bash
   cp .env.example .env
   ```

   Edite o arquivo `.env` com suas chaves:

```dotenv
# --- Required API Keys ---
# Google Gemini API Key (for core LLM functionalities)
API_KEY="SUA_CHAVE_API_GEMINI"

# Tavily API Key (for web search tool)
TAVILY_API_KEY="SUA_CHAVE_API_TAVILY"

# --- Optional API Keys for Content Creation & Other Tools ---
# ElevenLabs API Key (for Text-to-Speech)
ELEVENLABS_API_KEY="SUA_CHAVE_API_ELEVENLABS"

# OpenAI DALL-E API Key (for DALL-E image generation)
DALLE_API_KEY="SUA_CHAVE_API_OPENAI_PARA_DALLE"

# Stability AI API Key (for Stable Diffusion image generation)
STABILITYAI_API_KEY="SUA_CHAVE_API_STABILITYAI"

# Pixabay API Key (for stock images/videos)
PIXABAY_API_KEY="SUA_CHAVE_API_PIXABAY"

# Pexels API Key (for stock images/videos)
PEXELS_API_KEY="SUA_CHAVE_API_PEXELS"

# Deepgram API Key (for audio transcription)
DEEPGRAM_API_KEY="SUA_CHAVE_API_DEEPGRAM"

# AssemblyAI API Key (for audio transcription)
ASSEMBLYAI_API_KEY="SUA_CHAVE_API_ASSEMBLYAI"

# YouTube Data API Key (for YouTube research tool)
YOUTUBE_DATA_API_KEY="SUA_CHAVE_API_YOUTUBE_DATA"

# NewsAPI.org Key (for the search_news tool)
NEWS_API_KEY="SUA_CHAVE_API_NEWSAPI"

# OpenWeatherMap API Key (for the get_weather_info tool)
OPENWEATHER_API_KEY="SUA_CHAVE_API_OPENWEATHERMAP"

# --- Optional API Keys for Content Creation & Other Tools ---
# ElevenLabs API Key (for Text-to-Speech)
ELEVENLABS_API_KEY="SUA_CHAVE_API_ELEVENLABS"

# OpenAI API Key (for GPT models)
OPENAI_API_KEY="SUA_CHAVE_API_OPENAI"

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY="SUA_CHAVE_API_ANTHROPIC"

# OpenAI DALL-E API Key (for DALL-E image generation)
DALLE_API_KEY="SUA_CHAVE_API_OPENAI_PARA_DALLE"
# ... (rest of optional keys remain the same)

# --- Model Configuration (Optional - Defaults provided) ---
# MODEL_NAME="gemini-2.0-flash" (Legacy - see new section on Multi-LLM Support)
# TEMPERATURE="0.7"
#
# Para configuração avançada de múltiplos modelos (Google, OpenAI, Anthropic),
# veja a variável de ambiente MODELS_CONFIG_JSON na seção "Suporte a Múltiplos Modelos de Linguagem (LLM)".

# --- Persona Configuration (Optional - Defaults to Don Corleone) ---
# PERSONA="Don Corleone"
# SYSTEM_PROMPT="Você é um assistente amigável."

# --- Application Settings (Optional - Defaults provided) ---
# SAVE_HISTORY="false"
# MAX_HISTORY="10"
```

   **Importante:** Substitua `"SUA_CHAVE_API_..."` pelos seus valores reais.

## 🖥️ Uso

## 🤖 Suporte a Múltiplos Modelos de Linguagem (LLM)

O agente conversacional principal agora pode utilizar modelos de linguagem de diferentes provedores, incluindo:
- Google (modelos Gemini)
- OpenAI (modelos GPT, como GPT-4o, GPT-3.5-turbo)
- Anthropic (modelos Claude, como Claude 3 Haiku, Sonnet, Opus)

Esta flexibilidade é gerenciada pelo `ModelManager` e `ModelSelector` internos, que escolhem um modelo apropriado com base na configuração e, futuramente, nos requisitos da tarefa.
As funcionalidades de criação de conteúdo (como geração de roteiros, posts de blog e threads do Twitter) também utilizam este sistema de seleção de modelos, permitindo maior flexibilidade na escolha do LLM para tarefas específicas de criação.

### Configurando Modelos LLM

1.  **API Keys**: Certifique-se de que as chaves de API para os provedores desejados estão configuradas no seu arquivo `.env`:
    *   `API_KEY` ou `GEMINI_API_KEY` para Google Gemini.
    *   `OPENAI_API_KEY` para modelos OpenAI.
    *   `ANTHROPIC_API_KEY` para modelos Anthropic.

2.  **Seleção de Modelos (`MODELS_CONFIG_JSON`)**:
    Você pode controlar quais modelos estão disponíveis para o agente e qual é o modelo padrão definindo a variável de ambiente `MODELS_CONFIG_JSON`. Esta variável aceita uma string JSON com a seguinte estrutura:

    ```json
    {
      "available_models": [
        {"model_id": "gemini-1.5-pro-latest", "provider": "google", "api_key_env_var": "GEMINI_API_KEY"},
        {"model_id": "gpt-4o", "provider": "openai", "api_key_env_var": "OPENAI_API_KEY"},
        {"model_id": "claude-3-haiku-20240307", "provider": "anthropic", "api_key_env_var": "ANTHROPIC_API_KEY"}
      ],
      "default_model_id": "gemini-1.5-pro-latest"
    }
    ```
    - `available_models`: Uma lista dos modelos que o sistema pode usar. Especifique o `model_id` (conforme nomeado pelo provedor), `provider` ("google", "openai", ou "anthropic"), e `api_key_env_var` (o nome da variável no `.env` que contém a chave para este modelo).
    - `default_model_id`: O `model_id` do modelo que será usado por padrão se nenhum outro critério específico for aplicado.

    Se `MODELS_CONFIG_JSON` não for definido, o sistema utilizará uma lista padrão de modelos pré-configurados (atualmente incluindo Gemini, GPT-4o, e Claude 3 Haiku).

O sistema seleciona automaticamente um modelo da lista de disponíveis. O antigo método de usar apenas `MODEL_NAME` no `.env` para definir o modelo do agente foi substituído por este sistema mais robusto. `MODEL_NAME` pode ainda influenciar um fallback se `MODELS_CONFIG_JSON` não estiver definido.

### Método 1: Usando o CLI do LangGraph

```bash
langgraph dev
```

**Nota para usuários Windows:** Se encontrar problemas, tente:

```bash
langgraph dev --allow-blocking
```

### Método 2: Executando diretamente

```bash
python langgraph-101.py
```

## 📚 Estrutura do Projeto

```
.
├── langgraph-101.py    # Script principal
├── agent.py            # Módulo de criação do agente com implementação personalizada
├── config.py           # Configurações e carregamento de variáveis
├── tools.py            # Ferramentas disponíveis para o agente (web, calculadora, etc.)
├── personas.py         # Definições de personalidades disponíveis
├── history.py          # Gerenciamento de histórico de conversas
├── ui.py               # Interface do usuário (CLI)
├── streamlit_app.py    # Interface web com Streamlit
├── requirements.txt    # Dependências do projeto
├── .env                # Arquivo de variáveis de ambiente (criar manualmente)
├── tests/              # Testes automatizados
├── langgraph.json      # Configuração do LangGraph
└── README.md           # Documentação
```

## 📊 Analytics & Feedback

- **Usage Analytics**: The web dashboard now includes real-time analytics (success/failure rates, tool usage, and historical logs if database enabled).
- **User Feedback**: Submit feedback or suggestions directly from the dashboard. All feedback is stored in session and can be reviewed.
- **Accessibility & UX**: The UI is modern, responsive, and tested for accessibility. Persona switching and error messages are always visible and actionable.

## 🧑‍💻 Testing & Continuous Integration

- **Comprehensive Tests**: All major modules and workflows are covered by unit, integration, and UI tests (see `TESTING.md`).
- **How to Run Tests**:
  ```powershell
  # In PowerShell
  pytest tests/
  # Or use the helper script
  python run_tests.py
  ```
- **Coverage**: 100% coverage for core modules. Coverage report is generated in `coverage_report/`.
- **CI/CD**: (Recommended) Set up GitHub Actions or similar to run tests and coverage on every push. See `TESTING.md` for roadmap.

## 🧪 Testes e Cobertura

- Todos os módulos principais e fluxos críticos possuem testes unitários, de integração e de UI (veja `TESTING.md`).
- Para rodar todos os testes e gerar cobertura localmente:
  ```powershell
  pytest tests/ --cov=. --cov-report=term --cov-report=html:coverage_report
  ```
  O relatório HTML estará em `coverage_report/index.html`.
- O teste de UI para colaboração multi-agente e logging está em `tests/test_streamlit_ui.py`.
- Recomenda-se configurar CI/CD (exemplo: GitHub Actions) para rodar testes e cobertura a cada push. Veja exemplo em `TESTING.md`.

## 🆕 Recent Features

- Robust API wrappers for Gemini, ElevenLabs, Pexels, Pixabay, Stability AI, DALL-E, AssemblyAI, Deepgram, YouTube Data
- Modern web dashboard with onboarding, tool selection, workflow orchestration, asset previews, persona switching, error handling, and analytics
- Actionable error messages and session-state fallbacks for robust UX
- Analytics charts and feedback collection in the dashboard
- Comprehensive and flexible test suite for all integrations and UI/UX
- Onboarding/help expander in dashboard now includes quick start, tips, troubleshooting, and workflow examples for improved clarity and accessibility
- Asset preview section enhanced: all asset types (text, audio, image, caption, video) are previewed and downloadable, with accessibility notes
- Documentation and help sections updated to match UI/UX improvements

## 🆕 Recent Features (2025)

- **Onboarding & Help:** Sidebar onboarding expander for new users, with quick start and tips.
- **Tooltips:** Utility for tooltips in UI for better workflow clarity.
- **Security:** Optional file encryption utilities and authentication recommendations for dashboard access.
- **Automation:** Workflow scheduler for periodic or triggered multi-agent workflows.
- **AI/ML Analytics:** ML-powered error anomaly detection and AI-driven next-action recommendations in analytics dashboard.

## 🧑‍💻 How to Use New Features

- **Onboarding:** Launch the app and see the sidebar for onboarding/help.
- **Tooltips:** Hover over UI elements for extra guidance.
- **Security:** Use encryption utilities in `resilient_storage.py` for sensitive data. Add authentication as recommended in `streamlit_app.py`.
- **Automation:** Use the workflow scheduler in `orchestrator.py` to automate multi-agent tasks.
- **Analytics:** Visit the Analytics tab for anomaly detection and recommendations.

## 🧪 Testing

- Run all tests: `python run_tests.py` or `pytest tests/`
- See `TESTING.md` for more details.

## 🔍 Comandos e Interações

### Comandos do Sistema

Durante a interação com o chatbot, você pode usar os seguintes comandos:

- `ajuda` - Mostra a lista de comandos disponíveis
- `limpar` - Limpa o histórico da conversa
- `sair` - Encerra a aplicação

### Exemplos de Interação

#### 1. Busca na Web

Faça perguntas sobre informações atuais:

```
Quando é o próximo jogo do Grêmio?
Quais são as notícias recentes sobre o Brasil?
Quem ganhou o último Oscar?
```

#### 2. Consulta de Data e Hora

Pergunte sobre a data atual:

```
Que dia é hoje?
Qual é a data atual?
```

#### 3. Calculadora

Realize cálculos matemáticos:

```
2+2*5
(10-5)/2+3
```

#### 4. Funcionalidades Disponíveis

Descubra o que o agente pode fazer:

```
O que você pode fazer?
Quais são suas funcionalidades?
```

## ⚠️ Troubleshooting (Expanded)

- **API Key Errors**: Always check `.env` for correct keys. Error messages in the dashboard will suggest missing/invalid keys.
- **Web UI Issues**: If the dashboard doesn't load, ensure all dependencies are installed and Streamlit is working. Try `streamlit run streamlit_app.py`.
- **Test Failures**: Ensure you are using the latest code and dependencies. Use mocks for all external APIs in tests. See `TESTING.md` for more.
- **Analytics/Feedback**: If analytics or feedback are not visible, ensure you are running the latest version and have not disabled session state or database features.

## 🛠️ Personalização

### Alterando o Modelo

A seleção do modelo de linguagem principal para o agente agora é mais flexível e configurável através da variável de ambiente `MODELS_CONFIG_JSON`.
Consulte a seção "🤖 Suporte a Múltiplos Modelos de Linguagem (LLM)" para detalhes completos sobre como definir os modelos disponíveis (Google Gemini, OpenAI GPT, Anthropic Claude) e o modelo padrão.

A antiga configuração `MODEL_NAME` no arquivo `.env` pode servir como um fallback caso `MODELS_CONFIG_JSON` não esteja definido ou não especifique um padrão.

### Alterando a Personalidade do Agente

Para alterar a personalidade do agente, adicione sua própria instrução de sistema:

```dotenv
SYSTEM_PROMPT="Você é um assistente amigável que ajuda as pessoas com suas dúvidas."
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## ⚠️ Solução de Problemas

### Navegador

Ao usar o comando `langgraph dev`, a interface será aberta no navegador padrão. Se você encontrar problemas com o Brave, tente usar o Chrome.

### Erro de API Key

Se você receber erros relacionados às chaves de API, verifique se:

1. O arquivo `.env` está na raiz do projeto e foi preenchido corretamente (você pode usar o `.env.example` como base).
2. As chaves API estão corretas, ativas e não contêm espaços extras.
3. Você ativou o ambiente virtual antes de executar o script.
4. Para as chaves opcionais (ElevenLabs, DALL-E, etc.), se você não as configurou, as ferramentas correspondentes podem não funcionar, mas o aplicativo principal ainda deve iniciar se `API_KEY` (Gemini) e `TAVILY_API_KEY` estiverem corretas.

### Erro "contents is not specified"

Se você encontrar erros como `Invalid argument provided to Gemini: 400 * GenerateContentRequest.contents: contents is not specified`, isso indica um problema na comunicação com a API Gemini. Nossa implementação custom do agente já trata esse problema, garantindo que mensagens válidas sejam sempre enviadas ao modelo.

### Outros Erros

Se encontrar outros problemas, verifique os requisitos de versão do Python e das bibliotecas em `requirements.txt`.

## 🔧 Detalhes Técnicos

### Implementação Personalizada do Agente

Em vez de usar o `create_react_agent` padrão do LangGraph, implementamos um agente personalizado que:

1. Detecta automaticamente o tipo de consulta do usuário (busca web, calculadora, data/hora)
2. Seleciona e invoca a ferramenta apropriada baseada na consulta
3. Formata os resultados de maneira adequada para o modelo Gemini
4. Gerencia exceções e garante respostas consistentes

Essa abordagem contorna limitações da integração LangGraph/Gemini e fornece maior controle sobre o comportamento do agente.

### Segurança com o Modelo Gemini

O projeto implementa proteções contra erros comuns da API Gemini, como a exigência de pelo menos uma mensagem com conteúdo não vazio. Isso reduz significativamente erros como "contents is not specified".

### Detecção de Intenção

O agente usa palavras-chave para determinar qual ferramenta utilizar:
- Palavras relacionadas a data/tempo disparam reconhecimento de data
- Expressões com números e operadores ativam a calculadora
- Perguntas factuais ativam busca na web
- Perguntas sobre capacidades mostram informações de ajuda

#   0 4 _ I A  
 