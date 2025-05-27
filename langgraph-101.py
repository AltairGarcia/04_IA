"""
LangGraph 101 - Main Application

This is the main entry point for the LangGraph 101 application.
It creates a chat interface for interacting with a LangGraph agent.
"""

import sys # Added back sys
import traceback # Added back traceback
import os
from datetime import datetime  # Added import for datetime
import json  # Added for parsing Gemini JSON output

# Import modules from our project
from config import load_config, ConfigError, get_system_prompt, get_available_personas
from tools import get_tools, GeminiAPI, ElevenLabsTTS, PexelsAPI, PixabayAPI, StabilityAIAPI, DalleAPI, AssemblyAIAPI, DeepgramAPI, YouTubeDataAPI  # Added ElevenLabsTTS, PexelsAPI, PixabayAPI, StabilityAIAPI, DalleAPI, AssemblyAIAPI, DeepgramAPI, YouTubeDataAPI
from agent import create_agent
from content_creation import ContentCreator # Added
from history import get_history_manager
from memory_manager import get_memory_manager
from personas import get_persona_by_name
from export import export_conversation, get_export_formats
from email_sender import send_email, email_conversation
from ui import (
    print_welcome, print_help, print_error, print_success,
    print_agent_response, show_thinking_animation, get_user_input,
    clear_screen, print_colored, Colors
)


def print_personas_list():
    """Print a list of available personas."""
    personas = get_available_personas()

    print_colored("\n--- Personas Disponíveis ---", Colors.YELLOW, bold=True)
    for name, info in personas.items():
        print_colored(f"{name}", Colors.CYAN, bold=True)
        print_colored(f"  {info['description']}", Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)


def print_export_formats():
    """Print a list of available export formats."""
    formats = get_export_formats()

    print_colored("\n--- Formatos de Exportação Disponíveis ---", Colors.YELLOW, bold=True)
    for name, info in formats.items():
        print_colored(f"{name}", Colors.CYAN, bold=True)
        print_colored(f"  {info['description']}", Colors.WHITE)
    print_colored("\nUse 'exportar FORMAT' para exportar a conversa (ex: exportar html)", Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)


def update_help_menu():
    """Print updated help information including persona commands."""
    print_colored("\n--- Comandos Disponíveis ---", Colors.YELLOW, bold=True)
    print_colored("sair       - Encerra a conversa", Colors.WHITE)
    print_colored("ajuda      - Mostra esta mensagem de ajuda", Colors.WHITE)
    print_colored("limpar     - Limpa o histórico da conversa", Colors.WHITE)
    print_colored("cls/clear  - Limpa a tela", Colors.WHITE)
    print_colored("salvar     - Salva o histórico da conversa", Colors.WHITE)
    print_colored("personas   - Lista todas as personas disponíveis", Colors.WHITE)
    print_colored("persona X  - Muda para a persona X (ex: persona Yoda)", Colors.WHITE)

    # Memory commands
    print_colored("\n--- Comandos de Memória ---", Colors.YELLOW, bold=True)
    print_colored("memória    - Mostra as memórias mais importantes", Colors.WHITE)
    print_colored("esquece    - Limpa todas as memórias armazenadas", Colors.WHITE)
    print_colored("lembra X   - Adiciona manualmente um fato importante à memória", Colors.WHITE)

    # Export commands
    print_colored("\n--- Comandos de Exportação ---", Colors.YELLOW, bold=True)
    print_colored("exportar   - Lista os formatos de exportação disponíveis", Colors.WHITE)
    print_colored("exportar X - Exporta a conversa no formato X (ex: exportar html)", Colors.WHITE)
    print_colored("enviar     - Envia a conversa para um email", Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)


def print_cli_dashboard():
    """Print the CLI dashboard for tool selection and workflow orchestration."""
    print_colored("\n=== Content Creation Dashboard ===", Colors.YELLOW, bold=True)
    print_colored("Escolha uma opção:", Colors.CYAN)
    print_colored("1. Full Content Creation Workflow (automatize tudo)", Colors.WHITE)
    print_colored("2. Usar ferramentas individualmente", Colors.WHITE)
    print_colored("0. Sair", Colors.WHITE)
    print_colored("\n--- Ferramentas Disponíveis ---", Colors.YELLOW, bold=True)
    print_colored("------------------------\n", Colors.YELLOW)


def main():
    """Main application entry point."""
    try:
        # Load configuration
        config = load_config()

        # Define tool_list here so it's in scope for the dashboard and tool handling
        tool_list = [
            {
                'name': 'Google Gemini',
                'desc': 'Gera scripts, títulos, descrições, ideias e mais.',
                'example': 'Roteiro de vídeo, título chamativo, descrição',
                'key': 'gemini',
                'api_key_name': 'GEMINI_API_KEY'
            },
            {
                'name': 'ElevenLabs TTS',
                'desc': 'Converte roteiro em narração de voz.',
                'example': 'Narração para vídeo sem rosto',
                'key': 'elevenlabs',
                'api_key_name': 'ELEVENLABS_API_KEY'
            },
            {
                'name': 'Pexels/Pixabay',
                'desc': 'Busca imagens/vídeos de banco de imagens.',
                'example': 'B-roll, fundos, thumbnails',
                'key': 'pexels_pixabay',
                'api_key_name': ['PEXELS_API_KEY', 'PIXABAY_API_KEY']
            },
            {
                'name': 'Stability AI/DALL-E',
                'desc': 'Gera imagens customizadas (thumbnails, arte).',
                'example': 'Thumbnail única, mascote do canal',
                'key': 'stability_dalle',
                'api_key_name': ['STABILITY_API_KEY', 'DALLE_API_KEY']
            },
            {
                'name': 'AssemblyAI/Deepgram',
                'desc': 'Transcreve áudio/vídeo.',
                'example': 'Legendas, SEO, blog post a partir de vídeo',
                'key': 'assembly_deepgram',
                'api_key_name': ['ASSEMBLYAI_API_KEY', 'DEEPGRAM_API_KEY']
            },
            {
                'name': 'YouTube Data API',
                'desc': 'Pesquisa tendências, palavras-chave, concorrentes.',
                'example': 'Encontrar tópicos em alta, ideias de palavras-chave',
                'key': 'youtube_data',
                'api_key_name': 'YOUTUBE_DATA_API_KEY'
            },
        ]

        # Get current persona from config
        current_persona = config["current_persona"]

        # Initialize conversation history
        history_manager = get_history_manager(max_history=config["max_history"])

        # Initialize memory manager
        memory_manager = get_memory_manager(max_items=50, extraction_enabled=True)

        # Get tools
        tools = get_tools()

        # Create agent with current persona
        agent = create_agent(config, tools)        # Initialize ContentCreator
        gemini_api_key = config.get('api_key')
        if not gemini_api_key:
            print("Error: Gemini API key not found in configuration. Content creation features will be limited.")
            content_creator_instance = None  # Or handle as per your app's logic
        else:
            # Create API keys dictionary for ContentCreator
            api_keys_dict = {
                "api_key": gemini_api_key,
                "model_name": config.get("model_name", "gemini-2.0-flash"),
                "temperature": config.get("temperature", 0.7),
                "elevenlabs_api_key": config.get("elevenlabs_api_key"),
                "dalle_api_key": config.get("dalle_api_key"),
                "stabilityai_api_key": config.get("stabilityai_api_key"),
                "pixabay_api_key": config.get("pixabay_api_key"),
                "pexels_api_key": config.get("pexels_api_key"),
                "deepgram_api_key": config.get("deepgram_api_key"),
                "assemblyai_api_key": config.get("assemblyai_api_key"),
                "youtube_data_api_key": config.get("youtube_data_api_key"),
                "news_api_key": config.get("news_api_key"),
                "openweather_api_key": config.get("openweather_api_key")
            }
            content_creator_instance = ContentCreator(api_keys=api_keys_dict)

        # Pass ContentCreator to the agent
        agent_executor = create_agent(
            config=config,
            tools=tools,
            content_creator=content_creator_instance # Added
        )

        # Print welcome message
        print_welcome()
        print_colored(f"Persona atual: {current_persona.name} - {current_persona.description}", Colors.CYAN)

        # Show dashboard at start
        while True:
            # Display the dashboard with tools
            print_colored("\n=== Content Creation Dashboard ===", Colors.YELLOW, bold=True)
            print_colored("Escolha uma opção:", Colors.CYAN)
            print_colored("1. Full Content Creation Workflow (automatize tudo)", Colors.WHITE)
            print_colored("2. Usar ferramentas individualmente", Colors.WHITE)
            print_colored("0. Sair", Colors.WHITE)
            print_colored("\n--- Ferramentas Disponíveis ---", Colors.YELLOW, bold=True)
            for idx, tool_item in enumerate(tool_list, 1):
                print_colored(f"{idx}. {tool_item['name']}", Colors.CYAN)
                print_colored(f"   Descrição: {tool_item['desc']}", Colors.WHITE)
                print_colored(f"   Exemplo: {tool_item['example']}", Colors.WHITE)
            print_colored("------------------------\n", Colors.YELLOW)

            print_colored("\nDigite o número da opção desejada ou pressione Enter para continuar para o chat:", Colors.YELLOW)
            dashboard_input = get_user_input()
            if dashboard_input.strip() == "1":
                print_colored("\n[Full Content Creation Workflow selecionado]", Colors.GREEN)
                print_colored("Descreva o conteúdo que você quer criar (ex: vídeo de 5 min sobre segurança de senhas):", Colors.YELLOW)
                workflow_prompt = get_user_input()
                if not workflow_prompt.strip():
                    print_error("Descrição do conteúdo não pode ser vazia.")
                    continue                # Initialize API clients (check keys)
                gemini_api_key = config.get('api_key')
                elevenlabs_api_key = config.get('elevenlabs_api_key')
                pexels_api_key = config.get('pexels_api_key')
                pixabay_api_key = config.get('pixabay_api_key')
                stability_api_key = config.get('stabilityai_api_key')
                dalle_api_key = config.get('dalle_api_key')
                assemblyai_api_key = config.get('assemblyai_api_key')
                deepgram_api_key = config.get('deepgram_api_key')
                youtube_api_key = config.get('youtube_data_api_key')

                if not gemini_api_key:
                    print_error("Gemini API key não configurada. O workflow não pode continuar sem ela.")
                    continue

                gemini_tool = GeminiAPI(api_key=gemini_api_key)
                assets = {} # To store generated content

                # 1. Gemini: Script, Title, Description, Thumbnail Prompt
                try:
                    gemini_prompt_structured = f"""
Para o tópico: '{workflow_prompt}', por favor gere o seguinte em formato JSON:
{{
  "title": "Um título chamativo e otimizado para SEO",
  "description": "Uma descrição concisa e envolvente (máximo 150 palavras), ideal para YouTube ou metadados de blog.",
  "script": "Um roteiro detalhado e bem estruturado. Se for para vídeo, indique cenas ou visuais sugeridos. Se for para blog, use parágrafos e subtítulos. (aprox. 300-700 palavras).",
  "thumbnail_prompt": "Um prompt altamente descritivo e criativo, otimizado para IA de geração de imagem (como DALL-E ou Stable Diffusion), para criar uma thumbnail visualmente atraente e relevante para o tópico."
}}
Certifique-se de que o JSON é válido.
"""
                    print_colored("Gerando detalhes do conteúdo com Gemini...", Colors.BLUE)
                    show_thinking_animation(2.0, "Gemini - Gerando Estrutura do Conteúdo")
                    raw_gemini_output = gemini_tool.generate_content(prompt=gemini_prompt_structured)

                    parsed_gemini_output = None
                    try:
                        clean_json_str = raw_gemini_output.strip()
                        if clean_json_str.startswith("```json"):
                            clean_json_str = clean_json_str[7:-3].strip()
                        elif clean_json_str.startswith("```"):
                            clean_json_str = clean_json_str[3:-3].strip()

                        parsed_gemini_output = json.loads(clean_json_str)
                        assets['title'] = parsed_gemini_output.get('title', f'Título para {workflow_prompt}')
                        assets['description'] = parsed_gemini_output.get('description', f'Descrição para {workflow_prompt}')
                        assets['script'] = parsed_gemini_output.get('script', f'Roteiro para {workflow_prompt}')
                        assets['thumbnail_prompt'] = parsed_gemini_output.get('thumbnail_prompt', f'Thumbnail para {workflow_prompt}')
                        print_success("Detalhes do conteúdo gerados e parseados por Gemini.")
                    except json.JSONDecodeError as e:
                        print_error(f"Erro ao decodificar JSON do Gemini: {e}. Usando saída bruta.")
                        print_colored("Saída bruta do Gemini:", Colors.YELLOW)
                        print_colored(raw_gemini_output, Colors.WHITE)
                        assets['gemini_raw_output'] = raw_gemini_output
                        assets['title'] = f"Título para {workflow_prompt} (fallback Gemini)"
                        assets['description'] = f"Descrição para {workflow_prompt} (fallback Gemini)"
                        assets['script'] = raw_gemini_output # Store the whole raw output as script
                        assets['thumbnail_prompt'] = f"Uma imagem atraente para um vídeo sobre {workflow_prompt}"
                except Exception as e:
                    print_error(f"Erro crítico na etapa Gemini: {e}")
                    assets['title'] = f"Título para {workflow_prompt} (Erro Gemini)"
                    assets['description'] = f"Descrição para {workflow_prompt} (Erro Gemini)"
                    assets['script'] = f"Roteiro para {workflow_prompt} (Erro Gemini)"
                    assets['thumbnail_prompt'] = f"Imagem para {workflow_prompt} (Erro Gemini)"

                # 2. ElevenLabs: Text-to-Speech
                if elevenlabs_api_key and assets.get('script') and "Erro Gemini" not in assets['script'] and assets['script'] != assets.get('gemini_raw_output', ''):
                    try:
                        print_colored("Gerando áudio com ElevenLabs...", Colors.BLUE)
                        show_thinking_animation(1.5, "ElevenLabs Processando")
                        tts_tool = ElevenLabsTTS(api_key=elevenlabs_api_key)
                        # Limit script length for TTS; 2500 chars is often a safe bet for many APIs
                        script_for_tts = assets['script']
                        if len(script_for_tts) > 2500:
                            print_colored("Roteiro longo, usando os primeiros 2500 caracteres para áudio.", Colors.YELLOW)
                            script_for_tts = script_for_tts[:2500]

                        audio_bytes = tts_tool.text_to_speech(text=script_for_tts)
                        audio_output_dir = os.path.join(os.path.dirname(__file__), "audio_output")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        audio_filename = os.path.join(audio_output_dir, f"workflow_audio_{timestamp}.mp3")
                        with open(audio_filename, 'wb') as f:
                            f.write(audio_bytes)
                        assets['audio_file'] = audio_filename
                        print_success(f"Áudio salvo em: {audio_filename}")
                    except Exception as e:
                        print_error(f"Erro na etapa ElevenLabs: {e}")
                        assets['audio_file'] = "Erro ao gerar áudio"
                elif not elevenlabs_api_key:
                    print_colored("ELEVENLABS_API_KEY não configurada. Pulando geração de áudio.", Colors.YELLOW)
                elif not assets.get('script') or "Erro Gemini" in assets.get('script', '') or assets.get('script') == assets.get('gemini_raw_output', ''):
                    print_colored("Roteiro não disponível ou inválido devido a erro anterior no Gemini. Pulando geração de áudio.", Colors.YELLOW)

                # 3. Stability AI/DALL-E: Image Generation
                if assets.get('thumbnail_prompt') and "Erro Gemini" not in assets['thumbnail_prompt']:
                    if stability_api_key:
                        try:
                            print_colored("Gerando thumbnail com Stability AI...", Colors.BLUE)
                            show_thinking_animation(1.5, "Stability AI Processando")
                            stability_tool = StabilityAIAPI(api_key=stability_api_key)
                            image_bytes = stability_tool.generate_image(prompt=assets['thumbnail_prompt'])
                            image_output_dir = os.path.join(os.path.dirname(__file__), "image_output")
                            os.makedirs(image_output_dir, exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            image_filename = os.path.join(image_output_dir, f"workflow_thumbnail_stability_{timestamp}.png")
                            with open(image_filename, 'wb') as f:
                                f.write(image_bytes)
                            assets['thumbnail_file_stability'] = image_filename
                            print_success(f"Thumbnail (Stability AI) salvo em: {image_filename}")
                        except Exception as e:
                            print_error(f"Erro na etapa Stability AI: {e}")
                            assets['thumbnail_file_stability'] = "Erro ao gerar thumbnail com Stability AI"
                    elif dalle_api_key: # Fallback to DALL-E
                        try:
                            print_colored("Stability AI não disponível/falhou. Tentando DALL-E...", Colors.YELLOW if stability_api_key else Colors.BLUE)
                            show_thinking_animation(1.5, "DALL-E Processando")
                            dalle_tool = DalleAPI(api_key=dalle_api_key)
                            response_data = dalle_tool.generate_image(prompt=assets['thumbnail_prompt'])
                            if response_data and response_data.get('data') and response_data['data'][0].get('url'):
                                assets['thumbnail_url_dalle'] = response_data['data'][0]['url']
                                print_success(f"Thumbnail (DALL-E) URL: {assets['thumbnail_url_dalle']}")
                            else:
                                assets['thumbnail_url_dalle'] = "Erro ao gerar thumbnail com DALL-E (sem URL)"
                                print_error(f"DALL-E não retornou URL da imagem. Resposta: {response_data}")
                        except Exception as e:
                            print_error(f"Erro na etapa DALL-E: {e}")
                            assets['thumbnail_url_dalle'] = "Erro ao gerar thumbnail com DALL-E"
                    else:
                        print_colored("Nenhuma API Key para Stability AI ou DALL-E configurada. Pulando geração de thumbnail.", Colors.YELLOW)
                elif not assets.get('thumbnail_prompt') or "Erro Gemini" in assets.get('thumbnail_prompt',''):
                    print_colored("Prompt para thumbnail não disponível ou inválido. Pulando geração de thumbnail.", Colors.YELLOW)

                # 4. Pexels/Pixabay: Stock Media Search
                stock_media_query = assets.get('title', workflow_prompt)
                if pexels_api_key:
                    try:
                        print_colored(f"Buscando mídia stock no Pexels para '{stock_media_query}'...", Colors.BLUE)
                        show_thinking_animation(1.0, "Pexels Buscando")
                        pexels_tool = PexelsAPI(api_key=pexels_api_key)
                        assets['pexels_images'] = pexels_tool.search_images(query=stock_media_query, per_page=3)
                        assets['pexels_videos'] = pexels_tool.search_videos(query=stock_media_query, per_page=2)
                        print_success("Busca no Pexels concluída.")
                    except Exception as e:
                        print_error(f"Erro na etapa Pexels: {e}")
                elif pixabay_api_key: # Fallback
                    try:
                        print_colored(f"Pexels não disponível/falhou. Buscando mídia stock no Pixabay para '{stock_media_query}'...", Colors.YELLOW if pexels_api_key else Colors.BLUE)
                        show_thinking_animation(1.0, "Pixabay Buscando")
                        pixabay_tool = PixabayAPI(api_key=pixabay_api_key)
                        assets['pixabay_images'] = pixabay_tool.search_images(query=stock_media_query, per_page=3)
                        assets['pixabay_videos'] = pixabay_tool.search_videos(query=stock_media_query, per_page=2)
                        print_success("Busca no Pixabay concluída.")
                    except Exception as e:
                        print_error(f"Erro na etapa Pixabay: {e}")
                else:
                    print_colored("Nenhuma API Key para Pexels ou Pixabay configurada. Pulando busca de mídia stock.", Colors.YELLOW)

                # 5. AssemblyAI/Deepgram: Audio Transcription
                print_colored("\nPara a transcrição de um áudio relacionado, por favor forneça uma URL pública de um arquivo de áudio/vídeo (ou pressione Enter para pular):", Colors.YELLOW)
                transcription_audio_url = get_user_input()
                if transcription_audio_url.strip():
                    transcription_output_dir = os.path.join(os.path.dirname(__file__), "transcription_output")
                    os.makedirs(transcription_output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                    if assemblyai_api_key:
                        try:
                            print_colored("Transcrevendo áudio com AssemblyAI...", Colors.BLUE)
                            show_thinking_animation(1.5, "AssemblyAI Processando")
                            assembly_tool = AssemblyAIAPI(api_key=assemblyai_api_key)
                            transcript_data = assembly_tool.transcribe(audio_url=transcription_audio_url)
                            transcript_text = transcript_data.get('text', '')
                            if transcript_text:
                                transcript_filename = os.path.join(transcription_output_dir, f"workflow_transcript_assembly_{timestamp}.txt")
                                with open(transcript_filename, 'w', encoding='utf-8') as f:
                                    f.write(transcript_text)
                                assets['transcript_file_assembly'] = transcript_filename
                                assets['transcript_text_assembly'] = transcript_text
                                print_success(f"Transcrição (AssemblyAI) salva em: {transcript_filename}")
                            else:
                                print_error("AssemblyAI não retornou texto da transcrição.")
                                assets['transcript_text_assembly'] = "Transcrição vazia ou falhou (AssemblyAI)"
                        except Exception as e:
                            print_error(f"Erro na etapa AssemblyAI (transcrição): {e}")
                            assets['transcript_text_assembly'] = "Erro ao transcrever com AssemblyAI"
                    elif deepgram_api_key: # Fallback
                        try:
                            print_colored("AssemblyAI não disponível/falhou. Tentando Deepgram...", Colors.YELLOW if assemblyai_api_key else Colors.BLUE)
                            show_thinking_animation(1.5, "Deepgram Processando")
                            deepgram_tool = DeepgramAPI(api_key=deepgram_api_key)
                            response = deepgram_tool.transcribe(audio_url=transcription_audio_url)
                            transcript_text = ""
                            if isinstance(response, str): # Direct text response
                                transcript_text = response
                            elif isinstance(response, dict): # Check for common structures
                                if 'results' in response and 'channels' in response['results'] and len(response['results']['channels']) > 0:
                                    transcript_text = response['results']['channels'][0].get('alternatives', [{}])[0].get('transcript', '')
                                elif 'text' in response: # Simpler dict response
                                    transcript_text = response.get('text','')

                            if transcript_text:
                                transcript_filename = os.path.join(transcription_output_dir, f"workflow_transcript_deepgram_{timestamp}.txt")
                                with open(transcript_filename, 'w', encoding='utf-8') as f:
                                    f.write(transcript_text)
                                assets['transcript_file_deepgram'] = transcript_filename
                                assets['transcript_text_deepgram'] = transcript_text
                                print_success(f"Transcrição (Deepgram) salva em: {transcript_filename}")
                            else:
                                print_error("Deepgram não retornou texto da transcrição.")
                                assets['transcript_text_deepgram'] = "Transcrição vazia ou falhou (Deepgram)"
                        except Exception as e:
                            print_error(f"Erro na etapa Deepgram (transcrição): {e}")
                            assets['transcript_text_deepgram'] = "Erro ao transcrever com Deepgram"
                    else:
                        print_colored("Nenhuma API Key para AssemblyAI ou Deepgram configurada. Pulando transcrição.", Colors.YELLOW)
                else:
                    print_colored("Nenhuma URL fornecida para transcrição. Pulando etapa.", Colors.YELLOW)

                # 6. YouTube Data API: Trend Research
                if youtube_api_key:
                    try:
                        print_colored(f"Pesquisando no YouTube sobre '{assets.get('title', workflow_prompt)}'...", Colors.BLUE)
                        show_thinking_animation(1.0, "YouTube Data API Buscando")
                        youtube_tool = YouTubeDataAPI(api_key=youtube_api_key)
                        youtube_results = youtube_tool.search_videos(query=assets.get('title', workflow_prompt), max_results=5)
                        if youtube_results:
                            assets['youtube_research'] = youtube_results
                            print_success(f"Pesquisa no YouTube concluída. {len(youtube_results)} vídeos encontrados.")
                        else:
                            assets['youtube_research'] = []
                            print_colored("Nenhum vídeo encontrado no YouTube para esta pesquisa.", Colors.YELLOW)
                    except Exception as e:
                        print_error(f"Erro na etapa YouTube Data API: {e}")
                        assets['youtube_research'] = "Erro ao pesquisar no YouTube"
                else:
                    print_colored("YOUTUBE_DATA_API_KEY não configurada. Pulando pesquisa de tendências.", Colors.YELLOW)

                # 7. Display all generated assets
                print_colored("\n\n=== Workflow Concluído! Ativos Gerados: ===", Colors.MAGENTA, bold=True)
                if 'title' in assets: print_colored(f"Título: {assets['title']}", Colors.CYAN)
                if 'description' in assets: print_colored(f"Descrição: {assets['description']}", Colors.CYAN)
                if 'script' in assets and assets['script'] != assets.get('gemini_raw_output', ''):
                    print_colored("Roteiro:", Colors.CYAN)
                    script_display = assets['script']
                    print_colored(script_display[:500] + ("..." if len(script_display) > 500 else ""), Colors.WHITE)
                elif 'gemini_raw_output' in assets: # Display if parsing failed and raw output was used
                    print_colored("Saída Bruta Gemini (usada como roteiro devido a erro de parse):", Colors.YELLOW)
                    raw_display = assets['gemini_raw_output']
                    print_colored(raw_display[:500] + ("..." if len(raw_display) > 500 else ""), Colors.WHITE)

                if 'audio_file' in assets: print_colored(f"Arquivo de Áudio: {assets['audio_file']}", Colors.GREEN if "Erro" not in assets['audio_file'] else Colors.RED)

                if 'thumbnail_file_stability' in assets: print_colored(f"Thumbnail (Stability AI): {assets['thumbnail_file_stability']}", Colors.GREEN if "Erro" not in assets['thumbnail_file_stability'] else Colors.RED)
                if 'thumbnail_url_dalle' in assets:
                    print_colored(f"Thumbnail (DALL-E URL): {assets['thumbnail_url_dalle']}", Colors.GREEN if "Erro" not in assets['thumbnail_url_dalle'] else Colors.RED)

                if assets.get('pexels_images') or assets.get('pexels_videos'):
                    print_colored("Mídia Stock (Pexels):", Colors.CYAN)
                    if assets.get('pexels_images'):
                        print_colored("  Imagens Pexels:", Colors.GREEN)
                        for img in assets['pexels_images'][:3]: # Display first 3 images
                            url = img.get('src', {}).get('original') or img.get('url') # Common Pexels structures
                            photographer = img.get('photographer', 'N/A')
                            print_colored(f"    - URL: {url} (Fotógrafo: {photographer})", Colors.WHITE)
                    if assets.get('pexels_videos'):
                        print_colored("  Vídeos Pexels:", Colors.GREEN)
                        for vid in assets['pexels_videos'][:2]: # Display first 2 videos
                            video_link = next((vf.get('link') for vf in vid.get('video_files', []) if vf.get('quality') == 'hd'), None) \
                                         or vid.get('url') # Fallback
                            user = vid.get('user', {}).get('name', 'N/A')
                            print_colored(f"    - URL: {video_link} (Usuário: {user})", Colors.WHITE)

                if assets.get('pixabay_images') or assets.get('pixabay_videos'):
                    print_colored("Mídia Stock (Pixabay):", Colors.CYAN)
                    if assets.get('pixabay_images'):
                        print_colored("  Imagens Pixabay:", Colors.GREEN)
                        for img in assets['pixabay_images'][:3]: # Display first 3
                            url = img.get('largeImageURL') or img.get('pageURL') # Common Pixabay structures
                            user = img.get('user', 'N/A')
                            print_colored(f"    - URL: {url} (Usuário: {user})", Colors.WHITE)
                            print_colored(f"    LargeImageURL: {img.get('largeImageURL')}", Colors.GRAY) # Corrected LIGHT_GRAY to GRAY
                    if assets.get('pixabay_videos'):
                        print_colored("  Vídeos Pixabay:", Colors.GREEN)
                        for vid in assets.get('pixabay_videos', [])[:2]: # Display first 2
                            video_url = vid.get('videos', {}).get('large', {}).get('url') or \
                                        vid.get('videos', {}).get('medium', {}).get('url') or \
                                        vid.get('pageURL') # Fallback
                            user = vid.get('user', 'N/A')
                            print_colored(f"    - URL: {video_url} (Usuário: {user})", Colors.WHITE)

                if 'transcript_file_assembly' in assets:
                    print_colored(f"Transcrição (AssemblyAI): {assets['transcript_file_assembly']}", Colors.GREEN)
                    print_colored("  Preview:", Colors.WHITE)
                    print_colored(assets.get('transcript_text_assembly', '')[:200] + "...", Colors.WHITE)
                elif 'transcript_text_assembly' in assets: # If only text is there (e.g. error saving file but text retrieved)
                    print_colored(f"Transcrição (AssemblyAI - Texto): {assets['transcript_text_assembly'][:200] + '...' if 'Erro' not in assets['transcript_text_assembly'] else assets['transcript_text_assembly']}", Colors.RED if "Erro" in assets['transcript_text_assembly'] else Colors.YELLOW)

                if 'transcript_file_deepgram' in assets:
                    print_colored(f"Transcrição (Deepgram): {assets['transcript_file_deepgram']}", Colors.GREEN)
                    print_colored("  Preview:", Colors.WHITE)
                    print_colored(assets.get('transcript_text_deepgram', '')[:200] + "...", Colors.WHITE)
                elif 'transcript_text_deepgram' in assets: # If only text is there
                    print_colored(f"Transcrição (Deepgram - Texto): {assets['transcript_text_deepgram'][:200] + '...' if 'Erro' not in assets['transcript_text_deepgram'] else assets['transcript_text_deepgram']}", Colors.RED if "Erro" in assets['transcript_text_deepgram'] else Colors.YELLOW)

                if 'youtube_research' in assets:
                    if isinstance(assets['youtube_research'], str) and "Erro" in assets['youtube_research']:
                        print_colored(f"Pesquisa YouTube: {assets['youtube_research']}", Colors.RED)
                    elif assets['youtube_research']:
                        print_colored("Pesquisa YouTube (Vídeos Relacionados):", Colors.CYAN)
                        for video in assets['youtube_research']:
                            title = video.get('title', 'N/A')
                            link = video.get('link', 'N/A')
                            channel = video.get('channelTitle', video.get('channel', 'N/A')) # Added fallback for channel key
                            print_colored(f"  - Título: {title}", Colors.GREEN)
                            print_colored(f"    Canal: {channel}, Link: {link}", Colors.WHITE)
                    else:
                        print_colored("Pesquisa YouTube: Nenhum resultado encontrado ou não executada.", Colors.YELLOW)

                print_colored("\n----------------------------------------\n", Colors.MAGENTA)
                # Loop back to dashboard
                continue

            elif dashboard_input.strip() == "2":
                while True: # Loop for individual tools sub-menu
                    print_colored("\n=== Usar Ferramentas Individualmente ===", Colors.MAGENTA, bold=True)
                    print_colored("Escolha uma ferramenta:", Colors.YELLOW)
                    print_colored("1. Gerar Texto/Roteiro (Gemini)", Colors.CYAN)
                    print_colored("2. Gerar Áudio (ElevenLabs)", Colors.CYAN)
                    print_colored("3. Gerar Imagem (Stability AI / DALL-E)", Colors.CYAN)
                    print_colored("4. Buscar Mídia Stock (Pexels / Pixabay)", Colors.CYAN)
                    print_colored("5. Transcrever Áudio (AssemblyAI / Deepgram)", Colors.CYAN)
                    print_colored("6. Pesquisar Vídeos (YouTube Data API)", Colors.CYAN)
                    print_colored("0. Voltar ao Menu Principal", Colors.CYAN)

                    tool_choice = get_user_input("Sua escolha: ")

                    if tool_choice == "1": # Gemini
                        gemini_api_key = config.get('GEMINI_API_KEY')
                        if not gemini_api_key:
                            print_error("GEMINI_API_KEY não configurada.")
                            continue
                        gemini_tool = GeminiAPI(api_key=gemini_api_key)
                        prompt = get_user_input("Descreva o que você quer gerar com Gemini: ")
                        if not prompt.strip():
                            print_error("Prompt não pode ser vazio.")
                            continue
                        try:
                            print_colored("Gerando com Gemini...", Colors.BLUE)
                            show_thinking_animation(1.5, "Gemini Processando")
                            response = gemini_tool.generate_content(prompt=prompt)
                            print_success("Resposta do Gemini:")
                            print_colored(response, Colors.WHITE)
                        except Exception as e:
                            print_error(f"Erro ao usar Gemini: {e}")

                    elif tool_choice == "2": # ElevenLabs
                        elevenlabs_api_key = config.get('ELEVENLABS_API_KEY')
                        if not elevenlabs_api_key:
                            print_error("ELEVENLABS_API_KEY não configurada.")
                            continue
                        tts_tool = ElevenLabsTTS(api_key=elevenlabs_api_key)
                        text_to_speak = get_user_input("Texto para converter em áudio: ")
                        if not text_to_speak.strip():
                            print_error("Texto não pode ser vazio.")
                            continue
                        try:
                            print_colored("Gerando áudio com ElevenLabs...", Colors.BLUE)
                            show_thinking_animation(1.5, "ElevenLabs Processando")
                            script_for_tts = text_to_speak
                            if len(script_for_tts) > 2500:
                                print_colored("Texto muito longo, truncando para 2500 caracteres para ElevenLabs.", Colors.YELLOW)
                                script_for_tts = script_for_tts[:2500]

                            audio_bytes = tts_tool.text_to_speech(text=script_for_tts)
                            audio_output_dir = os.path.join(os.path.dirname(__file__), "audio_output")
                            os.makedirs(audio_output_dir, exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            audio_filename = os.path.join(audio_output_dir, f"individual_audio_elevenlabs_{timestamp}.mp3")
                            with open(audio_filename, 'wb') as f:
                                f.write(audio_bytes)
                            print_success(f"Áudio salvo em: {audio_filename}")
                        except Exception as e:
                            print_error(f"Erro ao usar ElevenLabs: {e}")

                    elif tool_choice == "3": # Stability AI / DALL-E
                        stability_api_key = config.get('STABILITY_API_KEY')
                        dalle_api_key = config.get('DALLE_API_KEY')
                        if not stability_api_key and not dalle_api_key:
                            print_error("Nenhuma API Key para Stability AI ou DALL-E configurada.")
                            continue

                        prompt = get_user_input("Prompt para gerar imagem: ")
                        if not prompt.strip():
                            print_error("Prompt não pode ser vazio.")
                            continue

                        image_output_dir = os.path.join(os.path.dirname(__file__), "image_output")
                        os.makedirs(image_output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                        if stability_api_key:
                            try:
                                print_colored("Gerando imagem com Stability AI...", Colors.BLUE)
                                show_thinking_animation(1.5, "Stability AI Processando")
                                stability_tool = StabilityAIAPI(api_key=stability_api_key)
                                image_bytes = stability_tool.generate_image(prompt=prompt)
                                image_filename = os.path.join(image_output_dir, f"individual_stability_{timestamp}.png")
                                with open(image_filename, 'wb') as f:
                                    f.write(image_bytes)
                                print_success(f"Imagem (Stability AI) salva em: {image_filename}")
                            except Exception as e:
                                print_error(f"Erro ao usar Stability AI: {e}")
                                if not dalle_api_key:
                                    continue # No fallback

                        if not stability_api_key or (stability_api_key and "Erro ao usar Stability AI" in locals().get('e', '') and dalle_api_key) : # Try DALL-E if Stability failed or not available
                            if stability_api_key and dalle_api_key : print_colored("Tentando DALL-E como fallback...", Colors.YELLOW)
                            try:
                                print_colored("Gerando imagem com DALL-E...", Colors.BLUE)
                                show_thinking_animation(1.5, "DALL-E Processando")
                                dalle_tool = DalleAPI(api_key=dalle_api_key)
                                response_data = dalle_tool.generate_image(prompt=prompt)
                                if response_data and response_data.get('data') and response_data['data'][0].get('url'):
                                    image_url = response_data['data'][0]['url']
                                    print_success(f"Imagem (DALL-E) gerada. URL: {image_url}")
                                    # Consider adding download option here in future
                                else:
                                    print_error("DALL-E não retornou URL da imagem.")
                            except Exception as e:
                                print_error(f"Erro ao usar DALL-E: {e}")

                    elif tool_choice == "4": # Pexels / Pixabay
                        pexels_api_key = config.get('PEXELS_API_KEY')
                        pixabay_api_key = config.get('PIXABAY_API_KEY')
                        if not pexels_api_key and not pixabay_api_key:
                            print_error("Nenhuma API Key para Pexels ou Pixabay configurada.")
                            continue

                        query = get_user_input("O que você quer buscar (Pexels/Pixabay)? ")
                        if not query.strip():
                            print_error("Query não pode ser vazia.")
                            continue

                        media_type = get_user_input("Buscar 'imagens' ou 'videos'? ").lower()
                        if media_type not in ['imagens', 'videos']:
                            print_error("Tipo de mídia inválido. Escolha 'imagens' ou 'videos'.")
                            continue

                        media_output_dir = os.path.join(os.path.dirname(__file__), "media_output") # Ensure this exists
                        os.makedirs(media_output_dir, exist_ok=True)
                        # Note: For individual tool usage, we might not save files directly here, just display URLs
                        # unless explicitly asked or if it's the primary function (like image generation saving the image)

                        # Flag to track if Pexels succeeded, to avoid running Pixabay if Pexels worked.
                        pexels_succeeded = False
                        if pexels_api_key:
                            try:
                                print_colored(f"Buscando {media_type} no Pexels...", Colors.BLUE)
                                show_thinking_animation(1.0, "Pexels Buscando")
                                pexels_tool = PexelsAPI(api_key=pexels_api_key)
                                if media_type == 'imagens':
                                    results = pexels_tool.search_images(query=query, per_page=5) # results is a list of photo dicts
                                    print_success("Resultados Pexels (Imagens):")
                                    if results: # Corrected: check if the list is not empty
                                        for item in results: # Corrected: iterate directly over the list
                                            print_colored(f"  Fotógrafo: {item.get('photographer', 'N/A')}, URL: {item.get('url', 'N/A')}", Colors.WHITE)
                                            print_colored(f"    Src Original: {item.get('src', {}).get('original')}", Colors.GRAY) # Corrected LIGHT_GRAY to GRAY
                                        pexels_succeeded = True
                                    else: print_colored("Nenhuma imagem encontrada no Pexels.", Colors.YELLOW)
                                else: # videos
                                    results = pexels_tool.search_videos(query=query, per_page=3) # results is a list of video dicts
                                    print_success("Resultados Pexels (Vídeos):")
                                    if results: # Corrected: check if the list is not empty
                                        for item in results: # Corrected: iterate directly over the list
                                            user_name = item.get('user', {}).get('name', 'N/A')
                                            video_file_link = None
                                            if item.get('video_files'):
                                                hd_file = next((vf for vf in item['video_files'] if vf.get('quality') == 'hd' and vf.get('link')), None)
                                                if hd_file:
                                                    video_file_link = hd_file['link']
                                                else: # Fallback to any other available link
                                                    any_file = next((vf.get('link') for vf in item['video_files'] if vf.get('link')), None)
                                                    if any_file:
                                                        video_file_link = any_file
                                            video_link_display = video_file_link if video_file_link else item.get('url', 'N/A')
                                            print_colored(f"  Usuário: {user_name}, Link: {video_link_display}", Colors.WHITE)
                                        pexels_succeeded = True
                                    else: print_colored("Nenhum vídeo encontrado no Pexels.", Colors.YELLOW)
                            except Exception as e_pexels:
                                print_error(f"Erro ao usar Pexels: {e_pexels}")
                                # Fallback to Pixabay will happen if pexels_succeeded is False and pixabay_api_key exists

                        if not pexels_succeeded and pixabay_api_key:
                            if pexels_api_key : print_colored("Tentando Pixabay como fallback...", Colors.YELLOW)
                            else: print_colored(f"Buscando {media_type} no Pixabay...", Colors.BLUE)
                            try:
                                show_thinking_animation(1.0, "Pixabay Buscando")
                                pixabay_tool = PixabayAPI(api_key=pixabay_api_key)
                                if media_type == 'imagens':
                                    results = pixabay_tool.search_images(query=query, per_page=5) # results is a list of image dicts
                                    print_success("Resultados Pixabay (Imagens):")
                                    if results: # Corrected: check if the list is not empty
                                        for item in results: # Corrected: iterate directly over the list
                                            print_colored(f"  Usuário: {item.get('user', 'N/A')}, URL: {item.get('pageURL', 'N/A')}", Colors.WHITE)
                                            print_colored(f"    LargeImageURL: {item.get('largeImageURL')}", Colors.GRAY) # Corrected LIGHT_GRAY to GRAY
                                    else: print_colored("Nenhuma imagem encontrada no Pixabay.", Colors.YELLOW)
                                else: # videos
                                    results = pixabay_tool.search_videos(query=query, per_page=3) # results is a list of video dicts
                                    print_success("Resultados Pixabay (Vídeos):")
                                    if results: # Corrected: check if the list is not empty
                                        for item in results: # Corrected: iterate directly over the list
                                            video_url_display = item.get('videos', {}).get('medium', {}).get('url', item.get('pageURL', 'N/A'))
                                            print_colored(f"  Usuário: {item.get('user', 'N/A')}, URL: {video_url_display}", Colors.WHITE)
                                    else: print_colored("Nenhum vídeo encontrado no Pixabay.", Colors.YELLOW)
                            except Exception as e_pixabay:
                                print_error(f"Erro ao usar Pixabay: {e_pixabay}")
                        elif not pexels_api_key and not pixabay_api_key:
                            print_error("Nenhuma API Key para Pexels ou Pixabay configurada para esta funcionalidade.")


                    elif tool_choice == "5": # AssemblyAI / Deepgram
                        assemblyai_api_key = config.get('ASSEMBLYAI_API_KEY')
                        deepgram_api_key = config.get('DEEPGRAM_API_KEY')
                        if not assemblyai_api_key and not deepgram_api_key:
                            print_error("Nenhuma API Key para AssemblyAI ou Deepgram configurada.")
                            continue

                        audio_url = get_user_input("URL do áudio/vídeo para transcrever: ")
                        if not audio_url.strip():
                            print_error("URL não pode ser vazia.")
                            continue

                        transcription_output_dir = os.path.join(os.path.dirname(__file__), "transcription_output")
                        os.makedirs(transcription_output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        transcribed_successfully = False

                        # Primary: AssemblyAI
                        if assemblyai_api_key:
                            try:
                                print_colored("Transcrevendo com AssemblyAI...", Colors.BLUE)
                                show_thinking_animation(2.0, "AssemblyAI Processando")
                                assembly_tool = AssemblyAIAPI(api_key=assemblyai_api_key)
                                transcript_data = assembly_tool.transcribe(audio_url=audio_url)
                                transcript_text = transcript_data.get('text', '')
                                if transcript_text:
                                    transcript_filename = os.path.join(transcription_output_dir, f"individual_transcript_assembly_{timestamp}.txt")
                                    with open(transcript_filename, 'w', encoding='utf-8') as f:
                                        f.write(transcript_text)
                                    print_success(f"Transcrição (AssemblyAI) salva em: {transcript_filename}")
                                    print_colored("Preview:", Colors.CYAN)
                                    print_colored(transcript_text[:300] + ("..." if len(transcript_text) > 300 else ""), Colors.WHITE)
                                    transcribed_successfully = True
                                else:
                                    print_error("AssemblyAI não retornou texto da transcrição.")
                            except Exception as e_assembly:
                                print_error(f"Erro ao usar AssemblyAI: {e_assembly}")
                                # Fallback will be triggered if not transcribed_successfully and deepgram_api_key exists

                        # Fallback: Deepgram
                        if not transcribed_successfully and deepgram_api_key:
                            if assemblyai_api_key : print_colored("Tentando Deepgram como fallback...", Colors.YELLOW)
                            else: print_colored("Transcrevendo com Deepgram...", Colors.BLUE)
                            try:
                                show_thinking_animation(2.0, "Deepgram Processando")
                                deepgram_tool = DeepgramAPI(api_key=deepgram_api_key)
                                # Assuming DeepgramAPI().transcribe returns a dict like {'text': '...'} or similar to AssemblyAI
                                # Or it might be more complex like {'results': {'channels': [{'alternatives': [{'transcript': '...'}]}]}}
                                response_deepgram = deepgram_tool.transcribe(audio_url=audio_url)
                                transcript_text = ""
                                if isinstance(response_deepgram, str): # Direct text
                                    transcript_text = response_deepgram
                                elif isinstance(response_deepgram, dict):
                                    if 'text' in response_deepgram: # Simple case
                                        transcript_text = response_deepgram['text']
                                    elif 'results' in response_deepgram and 'channels' in response_deepgram['results'] and \
                                        len(response_deepgram['results']['channels']) > 0 and \
                                        'alternatives' in response_deepgram['results']['channels'][0] and \
                                        len(response_deepgram['results']['channels'][0]['alternatives']) > 0:
                                        transcript_text = response_deepgram['results']['channels'][0]['alternatives'][0].get('transcript', '')

                                if transcript_text:
                                    transcript_filename = os.path.join(transcription_output_dir, f"individual_transcript_deepgram_{timestamp}.txt")
                                    with open(transcript_filename, 'w', encoding='utf-8') as f:
                                        f.write(transcript_text)
                                    print_success(f"Transcrição (Deepgram) salva em: {transcript_filename}")
                                    print_colored("Preview:", Colors.CYAN)
                                    print_colored(transcript_text[:300] + ("..." if len(transcript_text) > 300 else ""), Colors.WHITE)
                                    transcribed_successfully = True
                                else:
                                    print_error("Deepgram não retornou texto da transcrição.")
                            except Exception as e_deepgram:
                                print_error(f"Erro ao usar Deepgram: {e_deepgram}")

                        if not transcribed_successfully and not assemblyai_api_key and not deepgram_api_key:
                            print_error("Nenhuma API de transcrição configurada ou ambas falharam.")


                    elif tool_choice == "6": # YouTube Data API
                        youtube_api_key = config.get('YOUTUBE_DATA_API_KEY')
                        if not youtube_api_key:
                            print_error("YOUTUBE_DATA_API_KEY não configurada.")
                            continue

                        query = get_user_input("O que você quer pesquisar no YouTube? ")
                        if not query.strip():
                            print_error("Query não pode ser vazia.")
                            continue
                        try:
                            print_colored("Pesquisando no YouTube...", Colors.BLUE)
                            show_thinking_animation(1.5, "YouTube Data API Buscando")
                            youtube_tool = YouTubeDataAPI(api_key=youtube_api_key)
                            results = youtube_tool.search_videos(query=query, max_results=5)
                            print_success("Resultados da Pesquisa YouTube:")
                            if results:
                                for item in results:
                                    snippet = item.get('snippet', {})
                                    video_id = item.get('id', {}).get('videoId')

                                    title = snippet.get('title', 'N/A')
                                    channel = snippet.get('channelTitle', 'N/A')
                                    link = f"https://www.youtube.com/watch?v={video_id}" if video_id else "N/A"

                                    print_colored(f"  Título: {title}", Colors.GREEN)
                                    print_colored(f"    Canal: {channel}, Link: {link}", Colors.WHITE)
                            else:
                                print_colored("Nenhum vídeo encontrado.", Colors.YELLOW)
                        except Exception as e:
                            print_error(f"Erro ao usar YouTube Data API: {e}")

                    elif tool_choice == "0":
                        print_colored("Retornando ao menu principal...", Colors.YELLOW)
                        show_thinking_animation(0.5, "Retornando")
                        break # Exit individual tools sub-menu loop, goes back to main dashboard loop
                    else:
                        print_error("Opção inválida. Tente novamente.")

                    print_colored("\n----------------------------------------", Colors.MAGENTA)
                    sub_menu_action = get_user_input("Pressione Enter para escolher outra ferramenta ou '0' para voltar ao menu principal: ")
                    if sub_menu_action == "0":
                        print_colored("Retornando ao menu principal...", Colors.YELLOW)
                        show_thinking_animation(0.5, "Retornando")
                        break # Exit individual tools sub-menu loop

                # After breaking from sub-menu, the main dashboard loop will continue
                continue

            elif dashboard_input.strip() == "0":
                print_success("Saindo do Content Creation Dashboard.")
                break # Exit the dashboard loop, proceed to chat or exit app
            elif not dashboard_input.strip(): # User pressed Enter, proceed to chat
                break # Exit the dashboard loop
            else:
                print_error("Opção inválida. Por favor, escolha uma das opções do dashboard ou pressione Enter para o chat.")
                show_thinking_animation(1.0, "Processando")
                continue # Show dashboard again

        # Main chat loop (if user exits dashboard by pressing Enter or after option 0 if not exiting app)
        if dashboard_input.strip() == "0" and not config.get("allow_chat_after_dashboard_exit", True): # Example config
            print_colored("Saindo da aplicação.", Colors.BLUE)
            sys.exit(0)

        print_colored("\nIniciando sessão de chat. Digite 'ajuda' para comandos, 'sair' para terminar.", Colors.GREEN)
        # ... rest of the main chat loop ...

    except ConfigError as e:
        print_error(f"Erro de configuração: {str(e)}")
        print("Verifique seu arquivo .env e tente novamente.")
        sys.exit(1)

    except Exception as e:
        print_error(f"Erro inesperado: {str(e)}")
        print("Detalhes do erro:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
