"""
Document Analysis Agent for the Don Corleone AI project.

This module implements a specialized agent for document analysis,
summarization, and information extraction.
"""

import os
import json
import re
import tempfile
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .base_agent import Agent, TaskConfig, Task, ExecutionResult, AgentStatus, AgentRegistry


@AgentRegistry.register
class DocumentAgent(Agent):
    """Agent that processes and analyzes documents."""

    SUPPORTED_TASK_TYPES = ["document_summary", "text_extraction", "document_qa"]
    SUPPORTED_FILE_TYPES = [".txt", ".md", ".pdf", ".docx", ".json", ".csv"]

    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        """Initialize a new DocumentAgent.

        Args:
            agent_id: Optional ID for the agent. If not provided, a UUID will be generated.
            name: Optional name for the agent.
        """
        super().__init__(agent_id=agent_id, name=name or "Document Agent")
        self.logger = logging.getLogger("agent.document")

    def can_handle_task(self, task_config: TaskConfig) -> bool:
        """Check if this agent can handle a task.

        Args:
            task_config: Configuration for the task.

        Returns:
            True if the agent can handle the task, False otherwise.
        """
        return task_config.task_type in self.SUPPORTED_TASK_TYPES

    def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a document processing task.

        Args:
            task: The task to execute.

        Returns:
            The result of the execution.
        """
        if not task.config:
            return ExecutionResult(
                success=False,
                output=None,
                error="Task configuration is missing"
            )

        task_type = task.config.task_type
        parameters = task.config.parameters

        # Validate file path in parameters
        file_path = parameters.get("file_path")
        if not file_path or not os.path.isfile(file_path):
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Invalid or missing file path: {file_path}"
            )

        # Check file type
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_FILE_TYPES:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Unsupported file type: {file_ext}"
            )

        # Execute task based on type
        if task_type == "document_summary":
            return self._execute_document_summary(file_path, parameters)
        elif task_type == "text_extraction":
            return self._execute_text_extraction(file_path, parameters)
        elif task_type == "document_qa":
            return self._execute_document_qa(file_path, parameters)
        else:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Unsupported task type: {task_type}"
            )

    def _execute_document_summary(self, file_path: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a document summarization task.

        Args:
            file_path: Path to the document.
            parameters: Additional parameters for the task.

        Returns:
            The result of the execution.
        """
        max_length = parameters.get("max_length", 1000)
        format_type = parameters.get("format", "markdown")
        include_metadata = parameters.get("include_metadata", True)

        try:
            self.logger.info(f"Summarizing document: {file_path}")

            # Read document content
            content = self._read_document(file_path)

            # Extract document metadata
            metadata = self._extract_document_metadata(file_path)

            # Generate summary
            summary = self._generate_summary(content, max_length)

            # Format the output
            result = self._format_summary(summary, metadata, format_type, include_metadata)

            return ExecutionResult(
                success=True,
                output=result,
                artifacts=[
                    {
                        "type": "document_summary",
                        "document_path": file_path,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            )

        except Exception as e:
            self.logger.exception(f"Error summarizing document: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Error summarizing document: {str(e)}"
            )

    def _execute_text_extraction(self, file_path: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a text extraction task.

        Args:
            file_path: Path to the document.
            parameters: Additional parameters for the task.

        Returns:
            The result of the execution.
        """
        extraction_pattern = parameters.get("pattern")
        extraction_type = parameters.get("type", "regex")  # regex, section, or keyword

        try:
            self.logger.info(f"Extracting text from document: {file_path}")

            # Read document content
            content = self._read_document(file_path)

            # Extract text based on the extraction type
            if extraction_type == "regex" and extraction_pattern:
                extracted_text = self._extract_text_by_regex(content, extraction_pattern)
            elif extraction_type == "section":
                section_name = parameters.get("section_name", "")
                extracted_text = self._extract_text_by_section(content, section_name)
            elif extraction_type == "keyword":
                keywords = parameters.get("keywords", [])
                context_size = parameters.get("context_size", 100)
                extracted_text = self._extract_text_by_keywords(content, keywords, context_size)
            else:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Invalid extraction type or missing pattern: {extraction_type}"
                )

            return ExecutionResult(
                success=True,
                output=extracted_text,
                artifacts=[
                    {
                        "type": "text_extraction",
                        "document_path": file_path,
                        "extraction_type": extraction_type,
                        "pattern": extraction_pattern,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            )

        except Exception as e:
            self.logger.exception(f"Error extracting text: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Error extracting text: {str(e)}"
            )

    def _execute_document_qa(self, file_path: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a document Q&A task.

        Args:
            file_path: Path to the document.
            parameters: Additional parameters for the task.

        Returns:
            The result of the execution.
        """
        question = parameters.get("question")
        if not question:
            return ExecutionResult(
                success=False,
                output=None,
                error="Question parameter is missing"
            )

        try:
            self.logger.info(f"Answering question about document: {file_path}")

            # Read document content
            content = self._read_document(file_path)

            # Generate answer
            answer = self._answer_question(content, question)

            return ExecutionResult(
                success=True,
                output=answer,
                artifacts=[
                    {
                        "type": "document_qa",
                        "document_path": file_path,
                        "question": question,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            )

        except Exception as e:
            self.logger.exception(f"Error answering question: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Error answering question: {str(e)}"
            )

    def _read_document(self, file_path: str) -> str:
        """Read the content of a document.

        Args:
            file_path: Path to the document.

        Returns:
            Document content as text.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".txt" or file_ext == ".md":
            # Read plain text file
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_ext == ".json":
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert JSON to text representation
                return json.dumps(data, indent=2)

        elif file_ext == ".csv":
            # Read CSV file as text
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_ext == ".pdf":
            # For PDF files, we would normally use a library like PyPDF2 or pdfplumber
            # For simplicity, we'll return a placeholder message
            return f"[PDF CONTENT FROM {file_path}]"

        elif file_ext == ".docx":
            # For DOCX files, we would normally use a library like python-docx
            # For simplicity, we'll return a placeholder message
            return f"[DOCX CONTENT FROM {file_path}]"

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a document.

        Args:
            file_path: Path to the document.

        Returns:
            Dictionary of metadata.
        """
        # Basic file metadata
        stat_info = os.stat(file_path)

        metadata = {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_bytes": stat_info.st_size,
            "file_type": os.path.splitext(file_path)[1].lower(),
            "last_modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        }

        # Add file-specific metadata
        file_ext = os.path.splitext(file_path)[1].lower()

        # In a real implementation, we would extract additional metadata
        # based on file type (e.g., PDF author, DOCX properties)

        return metadata

    def _generate_summary(self, content: str, max_length: int) -> str:
        """Generate a summary of the document content.

        Args:
            content: Document content.
            max_length: Maximum length of the summary.

        Returns:
            Document summary.
        """
        # For a simple implementation, we'll just truncate the content
        # In a real implementation, we would use NLP techniques for summarization

        if len(content) <= max_length:
            return content

        # Simple extraction of the first part of the document
        summary = content[:max_length]

        # Try to end at a sentence boundary
        last_period = summary.rfind(".")
        if last_period > 0:
            summary = summary[:last_period + 1]

        summary += "\n\n[...Document continues...]"

        return summary

    def _format_summary(self, summary: str, metadata: Dict[str, Any],
                       format_type: str, include_metadata: bool) -> str:
        """Format the summary according to the specified format.

        Args:
            summary: Document summary.
            metadata: Document metadata.
            format_type: Output format (markdown, plain, html).
            include_metadata: Whether to include metadata in the output.

        Returns:
            Formatted summary.
        """
        if format_type == "markdown":
            output = f"# Document Summary: {metadata['filename']}\n\n"

            if include_metadata:
                output += "## Metadata\n\n"
                output += f"- **File:** {metadata['filename']}\n"
                output += f"- **Type:** {metadata['file_type']}\n"
                output += f"- **Size:** {metadata['file_size_bytes']} bytes\n"
                output += f"- **Last Modified:** {metadata['last_modified']}\n\n"

            output += "## Summary\n\n"
            output += summary

        elif format_type == "html":
            output = f"<h1>Document Summary: {metadata['filename']}</h1>\n\n"

            if include_metadata:
                output += "<h2>Metadata</h2>\n<ul>\n"
                output += f"<li><strong>File:</strong> {metadata['filename']}</li>\n"
                output += f"<li><strong>Type:</strong> {metadata['file_type']}</li>\n"
                output += f"<li><strong>Size:</strong> {metadata['file_size_bytes']} bytes</li>\n"
                output += f"<li><strong>Last Modified:</strong> {metadata['last_modified']}</li>\n"
                output += "</ul>\n\n"

            output += "<h2>Summary</h2>\n"
            output += f"<p>{summary.replace('\n', '<br>')}</p>"

        else:  # plain text
            output = f"DOCUMENT SUMMARY: {metadata['filename']}\n\n"

            if include_metadata:
                output += "METADATA:\n"
                output += f"File: {metadata['filename']}\n"
                output += f"Type: {metadata['file_type']}\n"
                output += f"Size: {metadata['file_size_bytes']} bytes\n"
                output += f"Last Modified: {metadata['last_modified']}\n\n"

            output += "SUMMARY:\n\n"
            output += summary

        return output

    def _extract_text_by_regex(self, content: str, pattern: str) -> str:
        """Extract text from document using a regex pattern.

        Args:
            content: Document content.
            pattern: Regex pattern to match.

        Returns:
            Extracted text.
        """
        try:
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

            if not matches:
                return "No matches found for the specified pattern."

            # Format the results
            result = f"Extracted {len(matches)} matches using pattern: {pattern}\n\n"

            for i, match in enumerate(matches, 1):
                result += f"--- Match {i} ---\n{match}\n\n"

            return result

        except re.error as e:
            return f"Error in regex pattern: {str(e)}"

    def _extract_text_by_section(self, content: str, section_name: str) -> str:
        """Extract a section from the document.

        Args:
            content: Document content.
            section_name: Name of the section to extract.

        Returns:
            Extracted section text.
        """
        # Simple section extraction - looks for headers in various formats
        section_patterns = [
            # Markdown headers
            rf"#+\s*{re.escape(section_name)}\s*\n(.*?)(?:\n#+\s|$)",
            # Document section headers
            rf"{re.escape(section_name)}[:\s]*\n(.*?)(?:\n\s*\n|$)",
        ]

        for pattern in section_patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if match:
                return f"Section: {section_name}\n\n{match.group(1).strip()}"

        return f"Section '{section_name}' not found in the document."

    def _extract_text_by_keywords(self, content: str, keywords: List[str], context_size: int) -> str:
        """Extract text around keywords from the document.

        Args:
            content: Document content.
            keywords: List of keywords to search for.
            context_size: Number of characters to include around each keyword.

        Returns:
            Extracted text with context.
        """
        if not keywords:
            return "No keywords specified for extraction."

        results = []

        for keyword in keywords:
            # Find all occurrences of the keyword
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)

            for match in pattern.finditer(content):
                start_pos = max(0, match.start() - context_size)
                end_pos = min(len(content), match.end() + context_size)

                # Extract context around the keyword
                context = content[start_pos:end_pos]

                # Highlight the keyword in the context
                highlighted = context.replace(
                    match.group(), f"**{match.group()}**"
                )

                results.append({
                    "keyword": keyword,
                    "context": highlighted,
                    "position": match.start()
                })

        if not results:
            return f"No occurrences of keywords {', '.join(keywords)} found in the document."

        # Format the results
        output = f"Found {len(results)} occurrences of keywords: {', '.join(keywords)}\n\n"

        # Sort by position in the document
        results.sort(key=lambda x: x["position"])

        for i, result in enumerate(results, 1):
            output += f"--- Occurrence {i} ({result['keyword']}) ---\n"
            output += f"{result['context']}\n\n"

        return output

    def _answer_question(self, content: str, question: str) -> str:
        """Answer a question based on document content.

        Args:
            content: Document content.
            question: Question to answer.

        Returns:
            Answer to the question.
        """
        # In a real implementation, we would use NLP/LLM techniques for Q&A
        # This is a simplified implementation

        # Format the question
        formatted_question = question.strip().rstrip("?") + "?"

        # Generate a simple answer based on keyword matching
        # This is just a placeholder - a real implementation would be more sophisticated
        keywords = [word.lower() for word in question.split() if len(word) > 3]

        # Search for sections containing keywords
        matches = []
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            match_score = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
            if match_score > 0:
                matches.append((match_score, sentence))

        # Sort by relevance score
        matches.sort(reverse=True)

        if not matches:
            return f"No relevant information found in the document to answer: {formatted_question}"

        # Build the answer
        answer = f"Q: {formatted_question}\n\nA: Based on the document content:\n\n"

        for i, (score, sentence) in enumerate(matches[:3], 1):
            answer += f"{i}. {sentence}.\n\n"

        answer += f"This information was extracted based on keyword relevance to your question."

        return answer

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentAgent':
        """Create a DocumentAgent from a dictionary.

        Args:
            data: Dictionary containing agent data.

        Returns:
            A DocumentAgent instance.
        """
        agent = cls(agent_id=data["agent_id"], name=data["name"])

        # Restore tasks
        for task_id, task_data in data.get("tasks", {}).items():
            agent.tasks[task_id] = Task.from_dict(task_data)

        return agent
