"""
Memory Manager for LangGraph 101 project.

This module handles the extraction, storage, and retrieval of important
contextual information from conversations.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from langchain_core.messages import HumanMessage, AIMessage

import uuid
import logging
from database import get_database
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download nltk resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryItem:
    """A single memory item that represents an important fact or context."""

    def __init__(self, content: str, source_message: str,
                 timestamp: Optional[str] = None,
                 category: str = "general",
                 memory_id: Optional[str] = None,
                 importance: float = 1.0,
                 access_count: int = 0):
        """Initialize a memory item.

        Args:
            content: The content of the memory.
            source_message: The original message that this memory was extracted from.
            timestamp: When this memory was created (ISO format).
            category: The category of this memory (e.g., "personal", "factual", "preference").
            memory_id: Optional ID for the memory. If not provided, a UUID will be generated.
            importance: Initial importance score of the memory.
            access_count: Initial access count of the memory.
        """
        self.content = content
        self.source_message = source_message
        self.timestamp = timestamp or datetime.now().isoformat()
        self.category = category
        self.memory_id = memory_id or str(uuid.uuid4())
        self.importance = importance  # Initial importance score
        self.access_count = access_count  # How many times this memory was accessed

    def to_dict(self) -> Dict[str, Any]:
        """Convert this memory item to a dictionary.

        Returns:
            Dictionary representation of this memory item.
        """
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "source_message": self.source_message,
            "timestamp": self.timestamp,
            "category": self.category,
            "importance": self.importance,
            "access_count": self.access_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create a memory item from a dictionary.

        Args:
            data: Dictionary containing memory item data.

        Returns:
            A MemoryItem instance.
        """
        return cls(
            content=data["content"],
            source_message=data.get("source_message", ""),
            timestamp=data.get("timestamp"),
            category=data.get("category", "general"),
            memory_id=data.get("memory_id"),
            importance=data.get("importance", 1.0),
            access_count=data.get("access_count", 0)
        )


class MemoryManager:
    """Manages the extraction, storage, and retrieval of memory items."""

    def __init__(self, conversation_id: Optional[str] = None,
                 max_items: int = 100,
                 extraction_enabled: bool = True):
        """Initialize the memory manager.

        Args:
            conversation_id: ID of the conversation these memories belong to.
            max_items: Maximum number of memory items to keep in memory.
            extraction_enabled: Whether to automatically extract memory items.
        """
        self.memories: List[MemoryItem] = []
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.db = get_database()
        self.max_items = max_items
        self.extraction_enabled = extraction_enabled        # Load existing memories from database if available
        self._load_from_db()

        # Initialize NLP tools with error handling
        try:
            self.stop_words = set(stopwords.words('english'))
        except (AttributeError, LookupError) as e:
            # Fallback to basic English stop words if NLTK data is not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
            print(f"Warning: NLTK stopwords not available, using fallback set: {e}")
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except (AttributeError, LookupError) as e:
            # Create a fallback lemmatizer that just returns the word as-is
            class FallbackLemmatizer:
                def lemmatize(self, word, pos='n'):
                    return word
            self.lemmatizer = FallbackLemmatizer()
            print(f"Warning: NLTK WordNetLemmatizer not available, using fallback: {e}")

    def add_memory(self, content: str, source_message: str,
                  category: str = "general") -> MemoryItem:
        """Add a memory item manually.

        Args:
            content: The content of the memory.
            source_message: The original message that this memory was extracted from.
            category: The category of this memory.

        Returns:
            The created memory item.
        """
        # Check if we already have a similar memory
        for memory in self.memories:
            if self._is_similar(memory.content, content):
                # Update access count
                memory.access_count += 1
                # Update in database
                self.db.update_memory(memory.memory_id, memory.to_dict())
                self._sort_and_trim()
                return memory

        # Create new memory
        memory = MemoryItem(content, source_message, category=category)
        self.memories.append(memory)

        # Store in database
        self.db.create_memory(self.conversation_id, memory.memory_id, memory.to_dict())

        # Sort and trim if needed
        self._sort_and_trim()

        return memory

    def extract_memories(self, message: str, is_user: bool = True) -> List[MemoryItem]:
        """Extract memories from a message.

        Args:
            message: The message to extract memories from.
            is_user: Whether this is a user message.

        Returns:
            List of extracted memory items.
        """
        if not self.extraction_enabled:
            return []

        extracted_items = []

        # Skip extraction for short or basic messages
        if len(message.split()) < 4:
            return []

        # Identify potential memory items
        # 1. Personal information
        if is_user:
            personal_info = self._extract_personal_info(message)
            for info in personal_info:
                item = self.add_memory(
                    content=info,
                    source_message=message,
                    category="personal"
                )
                extracted_items.append(item)

        # 2. Preferences
        preferences = self._extract_preferences(message)
        for preference in preferences:
            item = self.add_memory(
                content=preference,
                source_message=message,
                category="preference"
            )
            extracted_items.append(item)

        # 3. Important facts
        facts = self._extract_facts(message)
        for fact in facts:
            item = self.add_memory(
                content=fact,
                source_message=message,
                category="factual"
            )
            extracted_items.append(item)

        return extracted_items

    def get_relevant_memories(self, query: str, limit: int = 3) -> List[MemoryItem]:
        """Get memories relevant to a query.

        Args:
            query: The query to find relevant memories for.
            limit: Maximum number of memories to return.

        Returns:
            List of relevant memory items.
        """
        if not self.memories:
            return []

        # Preprocess query
        query_tokens = self._preprocess_text(query)

        # Preprocess memory contents
        memory_texts = [self._preprocess_text(memory.content) for memory in self.memories]

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(memory_texts + [query_tokens])

        # Compute cosine similarity
        query_vector = tfidf_matrix[-1]
        memory_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, memory_vectors).flatten()

        # Combine similarity scores with importance
        scored_memories = [
            (memory, similarity * memory.importance)
            for memory, similarity in zip(self.memories, similarities)
        ]

        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Update access count for retrieved memories
        results = []
        for memory, _ in scored_memories[:limit]:
            memory.access_count += 1
            results.append(memory)

        return results

    def get_memory_summary(self, limit: int = 5) -> str:
        """Get a summary of the most important memories.

        Args:
            limit: Maximum number of memories to include in the summary.

        Returns:
            A string summary of important memories.
        """
        if not self.memories:
            return "No memories stored yet."

        # Sort by importance and get top items
        sorted_memories = sorted(
            self.memories,
            key=lambda x: x.importance + (0.1 * x.access_count),
            reverse=True
        )

        # Build summary
        summary = "Key information I remember:\n"
        for i, memory in enumerate(sorted_memories[:limit], 1):
            category_emoji = {
                "personal": "👤",
                "preference": "❤️",
                "factual": "📝",
                "general": "💡"
            }.get(memory.category, "💡")

            summary += f"{i}. {category_emoji} {memory.content}\n"

        return summary

    def clear(self) -> None:
        """Clear all memories in memory.
        Note: This does not delete memories from the database.
        """
        self.memories = []

    def clear_all(self) -> bool:
        """Clear all memories from memory and database.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.memories = []
            self.db.delete_conversation_memories(self.conversation_id)
            return True
        except Exception as e:
            logger.error(f"Error clearing all memories: {e}")
            return False

    def save_all(self) -> bool:
        """Save all memories to the database.

        Returns:
            True if successful, False otherwise.
        """
        try:
            for memory in self.memories:
                self.db.update_memory(memory.memory_id, memory.to_dict())
            return True
        except Exception as e:
            logger.error(f"Error saving all memories to database: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from database and local cache.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Remove from local cache
            self.memories = [m for m in self.memories if m.memory_id != memory_id]
            # Remove from database
            self.db.delete_memory(memory_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False

    def _load_from_db(self) -> None:
        """Load memories from the database."""
        try:
            db_memories = self.db.get_memories(self.conversation_id)

            if db_memories:
                self.memories = [MemoryItem.from_dict(m) for m in db_memories]
                logger.info(f"Loaded {len(self.memories)} memories from database for conversation {self.conversation_id}")
                self._sort_and_trim()
            else:
                logger.info(f"No memories found in database for conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Error loading memories from database: {e}")

    def change_conversation(self, conversation_id: str) -> None:
        """Change the conversation ID and reload memories from database.

        Args:
            conversation_id: The new conversation ID to switch to.
        """
        if conversation_id == self.conversation_id:
            return  # No change needed

        # Update conversation ID
        self.conversation_id = conversation_id

        # Clear existing memories
        self.memories = []

        # Load memories for the new conversation
        self._load_from_db()

        logger.info(f"Switched to conversation: {conversation_id} with {len(self.memories)} memories")

    def _sort_and_trim(self) -> None:
        """Sort memories by importance and trim to max_items if needed."""
        # Sort by importance and recency
        self.memories.sort(
            key=lambda x: (x.importance + (0.1 * x.access_count), x.timestamp),
            reverse=True
        )

        # Trim if needed
        if len(self.memories) > self.max_items:
            self.memories = self.memories[:self.max_items]

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two texts are semantically similar using TF-IDF and cosine similarity.

        Args:
            text1: First text.
            text2: Second text.
            threshold: Similarity threshold (0.0 to 1.0).

        Returns:
            True if the texts are similar, False otherwise.
        """
        # Direct match
        if text1.lower() == text2.lower():
            return True

        # Preprocess texts
        t1 = self._preprocess_text(text1)
        t2 = self._preprocess_text(text2)

        # If either text is empty after preprocessing, use the simple method
        if not t1 or not t2:
            # Simple word overlap metric
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return False

            overlap = len(words1.intersection(words2))
            overlap_ratio = overlap / min(len(words1), len(words2))

            return overlap_ratio > threshold

        try:
            # Use TF-IDF and cosine similarity for more accurate comparison
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([t1, t2])

            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]

            return cosine_sim > threshold

        except Exception as e:
            logger.warning(f"Error in semantic similarity calculation: {e}. Falling back to basic method.")
            # Fall back to simple method
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return False

            overlap = len(words1.intersection(words2))
            overlap_ratio = overlap / min(len(words1), len(words2))

            return overlap_ratio > threshold

    def _extract_personal_info(self, message: str) -> List[str]:
        """Extract personal information from a message.

        Args:
            message: The message to extract from.

        Returns:
            List of extracted personal information items.
        """
        results = []

        # Name patterns
        name_patterns = [
            r"(?:my name is|I am|I'm) (?P<name>[A-Z][a-z]+ [A-Z][a-z]+)",
            r"(?:called|name's) (?P<name>[A-Z][a-z]+ [A-Z][a-z]+)"
        ]

        for pattern in name_patterns:
            matches = re.finditer(pattern, message)
            for match in matches:
                name = match.group("name")
                results.append(f"User's name is {name}")

        # Simple personal fact patterns
        personal_patterns = [
            r"I (?:am|'m) (?:a|an) (?P<role>[a-z]+(?:\s[a-z]+){0,2})",
            r"I work(?:ed)? (?:as|at|in) (?:a|an|the) (?P<workplace>[a-z]+(?:\s[a-z]+){0,3})",
            r"I live(?:d)? in (?P<location>[A-Za-z]+(?:\s[A-Za-z]+){0,3})"
        ]

        for pattern in personal_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                for key, value in match.groupdict().items():
                    if value:
                        if key == "role":
                            results.append(f"User is a {value}")
                        elif key == "workplace":
                            results.append(f"User works at {value}")
                        elif key == "location":
                            results.append(f"User lives in {value}")

        return results

    def _extract_preferences(self, message: str) -> List[str]:
        """Extract preferences from a message.

        Args:
            message: The message to extract from.

        Returns:
            List of extracted preference items.
        """
        results = []

        # Preference patterns
        preference_patterns = [
            r"I (?:like|love|enjoy|prefer) (?P<liked>[a-z]+(?:\s[a-z]+){0,5})",
            r"I (?:don't like|hate|dislike|can't stand) (?P<disliked>[a-z]+(?:\s[a-z]+){0,5})",
            r"My favorite (?P<category>[a-z]+) is (?P<favorite>[a-z]+(?:\s[a-z]+){0,5})"
        ]

        for pattern in preference_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                for key, value in match.groupdict().items():
                    if value:
                        if key == "liked":
                            results.append(f"User likes {value}")
                        elif key == "disliked":
                            results.append(f"User dislikes {value}")
                        elif key == "favorite" and "category" in match.groupdict():
                            category = match.groupdict()["category"]
                            results.append(f"User's favorite {category} is {value}")

        return results

    def _extract_facts(self, message: str) -> List[str]:
        """Extract important facts from a message.

        Args:
            message: The message to extract from.

        Returns:
            List of extracted fact items.
        """
        # This is a very simple implementation
        # In a real system, we'd use NLP to identify entities and facts

        # If the message is very short, don't try to extract facts
        if len(message.split()) < 8:
            return []

        # For now, just extract sentences that might contain facts
        # (sentences with dates, numbers, or specific keywords)
        sentences = re.split(r'[.!?]+', message)

        results = []
        fact_indicators = [
            r'\b\d{4}\b',  # Years
            r'\b\d+\s*%\b',  # Percentages
            r'\b(?:is|are|was|were)\b.*\b(?:first|oldest|largest|smallest|fastest|slowest)\b',  # Superlatives
            r'\b(?:discovered|invented|created|founded|established)\b'  # Creation verbs
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 5:
                continue

            for indicator in fact_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    results.append(sentence)
                    break

        return results

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for semantic similarity.

        Args:
            text: The text to preprocess.

        Returns:
            Preprocessed text as a single string.
        """
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(tokens)

    def cluster_memories(self, num_clusters: int = 3) -> Dict[str, List[MemoryItem]]:
        """Cluster memories into semantic groups.

        This method uses TF-IDF and K-means clustering to group related memories.

        Args:
            num_clusters: Number of clusters to create.

        Returns:
            Dictionary mapping cluster names to lists of memory items.
        """
        if len(self.memories) < num_clusters:
            # If we have fewer memories than requested clusters, return all in one group
            return {"all_memories": self.memories}

        try:
            # Preprocess memory texts
            memory_texts = [self._preprocess_text(memory.content) for memory in self.memories]

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(memory_texts)

            # Perform K-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(num_clusters, len(self.memories)), random_state=42)
            kmeans.fit(tfidf_matrix)

            # Get cluster labels
            labels = kmeans.labels_

            # Group memories by cluster
            clusters = {}
            for i, label in enumerate(labels):
                cluster_name = f"cluster_{label}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(self.memories[i])

            # Generate topic names for each cluster
            for cluster_name, cluster_memories in clusters.items():
                # Find most common words in the cluster
                cluster_texts = " ".join([memory.content for memory in cluster_memories])
                words = self._preprocess_text(cluster_texts).split()

                # Count word frequencies
                from collections import Counter
                word_counts = Counter(words)

                # Get top words for the cluster
                top_words = [word for word, _ in word_counts.most_common(3)]

                # Update cluster name if we have top words
                if top_words:
                    new_name = "_".join(top_words)
                    clusters[new_name] = clusters.pop(cluster_name)

            return clusters

        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
            # Fall back to returning all memories
            return {"all_memories": self.memories}


def get_memory_manager(conversation_id: Optional[str] = None,
                       max_items: int = 100,
                       extraction_enabled: bool = True) -> MemoryManager:
    """Get a memory manager instance.

    Args:
        conversation_id: Optional ID for the conversation. If not provided, a UUID will be generated.
        max_items: Maximum number of memory items to keep in memory.
        extraction_enabled: Whether to automatically extract memories.

    Returns:
        A MemoryManager instance.
    """
    return MemoryManager(
        conversation_id=conversation_id,
        max_items=max_items,
        extraction_enabled=extraction_enabled
    )
