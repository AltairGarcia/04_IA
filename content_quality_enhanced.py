"""
Enhanced Content Quality Enhancement Module

This module provides improved content quality analysis with:
- Better semantic understanding
- More flexible keyword matching
- Enhanced image-topic relevance scoring
- Contextual content analysis
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import statistics
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class EnhancedContentQualityAnalyzer:
    """Enhanced content quality analysis with improved algorithms."""

    def __init__(self):
        """Initialize the enhanced content quality analyzer."""
        self.quality_metrics = {
            "image_quality": {
                "min_resolution": 600,  # Reduced from 800 for more realistic scoring
                "preferred_aspect_ratios": [(16, 9), (4, 3), (1, 1), (3, 4)],  # Added vertical format
                "max_file_size_mb": 10
            },
            "text_quality": {
                "min_length": 50,  # Reduced from 100
                "max_length": 5000,
                "readability_target": 8.0,
                "keyword_density_max": 0.05  # Increased from 0.03
            },
            "content_relevance": {
                "min_relevance_score": 0.4,  # Reduced from 0.7
                "semantic_similarity_threshold": 0.3  # Reduced from 0.6
            }
        }

        # Common synonyms and related terms
        self.topic_expansions = {
            "machine learning": ["ml", "ai", "artificial intelligence", "algorithm", "neural network", "deep learning"],
            "artificial intelligence": ["ai", "machine learning", "ml", "neural", "algorithm", "automation"],
            "programming": ["coding", "development", "software", "code", "developer", "tech"],
            "web development": ["website", "html", "css", "javascript", "frontend", "backend"],
            "data science": ["data", "analytics", "statistics", "python", "analysis", "visualization"]
        }

    def get_expanded_keywords(self, topic: str) -> List[str]:
        """Get expanded keywords for a topic including synonyms and related terms."""
        topic_lower = topic.lower()
        keywords = [topic_lower]

        # Add direct topic words
        keywords.extend(topic_lower.split())

        # Add synonyms from predefined expansions
        for base_topic, expansions in self.topic_expansions.items():
            if base_topic in topic_lower or any(word in topic_lower for word in base_topic.split()):
                keywords.extend(expansions)

        # Remove duplicates and short words
        keywords = list(set([k for k in keywords if len(k) > 2]))
        return keywords

    def analyze_image_quality_enhanced(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced image quality analysis with more realistic scoring."""
        if not images:
            return {"score": 0, "issues": ["No images provided"], "recommendations": []}

        scores = []
        issues = []
        high_quality_images = []

        for i, img in enumerate(images):
            img_score = 0
            img_issues = []

            # Check resolution (more forgiving)
            width = img.get("width", 0)
            height = img.get("height", 0)

            if width >= 1200 and height >= 800:
                img_score += 35  # High quality
            elif width >= 800 and height >= 600:
                img_score += 30  # Good quality
            elif width >= 600 and height >= 400:
                img_score += 25  # Acceptable quality
            elif width >= 400 and height >= 300:
                img_score += 15  # Low but usable
            else:
                img_issues.append(f"Very low resolution: {width}x{height}")
                img_score += 5

            # Check aspect ratio (more forgiving)
            if width > 0 and height > 0:
                aspect_ratio = width / height
                best_ratio_match = min(
                    self.quality_metrics["image_quality"]["preferred_aspect_ratios"],
                    key=lambda r: abs(aspect_ratio - (r[0] / r[1]))
                )
                ratio_diff = abs(aspect_ratio - (best_ratio_match[0] / best_ratio_match[1]))

                if ratio_diff < 0.15:  # More forgiving
                    img_score += 25
                elif ratio_diff < 0.4:
                    img_score += 20
                elif ratio_diff < 0.8:
                    img_score += 15
                else:
                    img_issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
                    img_score += 10

            # Check if photographer/attribution is available
            if img.get("photographer") and img.get("photographer") != "Unknown":
                img_score += 20
            else:
                img_score += 10  # Partial credit instead of penalty

            # Check image source quality (more realistic)
            src_url = img.get("src", "")
            if any(qual in src_url.lower() for qual in ["large", "original", "high"]) or width > 1200:
                img_score += 20
            elif any(qual in src_url.lower() for qual in ["medium", "mid"]) or width > 800:
                img_score += 15
            else:
                img_score += 10  # Base score instead of penalty

            # Cap score at 100
            img_score = min(img_score, 100)
            scores.append(img_score)

            # Add to high quality if score > 80
            if img_score > 80:
                img_copy = img.copy()
                img_copy["quality_score"] = img_score
                high_quality_images.append(img_copy)

            if img_issues:
                issues.extend([f"Image {i+1}: {issue}" for issue in img_issues])

        # Calculate overall score
        overall_score = statistics.mean(scores) if scores else 0

        return {
            "score": round(overall_score, 1),
            "individual_scores": scores,
            "issues": issues,
            "recommendations": self._get_image_recommendations(overall_score),
            "high_quality_images": high_quality_images,
            "total_analyzed": len(images)
        }

    def analyze_text_quality_enhanced(self, text: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Enhanced text quality analysis with flexible keyword matching."""
        if not text:
            return {"score": 0, "issues": ["No text provided"], "recommendations": []}

        words = text.split()
        total_words = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        score = 0
        issues = []
        recommendations = []

        # Length scoring (more realistic)
        if total_words >= 300:
            score += 25
        elif total_words >= 150:
            score += 20
        elif total_words >= 50:
            score += 15
        else:
            issues.append(f"Text too short: {total_words} words")
            score += 5

        # Sentence structure
        if sentences:
            avg_sentence_length = total_words / len(sentences)
            if 10 <= avg_sentence_length <= 25:
                score += 20
            elif 8 <= avg_sentence_length <= 30:
                score += 15
            else:
                issues.append(f"Sentence length issues: avg {avg_sentence_length:.1f} words")
                score += 10

        # Character count and paragraphs
        char_count = len(text)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

        if char_count > 500:
            score += 15
        else:
            score += max(5, char_count // 100 * 2)

        # Keyword analysis (enhanced)
        if keywords:
            keyword_score = self._analyze_keywords_enhanced(text, keywords)
            score += keyword_score["score"]
            issues.extend(keyword_score["issues"])
            recommendations.extend(keyword_score["recommendations"])
        else:
            score += 15  # Neutral score if no keywords specified

        # Overall structure bonus
        if paragraph_count > 1:
            score += 10

        return {
            "score": min(score, 100),
            "word_count": total_words,
            "character_count": char_count,
            "sentence_count": len(sentences),
            "paragraph_count": paragraph_count,
            "avg_sentence_length": round(total_words / len(sentences), 1) if sentences else 0,
            "issues": issues,
            "recommendations": recommendations
        }

    def _analyze_keywords_enhanced(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """Enhanced keyword analysis with fuzzy matching and semantic understanding."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)

        keyword_analysis = {}
        issues = []
        recommendations = []
        score = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Exact match count
            exact_count = text_lower.count(keyword_lower)

            # Fuzzy match count (for slight variations)
            fuzzy_count = 0
            for word in words:
                similarity = SequenceMatcher(None, keyword_lower, word).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    fuzzy_count += 1

            # Combined count
            total_count = max(exact_count, fuzzy_count)
            density = total_count / total_words if total_words > 0 else 0

            keyword_analysis[keyword] = {
                "exact_count": exact_count,
                "fuzzy_count": fuzzy_count,
                "total_count": total_count,
                "density": round(density, 4)
            }

            # Scoring (more forgiving)
            if total_count > 0:
                if density <= self.quality_metrics["text_quality"]["keyword_density_max"]:
                    score += min(15, total_count * 5)  # Reward keyword presence
                else:
                    score += 10  # Reduced penalty for overuse
                    issues.append(f"Keyword '{keyword}' may be overused: {density:.2%}")
            else:
                # Check for partial matches or related terms
                keyword_words = keyword_lower.split()
                if any(word in text_lower for word in keyword_words):
                    score += 8  # Partial credit for related terms
                else:
                    issues.append(f"Missing keyword: '{keyword}'")
                    score += 2  # Small base score instead of zero

        if not keyword_analysis:
            recommendations.append("Include relevant keywords naturally")
        elif score < 20:
            recommendations.append("Consider adding more topic-relevant terms")

        return {
            "score": min(score, 40),  # Increased max score
            "keyword_analysis": keyword_analysis,
            "issues": issues,
            "recommendations": recommendations
        }

    def analyze_content_relevance_enhanced(self, content: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Enhanced content relevance analysis with better semantic understanding."""
        relevance_scores = {}
        overall_issues = []
        recommendations = []

        # Get expanded keywords for better matching
        expanded_keywords = self.get_expanded_keywords(topic)

        # Analyze script relevance
        if "script" in content:
            script_text = content["script"].get("script", "")
            script_relevance = self._calculate_text_relevance_enhanced(script_text, topic, expanded_keywords)
            relevance_scores["script"] = script_relevance

        # Analyze image relevance (enhanced)
        if "images" in content:
            images = content["images"]
            image_relevance = self._analyze_image_relevance_enhanced(images, topic, expanded_keywords)
            relevance_scores["images"] = image_relevance

        # Calculate overall relevance
        if relevance_scores:
            overall_relevance = statistics.mean(relevance_scores.values())
        else:
            overall_relevance = 0.0

        # Generate issues and recommendations
        if overall_relevance < 0.5:
            overall_issues.append("Content may not be well-matched to topic")
            recommendations.append("Search for more specific content related to the topic")

        if "images" in relevance_scores and relevance_scores["images"] < 0.3:
            overall_issues.append("Images may not be well-matched to topic")
            recommendations.append("Search for images with more specific keywords")

        quality_grade = self._get_quality_grade_enhanced(overall_relevance)

        return {
            "overall_relevance": round(overall_relevance, 3),
            "component_relevance": relevance_scores,
            "issues": overall_issues,
            "recommendations": recommendations,
            "quality_grade": quality_grade
        }

    def _calculate_text_relevance_enhanced(self, text: str, topic: str, expanded_keywords: List[str]) -> float:
        """Enhanced text relevance calculation with expanded keyword matching."""
        if not text:
            return 0.0

        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))

        # Score based on expanded keywords
        keyword_matches = 0
        for keyword in expanded_keywords:
            if keyword in text_lower:
                keyword_matches += 1
            else:
                # Check for fuzzy matches
                for word in text_words:
                    similarity = SequenceMatcher(None, keyword, word).ratio()
                    if similarity > 0.8:
                        keyword_matches += 0.8
                        break

        # Calculate relevance score
        if expanded_keywords:
            base_relevance = keyword_matches / len(expanded_keywords)
        else:
            base_relevance = 0.0

        # Boost score if exact topic appears
        if topic.lower() in text_lower:
            base_relevance += 0.3

        # Boost for topic words appearing as phrases
        topic_words = topic.lower().split()
        phrase_matches = sum(1 for word in topic_words if word in text_lower)
        if topic_words:
            base_relevance += (phrase_matches / len(topic_words)) * 0.2

        return min(base_relevance, 1.0)

    def _analyze_image_relevance_enhanced(self, images: List[Dict[str, Any]], topic: str, expanded_keywords: List[str]) -> float:
        """Enhanced image relevance analysis."""
        if not images:
            return 0.0

        relevance_scores = []

        for img in images:
            relevance = 0.0

            # Check all available text fields
            searchable_text = " ".join([
                str(img.get("url", "")),
                str(img.get("photographer", "")),
                str(img.get("alt", "")),
                str(img.get("tags", "")),
                str(img.get("description", ""))
            ]).lower()

            # Score based on expanded keywords
            for keyword in expanded_keywords:
                if keyword in searchable_text:
                    relevance += 0.1

            # Boost for exact topic match
            if topic.lower() in searchable_text:
                relevance += 0.3

            # Check for provider-specific quality indicators
            provider = img.get("provider", "").lower()
            if provider in ["pexels", "pixabay", "unsplash"]:
                relevance += 0.1  # Boost for professional sources

            relevance_scores.append(min(relevance, 1.0))

        return statistics.mean(relevance_scores) if relevance_scores else 0.0

    def _get_quality_grade_enhanced(self, score: float) -> str:
        """Convert numeric score to quality grade (more forgiving scale)."""
        if score >= 0.8:
            return "A+ (Excellent)"
        elif score >= 0.7:
            return "A (Very Good)"
        elif score >= 0.6:
            return "B+ (Good)"
        elif score >= 0.5:
            return "B (Fair)"
        elif score >= 0.4:
            return "C+ (Acceptable)"
        elif score >= 0.3:
            return "C (Needs Work)"
        elif score >= 0.2:
            return "D (Poor)"
        else:
            return "F (Needs Major Improvement)"

    def _get_image_recommendations(self, score: float) -> List[str]:
        """Get image quality recommendations based on score."""
        recommendations = []

        if score < 60:
            recommendations.append("Consider using higher resolution images")
            recommendations.append("Look for images from professional stock photo providers")
        elif score < 80:
            recommendations.append("Try to find images with better aspect ratios")
            recommendations.append("Ensure proper attribution for all images")

        return recommendations

    def analyze_content_bundle_enhanced(self, content: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Enhanced comprehensive content analysis."""
        analysis_results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "topic": topic,
            "components": {}
        }

        total_score = 0
        component_count = 0

        # Analyze script if available
        if "script" in content:
            script_analysis = self.analyze_text_quality_enhanced(
                content["script"].get("script", ""),
                self.get_expanded_keywords(topic)
            )
            analysis_results["components"]["script"] = script_analysis
            total_score += script_analysis["score"]
            component_count += 1

        # Analyze images if available
        if "images" in content:
            image_analysis = self.analyze_image_quality_enhanced(content["images"])
            analysis_results["components"]["images"] = image_analysis
            total_score += image_analysis["score"]
            component_count += 1

        # Analyze relevance
        relevance_analysis = self.analyze_content_relevance_enhanced(content, topic)
        analysis_results["components"]["relevance"] = relevance_analysis
        # Convert relevance score to 0-100 scale for overall scoring
        relevance_score = relevance_analysis["overall_relevance"] * 100
        total_score += relevance_score
        component_count += 1

        # Calculate overall score
        if component_count > 0:
            overall_score = total_score / component_count
        else:
            overall_score = 0

        analysis_results["overall_score"] = round(overall_score, 1)

        # Generate summary
        strengths = []
        weaknesses = []

        for component, data in analysis_results["components"].items():
            score = data.get("score", 0)
            if isinstance(score, (int, float)) and score > 70:
                strengths.append(f"Good {component} quality")
            elif isinstance(score, (int, float)) and score < 50:
                weaknesses.append(f"{component.title()} needs improvement")

        # Special handling for relevance
        if relevance_score > 70:        strengths.append("Good content relevance")
        elif relevance_score < 50:
            weaknesses.append("Content relevance needs improvement")

        analysis_results["summary"] = {
            "strengths": strengths if strengths else ["Basic content structure"],
            "weaknesses": weaknesses if weaknesses else ["Minor improvements needed"],
            "overall_assessment": self._get_overall_assessment(overall_score)
        }

        # Add overall grade for compatibility
        analysis_results["overall_grade"] = self._get_quality_grade_enhanced(overall_score / 100)

        return analysis_results

    def _get_overall_assessment(self, score: float) -> str:
        """Get overall assessment based on score."""
        if score >= 85:
            return "Excellent content ready for publication"
        elif score >= 75:
            return "Very good content with minor improvements needed"
        elif score >= 65:
            return "Good content that could benefit from some enhancements"
        elif score >= 55:
            return "Fair content requiring several improvements"
        elif score >= 45:
            return "Below average content needing significant work"
        else:
            return "Poor content requiring major improvements"    # Compatibility methods for original interface
    def analyze_image_quality(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compatibility wrapper for enhanced image quality analysis."""
        return self.analyze_image_quality_enhanced(images)

    def analyze_text_quality(self, text: str, target_keywords: List[str] = None) -> Dict[str, Any]:
        """Compatibility wrapper for enhanced text quality analysis."""
        return self.analyze_text_quality_enhanced(text, target_keywords)

    def analyze_content_relevance(self, content: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Compatibility wrapper for enhanced content relevance analysis."""
        return self.analyze_content_relevance_enhanced(content, topic)

    def generate_quality_report(self, content: Dict[str, Any], topic: str = None) -> Dict[str, Any]:
        """Generate a comprehensive quality report for content - compatible interface."""
        if topic is None:
            topic = "general content"
        return self.analyze_content_bundle_enhanced(content, topic)

    def filter_high_quality_content(self, content_list: List[Dict[str, Any]],
                                  quality_threshold: float = 70.0) -> List[Dict[str, Any]]:
        """Filter content to return only high-quality items - enhanced version."""
        high_quality_items = []

        for content in content_list:
            try:
                # Use a default topic if none available, or extract from content
                topic = content.get("topic", "general content")
                quality_report = self.generate_quality_report(content, topic)
                if quality_report["overall_score"] >= quality_threshold:
                    content["quality_score"] = quality_report["overall_score"]
                    content["quality_grade"] = quality_report.get("overall_grade", "Unknown")
                    high_quality_items.append(content)
            except Exception as e:
                logger.warning(f"Error analyzing content quality: {str(e)}")
                # Include content with unknown quality rather than exclude it
                content["quality_score"] = None
                content["quality_grade"] = "Unknown"
                high_quality_items.append(content)

        # Sort by quality score (highest first)
        high_quality_items.sort(
            key=lambda x: x.get("quality_score", 0) or 0,
            reverse=True
        )

        return high_quality_items

# Factory function for enhanced analyzer
def create_enhanced_quality_analyzer() -> EnhancedContentQualityAnalyzer:
    """Create an enhanced content quality analyzer instance."""
    return EnhancedContentQualityAnalyzer()
