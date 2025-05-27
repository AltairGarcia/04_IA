"""
Content Quality Enhancement Module

This module provides advanced content quality analysis, filtering,
and enhancement capabilities for the content creation system.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import statistics

logger = logging.getLogger(__name__)

class ContentQualityAnalyzer:
    """Advanced content quality analysis and enhancement."""

    def __init__(self):
        """Initialize the content quality analyzer."""
        self.quality_metrics = {
            "image_quality": {
                "min_resolution": 800,  # Minimum width/height
                "preferred_aspect_ratios": [(16, 9), (4, 3), (1, 1)],
                "max_file_size_mb": 10
            },
            "text_quality": {
                "min_length": 100,
                "max_length": 5000,
                "readability_target": 8.0,  # Grade level
                "keyword_density_max": 0.03
            },
            "content_relevance": {
                "min_relevance_score": 0.7,
                "semantic_similarity_threshold": 0.6
            }
        }

    def analyze_image_quality(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and score image quality.

        Args:
            images: List of image dictionaries

        Returns:
            Quality analysis results
        """
        if not images:
            return {"score": 0, "issues": ["No images provided"], "recommendations": []}

        scores = []
        issues = []
        high_quality_images = []

        for i, img in enumerate(images):
            img_score = 0
            img_issues = []

            # Check resolution
            width = img.get("width", 0)
            height = img.get("height", 0)

            if width >= self.quality_metrics["image_quality"]["min_resolution"] and \
               height >= self.quality_metrics["image_quality"]["min_resolution"]:
                img_score += 30
            else:
                img_issues.append(f"Low resolution: {width}x{height}")

            # Check aspect ratio
            if width > 0 and height > 0:
                aspect_ratio = width / height
                best_ratio_match = min(
                    self.quality_metrics["image_quality"]["preferred_aspect_ratios"],
                    key=lambda r: abs(aspect_ratio - (r[0] / r[1]))
                )
                ratio_diff = abs(aspect_ratio - (best_ratio_match[0] / best_ratio_match[1]))

                if ratio_diff < 0.1:
                    img_score += 25
                elif ratio_diff < 0.3:
                    img_score += 15
                else:
                    img_issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")

            # Check if photographer/attribution is available
            if img.get("photographer") and img.get("photographer") != "Unknown":
                img_score += 20
            else:
                img_issues.append("Missing photographer attribution")

            # Check image source quality
            src_url = img.get("src", "")
            if "large" in src_url or "original" in src_url or width > 1200:
                img_score += 25
            elif "medium" in src_url or width > 800:
                img_score += 15
            else:
                img_issues.append("Low quality source image")

            scores.append(img_score)

            if img_score >= 70:  # High quality threshold
                high_quality_images.append({**img, "quality_score": img_score})

            if img_issues:
                issues.append(f"Image {i+1}: {', '.join(img_issues)}")

        overall_score = statistics.mean(scores) if scores else 0

        recommendations = []
        if overall_score < 50:
            recommendations.append("Consider using higher resolution images (800px+)")
        if overall_score < 70:
            recommendations.append("Look for images with proper attribution")
            recommendations.append("Prefer standard aspect ratios (16:9, 4:3, 1:1)")

        return {
            "score": round(overall_score, 1),
            "individual_scores": scores,
            "issues": issues,
            "recommendations": recommendations,
            "high_quality_images": high_quality_images,
            "total_analyzed": len(images)
        }

    def analyze_text_quality(self, text: str, target_keywords: List[str] = None) -> Dict[str, Any]:
        """Analyze text content quality.

        Args:
            text: Text content to analyze
            target_keywords: Optional list of target keywords

        Returns:
            Text quality analysis
        """
        if not text:
            return {"score": 0, "issues": ["No text provided"], "recommendations": []}

        issues = []
        recommendations = []
        score = 0

        # Length analysis
        word_count = len(text.split())
        char_count = len(text)

        if self.quality_metrics["text_quality"]["min_length"] <= char_count <= self.quality_metrics["text_quality"]["max_length"]:
            score += 25
        elif char_count < self.quality_metrics["text_quality"]["min_length"]:
            issues.append(f"Text too short: {char_count} characters")
            recommendations.append("Add more detailed content")
        else:
            issues.append(f"Text too long: {char_count} characters")
            recommendations.append("Consider breaking into shorter sections")

        # Readability analysis (simplified)
        sentences = text.split('.')
        avg_sentence_length = word_count / max(len(sentences), 1)

        if 10 <= avg_sentence_length <= 20:
            score += 25
        elif avg_sentence_length > 25:
            issues.append("Sentences too long")
            recommendations.append("Break down complex sentences")
        elif avg_sentence_length < 8:
            issues.append("Sentences too short")
            recommendations.append("Add more detail to sentences")

        # Structure analysis
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 3:
            score += 20
        else:
            recommendations.append("Add more paragraph breaks for better readability")

        # Keyword analysis
        if target_keywords:
            keyword_scores = self._analyze_keywords(text, target_keywords)
            score += min(keyword_scores["score"], 30)
            issues.extend(keyword_scores["issues"])
            recommendations.extend(keyword_scores["recommendations"])
        else:
            score += 15  # Bonus for having content even without keyword targets

        return {
            "score": min(score, 100),
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "issues": issues,
            "recommendations": recommendations
        }

    def _analyze_keywords(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze keyword usage in text."""
        text_lower = text.lower()
        total_words = len(text.split())

        keyword_analysis = {}
        issues = []
        recommendations = []
        score = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = text_lower.count(keyword_lower)
            density = count / total_words if total_words > 0 else 0

            keyword_analysis[keyword] = {
                "count": count,
                "density": round(density, 4)
            }

            if count == 0:
                issues.append(f"Missing keyword: '{keyword}'")
            elif density > self.quality_metrics["text_quality"]["keyword_density_max"]:
                issues.append(f"Keyword '{keyword}' overused: {density:.2%}")
            else:
                score += 10  # Points for appropriate keyword usage

        if not keyword_analysis:
            recommendations.append("Include relevant keywords naturally")

        return {
            "score": min(score, 30),
            "keyword_analysis": keyword_analysis,
            "issues": issues,
            "recommendations": recommendations
        }

    def analyze_content_relevance(self, content: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Analyze how relevant content is to the given topic.

        Args:
            content: Content dictionary (script, images, etc.)
            topic: Target topic

        Returns:
            Relevance analysis
        """
        relevance_scores = {}
        overall_issues = []
        recommendations = []

        # Analyze script relevance
        if "script" in content:
            script_text = content["script"].get("script", "")
            script_relevance = self._calculate_text_relevance(script_text, topic)
            relevance_scores["script"] = script_relevance

            if script_relevance < 0.7:
                overall_issues.append("Script content may not be closely related to topic")
                recommendations.append("Ensure script content directly addresses the topic")

        # Analyze image relevance
        if "images" in content:
            images = content["images"]
            if isinstance(images, dict) and "aggregated_results" in images:
                image_relevance = self._analyze_image_relevance(images["aggregated_results"], topic)
                relevance_scores["images"] = image_relevance

                if image_relevance < 0.6:
                    overall_issues.append("Images may not be well-matched to topic")
                    recommendations.append("Search for more specific images related to the topic")

        # Calculate overall relevance
        if relevance_scores:
            overall_relevance = statistics.mean(relevance_scores.values())
        else:
            overall_relevance = 0

        return {
            "overall_relevance": round(overall_relevance, 3),
            "component_relevance": relevance_scores,
            "issues": overall_issues,
            "recommendations": recommendations,
            "quality_grade": self._get_quality_grade(overall_relevance)
        }

    def _calculate_text_relevance(self, text: str, topic: str) -> float:
        """Calculate how relevant text is to a topic (simplified)."""
        if not text or not topic:
            return 0.0

        text_lower = text.lower()
        topic_lower = topic.lower()

        # Simple keyword matching approach
        topic_words = set(topic_lower.split())
        text_words = set(text_lower.split())

        # Calculate overlap
        common_words = topic_words.intersection(text_words)
        if not topic_words:
            return 0.0

        word_overlap = len(common_words) / len(topic_words)

        # Boost score if topic appears as a phrase
        if topic_lower in text_lower:
            word_overlap += 0.3

        return min(word_overlap, 1.0)

    def _analyze_image_relevance(self, images: List[Dict[str, Any]], topic: str) -> float:
        """Analyze how relevant images are to the topic."""
        if not images:
            return 0.0

        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        relevance_scores = []

        for img in images:
            # Check image metadata for relevance
            metadata_text = " ".join([
                str(img.get("url", "")),
                str(img.get("photographer", "")),
                # Could include alt text or tags if available
            ]).lower()

            metadata_words = set(metadata_text.split())
            common_words = topic_words.intersection(metadata_words)

            if topic_words:
                relevance = len(common_words) / len(topic_words)
            else:
                relevance = 0.0

            relevance_scores.append(relevance)

        return statistics.mean(relevance_scores) if relevance_scores else 0.0

    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to quality grade."""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Fair)"
        elif score >= 0.5:
            return "D (Poor)"
        else:
            return "F (Needs Improvement)"

    def generate_quality_report(self, content: Dict[str, Any], topic: str = None) -> Dict[str, Any]:
        """Generate a comprehensive quality report for content.

        Args:
            content: Content bundle to analyze
            topic: Optional topic for relevance analysis

        Returns:
            Comprehensive quality report
        """
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "topic": topic,
            "overall_score": 0,
            "components": {},
            "summary": {
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }
        }

        component_scores = []

        # Analyze script if present
        if "script" in content:
            script_analysis = self.analyze_text_quality(
                content["script"].get("script", ""),
                target_keywords=[topic] if topic else None
            )
            report["components"]["script"] = script_analysis
            component_scores.append(script_analysis["score"])

            if script_analysis["score"] >= 75:
                report["summary"]["strengths"].append("High-quality script content")
            else:
                report["summary"]["weaknesses"].append("Script needs improvement")

            report["summary"]["recommendations"].extend(script_analysis["recommendations"])

        # Analyze images if present
        if "images" in content:
            images_data = content["images"]
            if isinstance(images_data, dict) and "aggregated_results" in images_data:
                image_analysis = self.analyze_image_quality(images_data["aggregated_results"])
                report["components"]["images"] = image_analysis
                component_scores.append(image_analysis["score"])

                if image_analysis["score"] >= 70:
                    report["summary"]["strengths"].append("Good image quality")
                else:
                    report["summary"]["weaknesses"].append("Image quality could be better")

                report["summary"]["recommendations"].extend(image_analysis["recommendations"])

        # Analyze content relevance if topic provided
        if topic:
            relevance_analysis = self.analyze_content_relevance(content, topic)
            report["components"]["relevance"] = relevance_analysis
            component_scores.append(relevance_analysis["overall_relevance"] * 100)

            if relevance_analysis["overall_relevance"] >= 0.7:
                report["summary"]["strengths"].append("Content well-aligned with topic")
            else:
                report["summary"]["weaknesses"].append("Content relevance needs improvement")

            report["summary"]["recommendations"].extend(relevance_analysis["recommendations"])

        # Calculate overall score
        if component_scores:
            report["overall_score"] = round(statistics.mean(component_scores), 1)

        # Add overall grade
        report["overall_grade"] = self._get_quality_grade(report["overall_score"] / 100)

        return report

    def filter_high_quality_content(self, content_list: List[Dict[str, Any]],
                                  quality_threshold: float = 70.0) -> List[Dict[str, Any]]:
        """Filter content to return only high-quality items.

        Args:
            content_list: List of content items to filter
            quality_threshold: Minimum quality score (0-100)

        Returns:
            Filtered list of high-quality content
        """
        high_quality_items = []

        for content in content_list:
            try:
                quality_report = self.generate_quality_report(content)
                if quality_report["overall_score"] >= quality_threshold:
                    content["quality_score"] = quality_report["overall_score"]
                    content["quality_grade"] = quality_report["overall_grade"]
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

    def suggest_quality_improvements(self, content: Dict[str, Any], topic: str = None) -> Dict[str, Any]:
        """Suggest specific improvements for content quality.

        Args:
            content: Content to analyze
            topic: Optional topic for context

        Returns:
            Detailed improvement suggestions
        """
        quality_report = self.generate_quality_report(content, topic)

        improvements = {
            "priority_improvements": [],
            "quick_fixes": [],
            "advanced_enhancements": [],
            "estimated_impact": {}
        }

        # Prioritize improvements based on component scores
        if "script" in quality_report["components"]:
            script_score = quality_report["components"]["script"]["score"]
            if script_score < 50:
                improvements["priority_improvements"].append({
                    "component": "script",
                    "issue": "Low script quality",
                    "suggestion": "Rewrite script with better structure and content",
                    "estimated_improvement": "+30-40 points"
                })
            elif script_score < 75:
                improvements["quick_fixes"].append({
                    "component": "script",
                    "issue": "Script needs refinement",
                    "suggestion": "Improve sentence structure and add more detail",
                    "estimated_improvement": "+10-20 points"
                })

        if "images" in quality_report["components"]:
            image_score = quality_report["components"]["images"]["score"]
            if image_score < 50:
                improvements["priority_improvements"].append({
                    "component": "images",
                    "issue": "Poor image quality",
                    "suggestion": "Search for higher resolution images with better composition",
                    "estimated_improvement": "+25-35 points"
                })
            elif image_score < 70:
                improvements["quick_fixes"].append({
                    "component": "images",
                    "issue": "Image quality could be better",
                    "suggestion": "Filter for high-resolution images with proper attribution",
                    "estimated_improvement": "+10-15 points"
                })

        # Advanced enhancements
        improvements["advanced_enhancements"] = [
            {
                "enhancement": "SEO optimization",
                "description": "Add meta descriptions and optimize keyword placement",
                "estimated_improvement": "+5-10 points"
            },
            {
                "enhancement": "Multimedia integration",
                "description": "Add videos or interactive elements",
                "estimated_improvement": "+10-15 points"
            },
            {
                "enhancement": "Accessibility improvements",
                "description": "Add alt text and improve readability",
                "estimated_improvement": "+5-8 points"
            }
        ]

        return improvements

# Factory function
def create_quality_analyzer() -> ContentQualityAnalyzer:
    """Create a content quality analyzer instance."""
    return ContentQualityAnalyzer()
