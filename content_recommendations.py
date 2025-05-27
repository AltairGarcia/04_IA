"""
AI-Powered Content Recommendation Engine

This module provides intelligent content recommendations using machine learning
techniques, content analysis, and user behavior patterns.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import asyncio
import re

logger = logging.getLogger(__name__)

class ContentRecommendationEngine:
    """Advanced AI-powered content recommendation system."""

    def __init__(self, database_manager=None):
        """Initialize the recommendation engine.

        Args:
            database_manager: Database manager instance for historical data
        """
        self.database_manager = database_manager
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.content_vectors = None
        self.content_data = []
        self.user_preferences = {}
        self.trend_patterns = {}

        # Content categories for better recommendations
        self.content_categories = {
            "educational": ["tutorial", "guide", "how-to", "learn", "course"],
            "entertainment": ["fun", "funny", "viral", "trending", "meme"],
            "business": ["strategy", "marketing", "sales", "growth", "profit"],
            "technology": ["ai", "tech", "software", "digital", "innovation"],
            "lifestyle": ["health", "fitness", "travel", "food", "fashion"],
            "news": ["breaking", "latest", "update", "current", "news"]
        }

    def analyze_content_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content trends from historical data.

        Args:
            historical_data: Historical content creation data

        Returns:
            Trend analysis results
        """
        try:
            if not historical_data:
                return {"trends": [], "patterns": {}, "insights": []}

            # Extract topics and performance metrics
            topics = []
            performance_scores = []
            creation_dates = []
            tags_all = []

            for record in historical_data:
                topics.append(record.get('topic', ''))
                performance_scores.append(record.get('quality_score', 0))
                creation_dates.append(
                    datetime.fromisoformat(record.get('created_at', datetime.now().isoformat()))
                )
                tags_all.extend(record.get('tags', []))

            # Analyze topic popularity
            topic_frequency = Counter(topics)
            popular_topics = topic_frequency.most_common(10)

            # Analyze performance patterns
            avg_performance = np.mean(performance_scores) if performance_scores else 0
            high_performers = [
                topic for topic, score in zip(topics, performance_scores)
                if score > avg_performance
            ]

            # Analyze tag trends
            tag_frequency = Counter(tags_all)
            trending_tags = tag_frequency.most_common(15)

            # Time-based analysis
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_topics = [
                topic for topic, date in zip(topics, creation_dates)
                if date > recent_cutoff
            ]
            recent_trend = Counter(recent_topics).most_common(5)

            # Category analysis
            category_performance = defaultdict(list)
            for topic, score in zip(topics, performance_scores):
                category = self._categorize_content(topic)
                category_performance[category].append(score)

            category_avg = {
                cat: np.mean(scores) if scores else 0
                for cat, scores in category_performance.items()
            }

            return {
                "trends": {
                    "popular_topics": popular_topics,
                    "recent_trending": recent_trend,
                    "trending_tags": trending_tags,
                    "high_performing_topics": Counter(high_performers).most_common(5)
                },
                "patterns": {
                    "avg_performance": avg_performance,
                    "category_performance": dict(category_avg),
                    "total_content_pieces": len(topics),
                    "unique_topics": len(set(topics))
                },
                "insights": self._generate_trend_insights(
                    popular_topics, category_avg, recent_trend
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing content trends: {str(e)}")
            return {"trends": [], "patterns": {}, "insights": []}

    def _categorize_content(self, topic: str) -> str:
        """Categorize content based on topic keywords.

        Args:
            topic: Content topic

        Returns:
            Content category
        """
        topic_lower = topic.lower()

        for category, keywords in self.content_categories.items():
            if any(keyword in topic_lower for keyword in keywords):
                return category

        return "general"

    def _generate_trend_insights(self, popular_topics: List[Tuple[str, int]],
                               category_performance: Dict[str, float],
                               recent_trends: List[Tuple[str, int]]) -> List[str]:
        """Generate actionable insights from trend analysis.

        Args:
            popular_topics: Most popular topics
            category_performance: Performance by category
            recent_trends: Recent trending topics

        Returns:
            List of insights
        """
        insights = []

        # Popular topic insights
        if popular_topics:
            top_topic = popular_topics[0][0]
            insights.append(f"'{top_topic}' is your most created content topic")

        # Category performance insights
        if category_performance:
            best_category = max(category_performance.items(), key=lambda x: x[1])
            worst_category = min(category_performance.items(), key=lambda x: x[1])

            if best_category[1] > 70:
                insights.append(f"{best_category[0].title()} content performs best (avg: {best_category[1]:.1f})")

            if worst_category[1] < 50:
                insights.append(f"{worst_category[0].title()} content needs improvement (avg: {worst_category[1]:.1f})")

        # Recent trend insights
        if recent_trends:
            recent_topic = recent_trends[0][0]
            insights.append(f"'{recent_topic}' is trending recently - consider expanding on it")

        # General insights
        insights.extend([
            "Consider creating content series around your best-performing topics",
            "Diversify content categories to reach broader audiences",
            "Monitor trending topics for timely content opportunities"
        ])

        return insights

    def generate_content_recommendations(self, user_input: str = None,
                                       count: int = 8,
                                       include_analysis: bool = True) -> List[Dict[str, Any]]:
        """Generate intelligent content recommendations.

        Args:
            user_input: User's input topic or preference
            count: Number of recommendations to generate
            include_analysis: Whether to include detailed analysis

        Returns:
            List of content recommendations
        """
        try:
            recommendations = []

            # Get historical data for analysis
            historical_data = []
            if self.database_manager:
                historical_data = self.database_manager.get_content_history(limit=100)

            # Analyze trends
            trend_analysis = self.analyze_content_trends(historical_data)

            # Base recommendations from different strategies
            strategies = [
                self._trending_based_recommendations,
                self._performance_based_recommendations,
                self._category_based_recommendations,
                self._ai_generated_recommendations
            ]

            # Generate recommendations from each strategy
            for strategy in strategies:
                try:
                    strategy_recs = strategy(user_input, trend_analysis, count // len(strategies))
                    recommendations.extend(strategy_recs)
                except Exception as e:
                    logger.warning(f"Strategy failed: {str(e)}")
                    continue

            # Remove duplicates and enhance recommendations
            unique_recs = self._deduplicate_recommendations(recommendations)
            enhanced_recs = self._enhance_recommendations(unique_recs, include_analysis)

            # Sort by relevance score
            enhanced_recs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            return enhanced_recs[:count]

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._fallback_recommendations(count)

    def _trending_based_recommendations(self, user_input: str,
                                      trend_analysis: Dict[str, Any],
                                      count: int) -> List[Dict[str, Any]]:
        """Generate recommendations based on trending topics.

        Args:
            user_input: User input
            trend_analysis: Trend analysis data
            count: Number of recommendations

        Returns:
            Trending-based recommendations
        """
        recommendations = []
        trending_topics = trend_analysis.get('trends', {}).get('recent_trending', [])

        for topic, frequency in trending_topics[:count]:
            recommendations.append({
                "title": f"Deep Dive: {topic}",
                "description": f"Expand on the trending topic '{topic}' with detailed analysis",
                "category": self._categorize_content(topic),
                "strategy": "trending",
                "base_topic": topic,
                "trend_score": frequency,
                "relevance_score": 85 + min(frequency * 5, 15)
            })

        return recommendations

    def _performance_based_recommendations(self, user_input: str,
                                         trend_analysis: Dict[str, Any],
                                         count: int) -> List[Dict[str, Any]]:
        """Generate recommendations based on high-performing content.

        Args:
            user_input: User input
            trend_analysis: Trend analysis data
            count: Number of recommendations

        Returns:
            Performance-based recommendations
        """
        recommendations = []
        high_performers = trend_analysis.get('trends', {}).get('high_performing_topics', [])

        variations = ["Advanced Guide", "Beginner's Introduction", "Common Mistakes", "Best Practices"]

        for i, (topic, score) in enumerate(high_performers[:count]):
            variation = variations[i % len(variations)]
            recommendations.append({
                "title": f"{variation}: {topic}",
                "description": f"Build on your successful '{topic}' content with a {variation.lower()} approach",
                "category": self._categorize_content(topic),
                "strategy": "performance",
                "base_topic": topic,
                "performance_score": score,
                "relevance_score": 80 + min(score * 2, 20)
            })

        return recommendations

    def _category_based_recommendations(self, user_input: str,
                                      trend_analysis: Dict[str, Any],
                                      count: int) -> List[Dict[str, Any]]:
        """Generate recommendations based on category analysis.

        Args:
            user_input: User input
            trend_analysis: Trend analysis data
            count: Number of recommendations

        Returns:
            Category-based recommendations
        """
        recommendations = []
        category_performance = trend_analysis.get('patterns', {}).get('category_performance', {})

        # Focus on underperforming categories for improvement
        sorted_categories = sorted(category_performance.items(), key=lambda x: x[1])

        category_topics = {
            "educational": ["Complete Tutorial", "Step-by-Step Guide", "Expert Tips"],
            "technology": ["Latest Innovations", "Future Trends", "Implementation Guide"],
            "business": ["Growth Strategies", "Success Stories", "Market Analysis"],
            "lifestyle": ["Daily Habits", "Life Hacks", "Wellness Tips"]
        }

        for category, avg_score in sorted_categories[:count]:
            if category in category_topics:
                topic_variations = category_topics[category]
                selected_topic = topic_variations[len(recommendations) % len(topic_variations)]

                recommendations.append({
                    "title": f"{selected_topic} in {category.title()}",
                    "description": f"Improve your {category} content performance with {selected_topic.lower()}",
                    "category": category,
                    "strategy": "category_improvement",
                    "current_performance": avg_score,
                    "improvement_potential": max(0, 80 - avg_score),
                    "relevance_score": 75 + (80 - avg_score) * 0.5
                })

        return recommendations

    def _ai_generated_recommendations(self, user_input: str,
                                    trend_analysis: Dict[str, Any],
                                    count: int) -> List[Dict[str, Any]]:
        """Generate AI-powered content recommendations.

        Args:
            user_input: User input
            trend_analysis: Trend analysis data
            count: Number of recommendations

        Returns:
            AI-generated recommendations
        """
        recommendations = []

        # AI-generated innovative content ideas
        ai_topics = [
            {
                "title": "The Future of Content Creation with AI",
                "description": "Explore how AI is revolutionizing content creation workflows",
                "category": "technology",
                "innovation_score": 95
            },
            {
                "title": "Data-Driven Content Strategy",
                "description": "Use analytics to optimize your content performance",
                "category": "business",
                "innovation_score": 88
            },
            {
                "title": "Interactive Content Experiences",
                "description": "Create engaging interactive content that captivates audiences",
                "category": "entertainment",
                "innovation_score": 92
            },
            {
                "title": "Sustainable Content Production",
                "description": "Build efficient and sustainable content creation processes",
                "category": "lifestyle",
                "innovation_score": 85
            }
        ]

        for i, topic_data in enumerate(ai_topics[:count]):
            recommendations.append({
                **topic_data,
                "strategy": "ai_innovation",
                "relevance_score": topic_data["innovation_score"],
                "future_potential": "high"
            })

        return recommendations

    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations based on similarity.

        Args:
            recommendations: List of recommendations

        Returns:
            Deduplicated recommendations
        """
        if not recommendations:
            return []

        unique_recs = []
        seen_titles = set()

        for rec in recommendations:
            title = rec.get('title', '').lower()

            # Check for exact duplicates
            if title in seen_titles:
                continue

            # Check for similar titles (simple similarity)
            is_similar = False
            for seen_title in seen_titles:
                if self._calculate_title_similarity(title, seen_title) > 0.8:
                    is_similar = True
                    break

            if not is_similar:
                unique_recs.append(rec)
                seen_titles.add(title)

        return unique_recs

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles.

        Args:
            title1: First title
            title2: Second title

        Returns:
            Similarity score (0-1)
        """
        # Simple word-based similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _enhance_recommendations(self, recommendations: List[Dict[str, Any]],
                               include_analysis: bool) -> List[Dict[str, Any]]:
        """Enhance recommendations with additional metadata.

        Args:
            recommendations: List of recommendations
            include_analysis: Whether to include detailed analysis

        Returns:
            Enhanced recommendations
        """
        enhanced = []

        for rec in recommendations:
            enhanced_rec = rec.copy()

            # Add metadata
            enhanced_rec.update({
                "generated_at": datetime.now().isoformat(),
                "estimated_duration": self._estimate_content_duration(rec),
                "difficulty_level": self._assess_difficulty_level(rec),
                "target_audience": self._identify_target_audience(rec),
                "content_format": self._suggest_content_format(rec)
            })

            if include_analysis:
                enhanced_rec.update({
                    "keyword_suggestions": self._generate_keywords(rec),
                    "content_structure": self._suggest_content_structure(rec),
                    "engagement_prediction": self._predict_engagement(rec)
                })

            enhanced.append(enhanced_rec)

        return enhanced

    def _estimate_content_duration(self, recommendation: Dict[str, Any]) -> str:
        """Estimate content creation duration.

        Args:
            recommendation: Recommendation data

        Returns:
            Estimated duration
        """
        category = recommendation.get('category', 'general')

        duration_map = {
            "educational": "4-6 minutes",
            "technology": "5-8 minutes",
            "business": "3-5 minutes",
            "entertainment": "2-4 minutes",
            "lifestyle": "3-5 minutes",
            "general": "3-5 minutes"
        }

        return duration_map.get(category, "3-5 minutes")

    def _assess_difficulty_level(self, recommendation: Dict[str, Any]) -> str:
        """Assess content difficulty level.

        Args:
            recommendation: Recommendation data

        Returns:
            Difficulty level
        """
        title = recommendation.get('title', '').lower()

        if any(word in title for word in ['beginner', 'introduction', 'basic']):
            return "Beginner"
        elif any(word in title for word in ['advanced', 'expert', 'deep dive']):
            return "Advanced"
        else:
            return "Intermediate"

    def _identify_target_audience(self, recommendation: Dict[str, Any]) -> str:
        """Identify target audience for content.

        Args:
            recommendation: Recommendation data

        Returns:
            Target audience description
        """
        category = recommendation.get('category', 'general')
        difficulty = self._assess_difficulty_level(recommendation)

        audience_map = {
            "educational": f"{difficulty} learners and students",
            "technology": f"Tech professionals and {difficulty.lower()} developers",
            "business": f"Business professionals and {difficulty.lower()} entrepreneurs",
            "entertainment": "General audience seeking entertainment",
            "lifestyle": f"Lifestyle enthusiasts and {difficulty.lower()} practitioners"
        }

        return audience_map.get(category, "General audience")

    def _suggest_content_format(self, recommendation: Dict[str, Any]) -> List[str]:
        """Suggest optimal content formats.

        Args:
            recommendation: Recommendation data

        Returns:
            List of suggested formats
        """
        category = recommendation.get('category', 'general')

        format_map = {
            "educational": ["Tutorial Video", "Step-by-step Guide", "Interactive Workshop"],
            "technology": ["Demo Video", "Technical Documentation", "Code Walkthrough"],
            "business": ["Case Study", "Strategy Presentation", "Interview"],
            "entertainment": ["Short Video", "Interactive Content", "Story-driven Content"],
            "lifestyle": ["How-to Video", "Personal Story", "Tips & Tricks"]
        }

        return format_map.get(category, ["Article", "Video", "Infographic"])

    def _generate_keywords(self, recommendation: Dict[str, Any]) -> List[str]:
        """Generate relevant keywords for content.

        Args:
            recommendation: Recommendation data

        Returns:
            List of keywords
        """
        title = recommendation.get('title', '')
        category = recommendation.get('category', 'general')

        # Extract keywords from title
        title_words = re.findall(r'\b\w+\b', title.lower())

        # Add category-specific keywords
        category_keywords = {
            "educational": ["tutorial", "learn", "guide", "course", "training"],
            "technology": ["tech", "digital", "innovation", "software", "AI"],
            "business": ["strategy", "growth", "marketing", "success", "profit"],
            "entertainment": ["fun", "engaging", "viral", "trending", "popular"],
            "lifestyle": ["tips", "health", "wellness", "personal", "improvement"]
        }

        keywords = list(set(title_words + category_keywords.get(category, [])))
        return keywords[:10]  # Limit to 10 keywords

    def _suggest_content_structure(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest content structure.

        Args:
            recommendation: Recommendation data

        Returns:
            Content structure suggestions
        """
        difficulty = self._assess_difficulty_level(recommendation)
        category = recommendation.get('category', 'general')

        if difficulty == "Beginner":
            structure = {
                "introduction": "Hook with relatable problem",
                "main_sections": ["Basic concepts", "Step-by-step process", "Common mistakes"],
                "conclusion": "Summary and next steps",
                "estimated_sections": 4
            }
        elif difficulty == "Advanced":
            structure = {
                "introduction": "Context and advanced applications",
                "main_sections": ["Deep technical details", "Advanced techniques", "Expert insights"],
                "conclusion": "Advanced applications and future considerations",
                "estimated_sections": 6
            }
        else:
            structure = {
                "introduction": "Problem overview and solution preview",
                "main_sections": ["Core concepts", "Practical examples", "Implementation tips"],
                "conclusion": "Recap and actionable takeaways",
                "estimated_sections": 5
            }

        return structure

    def _predict_engagement(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content engagement potential.

        Args:
            recommendation: Recommendation data

        Returns:
            Engagement prediction
        """
        relevance_score = recommendation.get('relevance_score', 50)
        category = recommendation.get('category', 'general')

        # Base engagement on relevance and category performance
        engagement_multipliers = {
            "entertainment": 1.3,
            "educational": 1.1,
            "technology": 1.2,
            "business": 1.0,
            "lifestyle": 1.15
        }

        multiplier = engagement_multipliers.get(category, 1.0)
        predicted_score = min(100, relevance_score * multiplier)

        if predicted_score >= 85:
            engagement_level = "High"
        elif predicted_score >= 70:
            engagement_level = "Medium-High"
        elif predicted_score >= 55:
            engagement_level = "Medium"
        else:
            engagement_level = "Low-Medium"

        return {
            "predicted_score": round(predicted_score),
            "engagement_level": engagement_level,
            "confidence": "High" if relevance_score > 80 else "Medium"
        }

    def _fallback_recommendations(self, count: int) -> List[Dict[str, Any]]:
        """Generate fallback recommendations when other methods fail.

        Args:
            count: Number of recommendations needed

        Returns:
            Fallback recommendations
        """
        fallback_topics = [
            {
                "title": "Content Creation Best Practices",
                "description": "Essential guidelines for creating high-quality content",
                "category": "educational"
            },
            {
                "title": "Trending Topics in Digital Marketing",
                "description": "Explore the latest trends shaping digital marketing",
                "category": "business"
            },
            {
                "title": "AI Tools for Content Creators",
                "description": "Discover AI-powered tools to enhance your content workflow",
                "category": "technology"
            },
            {
                "title": "Building Your Personal Brand",
                "description": "Strategies for establishing a strong personal brand online",
                "category": "lifestyle"
            }
        ]

        recommendations = []
        for i, topic in enumerate(fallback_topics[:count]):
            recommendations.append({
                **topic,
                "strategy": "fallback",
                "relevance_score": 60 + (i * 5),
                "generated_at": datetime.now().isoformat(),
                "estimated_duration": "3-5 minutes",
                "difficulty_level": "Intermediate"
            })

        return recommendations

    def update_user_feedback(self, recommendation_id: str, feedback: str, rating: int):
        """Update recommendation system with user feedback.

        Args:
            recommendation_id: ID of the recommendation
            feedback: User feedback text
            rating: User rating (1-5)
        """
        try:
            # Store feedback for future improvement
            if self.database_manager:
                self.database_manager.save_user_preference(
                    f"recommendation_feedback_{recommendation_id}",
                    {
                        "feedback": feedback,
                        "rating": rating,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            logger.info(f"User feedback recorded for recommendation {recommendation_id}")

        except Exception as e:
            logger.error(f"Failed to save user feedback: {str(e)}")


# Factory function
def create_recommendation_engine(database_manager=None) -> ContentRecommendationEngine:
    """Create content recommendation engine instance.

    Args:
        database_manager: Database manager instance

    Returns:
        ContentRecommendationEngine instance
    """
    return ContentRecommendationEngine(database_manager)
