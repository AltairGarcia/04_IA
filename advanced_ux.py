"""
Enhanced User Experience Module

This module provides advanced UX features including real-time previews,
interactive content editing, collaboration tools, and personalized interfaces.
"""

import streamlit as st
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)

class AdvancedUXManager:
    """Advanced user experience management system."""

    def __init__(self, database_manager=None, recommendation_engine=None):
        """Initialize UX manager.

        Args:
            database_manager: Database manager instance
            recommendation_engine: Recommendation engine instance
        """
        self.database_manager = database_manager
        self.recommendation_engine = recommendation_engine
        self.user_preferences = {}

    def render_interactive_content_editor(self, content_data: Dict[str, Any] = None):
        """Render interactive content editor with real-time preview.

        Args:
            content_data: Existing content data to edit
        """
        st.header("✏️ Interactive Content Editor")

        # Initialize content if not provided
        if content_data is None:
            content_data = {
                "title": "",
                "description": "",
                "script": "",
                "tags": [],
                "metadata": {}
            }

        # Create tabs for different editing modes
        tab1, tab2, tab3, tab4 = st.tabs(["✏️ Edit", "👁️ Preview", "🏷️ Metadata", "📊 Analytics"])

        with tab1:
            self._render_content_editing_interface(content_data)

        with tab2:
            self._render_content_preview(content_data)

        with tab3:
            self._render_metadata_editor(content_data)

        with tab4:
            self._render_content_analytics(content_data)

    def _render_content_editing_interface(self, content_data: Dict[str, Any]):
        """Render the main content editing interface."""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📝 Content Editor")

            # Title editing
            new_title = st.text_input(
                "Title",
                value=content_data.get("title", ""),
                key="editor_title",
                help="Enter your content title"
            )
            content_data["title"] = new_title

            # Description editing
            new_description = st.text_area(
                "Description",
                value=content_data.get("description", ""),
                height=100,
                key="editor_description",
                help="Provide a brief description of your content"
            )
            content_data["description"] = new_description

            # Script/Content editing with advanced features
            st.write("**Script/Content:**")

            # Editing toolbar
            col_format1, col_format2, col_format3, col_format4 = st.columns(4)

            with col_format1:
                if st.button("🔤 Add Heading"):
                    content_data["script"] = content_data.get("script", "") + "\n## New Heading\n"

            with col_format2:
                if st.button("📝 Add Paragraph"):
                    content_data["script"] = content_data.get("script", "") + "\n\nNew paragraph content...\n"

            with col_format3:
                if st.button("📋 Add List"):
                    content_data["script"] = content_data.get("script", "") + "\n- List item 1\n- List item 2\n- List item 3\n"

            with col_format4:
                if st.button("💡 Add Callout"):
                    content_data["script"] = content_data.get("script", "") + "\n> **Important:** This is a callout box\n"

            # Main script editor
            new_script = st.text_area(
                "Script Content",
                value=content_data.get("script", ""),
                height=400,
                key="editor_script",
                help="Write your main content here. Use Markdown formatting."
            )
            content_data["script"] = new_script

            # Word count and statistics
            if new_script:
                word_count = len(new_script.split())
                char_count = len(new_script)
                estimated_read_time = max(1, word_count // 200)  # Average reading speed

                st.caption(f"📊 Words: {word_count} | Characters: {char_count} | Est. reading time: {estimated_read_time} min")

        with col2:
            st.subheader("🎯 Writing Assistant")

            # AI-powered suggestions
            if st.button("🤖 Get AI Suggestions"):
                suggestions = self._generate_content_suggestions(content_data)
                for suggestion in suggestions:
                    with st.expander(f"💡 {suggestion['type']}"):
                        st.write(suggestion['content'])
                        if st.button(f"Apply", key=f"apply_{suggestion['type']}"):
                            content_data["script"] += f"\n\n{suggestion['content']}\n"
                            st.rerun()

            # Content templates
            st.write("**📋 Quick Templates:**")

            templates = {
                "Introduction": "Welcome to [TOPIC]. In this content, we'll explore...",
                "How-to Section": "Here's how to [ACTION]:\n1. Step one\n2. Step two\n3. Step three",
                "Conclusion": "To summarize, we've covered... The key takeaway is...",
                "Call to Action": "Ready to get started? Here's what you should do next..."
            }

            for template_name, template_content in templates.items():
                if st.button(f"➕ {template_name}", key=f"template_{template_name}"):
                    content_data["script"] += f"\n\n{template_content}\n"
                    st.rerun()

            # Content quality indicators
            st.write("**📈 Content Quality:**")
            quality_score = self._calculate_content_quality(content_data)

            st.metric("Quality Score", f"{quality_score}/100")

            # Quality breakdown
            quality_breakdown = self._get_quality_breakdown(content_data)
            for aspect, score in quality_breakdown.items():
                st.progress(score / 100, text=f"{aspect}: {score}/100")

    def _render_content_preview(self, content_data: Dict[str, Any]):
        """Render real-time content preview."""
        st.subheader("👁️ Live Preview")

        # Preview mode selector
        preview_mode = st.selectbox(
            "Preview Mode",
            ["📱 Mobile", "💻 Desktop", "📰 Article", "🎬 Video Script"],
            key="preview_mode"
        )

        # Apply preview styling based on mode
        if preview_mode == "📱 Mobile":
            preview_container = st.container()
            with preview_container:
                st.markdown(
                    """
                    <div style="max-width: 375px; margin: 0 auto; padding: 20px;
                         border: 2px solid #ddd; border-radius: 10px; background: #f9f9f9;">
                    """,
                    unsafe_allow_html=True
                )

                # Mobile preview content
                if content_data.get("title"):
                    st.markdown(f"# {content_data['title']}")

                if content_data.get("description"):
                    st.markdown(f"*{content_data['description']}*")

                if content_data.get("script"):
                    st.markdown(content_data["script"])

                st.markdown("</div>", unsafe_allow_html=True)

        elif preview_mode == "🎬 Video Script":
            # Video script preview with timestamps
            st.write("**Video Script Format:**")

            if content_data.get("script"):
                script_lines = content_data["script"].split('\n')
                current_time = 0

                for line in script_lines:
                    if line.strip():
                        # Estimate line duration (rough calculation)
                        words_in_line = len(line.split())
                        line_duration = max(3, words_in_line * 0.5)  # ~2 words per second

                        minutes = current_time // 60
                        seconds = current_time % 60

                        st.markdown(f"**[{minutes:02d}:{seconds:02d}]** {line}")
                        current_time += line_duration

        else:
            # Standard preview
            if content_data.get("title"):
                st.markdown(f"# {content_data['title']}")

            if content_data.get("description"):
                st.markdown(f"*{content_data['description']}*")
                st.markdown("---")

            if content_data.get("script"):
                st.markdown(content_data["script"])

        # Preview statistics
        if content_data.get("script"):
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                word_count = len(content_data["script"].split())
                st.metric("Words", word_count)

            with stats_col2:
                estimated_duration = max(1, word_count // 150)  # Speaking speed
                st.metric("Est. Duration", f"{estimated_duration} min")

            with stats_col3:
                readability = self._calculate_readability_score(content_data["script"])
                st.metric("Readability", f"{readability}/100")

    def _render_metadata_editor(self, content_data: Dict[str, Any]):
        """Render metadata editing interface."""
        st.subheader("🏷️ Content Metadata")

        col1, col2 = st.columns(2)

        with col1:
            # Tags management
            st.write("**Tags:**")

            # Current tags
            current_tags = content_data.get("tags", [])
            if current_tags:
                for i, tag in enumerate(current_tags):
                    col_tag, col_remove = st.columns([3, 1])
                    with col_tag:
                        st.write(f"🏷️ {tag}")
                    with col_remove:
                        if st.button("❌", key=f"remove_tag_{i}"):
                            current_tags.remove(tag)
                            content_data["tags"] = current_tags
                            st.rerun()

            # Add new tag
            new_tag = st.text_input("Add new tag", key="new_tag")
            if st.button("➕ Add Tag") and new_tag:
                if "tags" not in content_data:
                    content_data["tags"] = []
                content_data["tags"].append(new_tag)
                st.rerun()

            # Suggested tags based on content
            if content_data.get("script") or content_data.get("title"):
                suggested_tags = self._generate_suggested_tags(content_data)
                if suggested_tags:
                    st.write("**Suggested Tags:**")
                    for tag in suggested_tags[:5]:
                        if st.button(f"➕ {tag}", key=f"suggest_{tag}"):
                            if "tags" not in content_data:
                                content_data["tags"] = []
                            if tag not in content_data["tags"]:
                                content_data["tags"].append(tag)
                                st.rerun()

        with col2:
            # Content metadata
            st.write("**Content Settings:**")

            # Content type
            content_type = st.selectbox(
                "Content Type",
                ["Article", "Video Script", "Social Media Post", "Tutorial", "Case Study"],
                key="content_type"
            )

            # Target audience
            target_audience = st.selectbox(
                "Target Audience",
                ["General", "Beginners", "Intermediate", "Advanced", "Professionals"],
                key="target_audience"
            )

            # Content goal
            content_goal = st.selectbox(
                "Content Goal",
                ["Educate", "Entertain", "Inform", "Persuade", "Inspire"],
                key="content_goal"
            )

            # Tone
            tone = st.selectbox(
                "Tone",
                ["Professional", "Casual", "Friendly", "Authoritative", "Conversational"],
                key="content_tone"
            )

            # Save metadata
            content_data["metadata"] = {
                "content_type": content_type,
                "target_audience": target_audience,
                "content_goal": content_goal,
                "tone": tone,
                "last_updated": datetime.now().isoformat()
            }

        # SEO optimization
        st.write("**🔍 SEO Optimization:**")

        # Meta description
        meta_description = st.text_area(
            "Meta Description",
            value=content_data.get("meta_description", ""),
            max_chars=160,
            help="Brief description for search engines (max 160 characters)"
        )
        content_data["meta_description"] = meta_description

        # Keywords
        keywords = st.text_input(
            "Keywords (comma-separated)",
            value=", ".join(content_data.get("keywords", [])),
            help="Keywords for SEO optimization"
        )
        if keywords:
            content_data["keywords"] = [k.strip() for k in keywords.split(",")]

    def _render_content_analytics(self, content_data: Dict[str, Any]):
        """Render content analytics and insights."""
        st.subheader("📊 Content Analytics")

        # Content analysis
        if content_data.get("script"):
            analysis = self._analyze_content_comprehensive(content_data)

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Readability Grade", f"{analysis['readability_grade']}")

            with col2:
                st.metric("Sentiment Score", f"{analysis['sentiment_score']}/100")

            with col3:
                st.metric("Keyword Density", f"{analysis['keyword_density']:.1%}")

            with col4:
                st.metric("Engagement Potential", f"{analysis['engagement_score']}/100")

            # Detailed analysis
            st.write("**📈 Detailed Analysis:**")

            # Word frequency chart
            if analysis['word_frequency']:
                word_freq_df = pd.DataFrame(
                    list(analysis['word_frequency'].items())[:10],
                    columns=['Word', 'Frequency']
                )

                fig = px.bar(
                    word_freq_df,
                    x='Frequency',
                    y='Word',
                    orientation='h',
                    title="Top 10 Most Used Words"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Content structure analysis
            st.write("**📋 Content Structure:**")
            structure_analysis = analysis['structure_analysis']

            for section, details in structure_analysis.items():
                with st.expander(f"{section.title()} Analysis"):
                    st.json(details)

            # Improvement suggestions
            if analysis['suggestions']:
                st.write("**💡 Improvement Suggestions:**")
                for suggestion in analysis['suggestions']:
                    st.info(f"💡 {suggestion}")
        else:
            st.info("Add content to see analytics")

    def render_collaboration_features(self):
        """Render collaboration and sharing features."""
        st.header("🤝 Collaboration Features")

        tab1, tab2, tab3 = st.tabs(["👥 Team", "💬 Comments", "📤 Share"])

        with tab1:
            self._render_team_collaboration()

        with tab2:
            self._render_comment_system()

        with tab3:
            self._render_sharing_options()

    def _render_team_collaboration(self):
        """Render team collaboration interface."""
        st.subheader("👥 Team Collaboration")

        # Team members (simulated)
        st.write("**Current Team Members:**")

        team_members = [
            {"name": "Alice Johnson", "role": "Content Lead", "status": "online"},
            {"name": "Bob Smith", "role": "Editor", "status": "offline"},
            {"name": "Carol Brown", "role": "Designer", "status": "online"}
        ]

        for member in team_members:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                status_icon = "🟢" if member["status"] == "online" else "🔴"
                st.write(f"{status_icon} {member['name']}")

            with col2:
                st.write(member["role"])

            with col3:
                if st.button("💬 Message", key=f"message_{member['name']}"):
                    st.info(f"Message feature for {member['name']} would open here")

        # Add team member
        st.write("**Add Team Member:**")
        col1, col2 = st.columns([2, 1])

        with col1:
            new_member_email = st.text_input("Email address", key="new_member_email")

        with col2:
            member_role = st.selectbox("Role", ["Viewer", "Editor", "Admin"], key="member_role")

        if st.button("➕ Invite Member"):
            if new_member_email:
                st.success(f"Invitation sent to {new_member_email} as {member_role}")

    def _render_comment_system(self):
        """Render comment and feedback system."""
        st.subheader("💬 Comments & Feedback")

        # Existing comments (simulated)
        comments = [
            {
                "author": "Alice Johnson",
                "timestamp": "2 hours ago",
                "content": "Great start! Consider adding more examples in the introduction section.",
                "type": "suggestion"
            },
            {
                "author": "Bob Smith",
                "timestamp": "1 hour ago",
                "content": "The flow between sections 2 and 3 could be smoother.",
                "type": "feedback"
            }
        ]

        for comment in comments:
            with st.expander(f"💬 {comment['author']} - {comment['timestamp']}"):
                st.write(comment['content'])

                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("👍", key=f"like_{comment['author']}"):
                        st.success("Liked!")
                with col2:
                    if st.button("💬 Reply", key=f"reply_{comment['author']}"):
                        st.info("Reply feature would open here")

        # Add new comment
        st.write("**Add Comment:**")
        new_comment = st.text_area("Your comment", key="new_comment")
        comment_type = st.selectbox("Type", ["General", "Suggestion", "Question", "Approval"])

        if st.button("💬 Post Comment") and new_comment:
            st.success("Comment posted successfully!")

    def _render_sharing_options(self):
        """Render content sharing and export options."""
        st.subheader("📤 Share & Export")

        # Sharing links
        st.write("**Share Links:**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔗 Generate Share Link"):
                share_link = f"https://your-app.com/content/shared/{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.code(share_link)
                st.success("Share link generated!")

        with col2:
            if st.button("📧 Email Share"):
                st.success("Email sharing dialog would open here")

        # Export options
        st.write("**Export Options:**")

        export_formats = ["PDF", "Word Document", "HTML", "Markdown", "Plain Text"]
        selected_format = st.selectbox("Export Format", export_formats)

        if st.button(f"📄 Export as {selected_format}"):
            st.success(f"Content exported as {selected_format}")
            # In a real implementation, this would generate and download the file

        # Social media sharing
        st.write("**Social Media:**")

        social_platforms = {
            "Twitter": "🐦",
            "LinkedIn": "💼",
            "Facebook": "📘",
            "Instagram": "📷"
        }

        cols = st.columns(len(social_platforms))

        for i, (platform, icon) in enumerate(social_platforms.items()):
            with cols[i]:
                if st.button(f"{icon} {platform}", key=f"share_{platform}"):
                    st.success(f"Shared to {platform}!")

    def render_personalized_dashboard(self, user_preferences: Dict[str, Any] = None):
        """Render personalized user dashboard."""
        st.header("🎯 Personalized Dashboard")

        if user_preferences is None:
            user_preferences = self._get_default_preferences()

        # Dashboard customization
        st.sidebar.subheader("🎨 Customize Dashboard")

        layout_style = st.sidebar.selectbox(
            "Layout Style",
            ["Compact", "Detailed", "Grid", "List"],
            index=0
        )

        theme = st.sidebar.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"],
            index=0
        )

        show_analytics = st.sidebar.checkbox("Show Analytics", value=True)
        show_recommendations = st.sidebar.checkbox("Show Recommendations", value=True)
        show_recent_content = st.sidebar.checkbox("Show Recent Content", value=True)

        # Main dashboard content
        if layout_style == "Grid":
            self._render_grid_dashboard(user_preferences, show_analytics, show_recommendations, show_recent_content)
        else:
            self._render_standard_dashboard(user_preferences, show_analytics, show_recommendations, show_recent_content)

    def _render_grid_dashboard(self, preferences: Dict[str, Any], show_analytics: bool,
                             show_recommendations: bool, show_recent_content: bool):
        """Render grid-style dashboard."""

        if show_analytics:
            with st.container():
                st.subheader("📊 Quick Analytics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Content Created", "24", delta="3")

                with col2:
                    st.metric("Avg Quality", "87", delta="5")

                with col3:
                    st.metric("Total Views", "1.2K", delta="250")

                with col4:
                    st.metric("Engagement", "92%", delta="8%")

        # Two-column layout for recommendations and recent content
        if show_recommendations or show_recent_content:
            col1, col2 = st.columns(2)

            if show_recommendations:
                with col1:
                    self._render_recommendations_widget()

            if show_recent_content:
                with col2:
                    self._render_recent_content_widget()

    def _render_standard_dashboard(self, preferences: Dict[str, Any], show_analytics: bool,
                                 show_recommendations: bool, show_recent_content: bool):
        """Render standard dashboard layout."""

        if show_analytics:
            self._render_analytics_widget()

        if show_recommendations:
            self._render_recommendations_widget()

        if show_recent_content:
            self._render_recent_content_widget()

    def _render_recommendations_widget(self):
        """Render content recommendations widget."""
        st.subheader("💡 Recommended for You")

        if self.recommendation_engine:
            recommendations = self.recommendation_engine.generate_content_recommendations(count=3)

            for rec in recommendations:
                with st.expander(f"💡 {rec['title']}"):
                    st.write(rec['description'])
                    st.write(f"**Category:** {rec['category']}")
                    st.write(f"**Difficulty:** {rec.get('difficulty_level', 'N/A')}")

                    if st.button(f"🚀 Create Content", key=f"create_{rec['title'][:20]}"):
                        st.success("Content creation started!")
        else:
            st.info("Connect recommendation engine to see personalized suggestions")

    def _render_recent_content_widget(self):
        """Render recent content widget."""
        st.subheader("📚 Recent Content")

        if self.database_manager:
            recent_content = self.database_manager.get_content_history(limit=5)

            for content in recent_content:
                with st.expander(f"📄 {content['topic']}"):
                    st.write(f"**Created:** {content['created_at']}")
                    st.write(f"**Type:** {content['content_type']}")
                    if content['quality_score']:
                        st.write(f"**Quality:** {content['quality_score']}/100")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✏️ Edit", key=f"edit_{content['id']}"):
                            st.success("Opening editor...")
                    with col2:
                        if st.button("📤 Share", key=f"share_{content['id']}"):
                            st.success("Sharing options...")
        else:
            st.info("Connect database to see recent content")

    def _render_analytics_widget(self):
        """Render analytics widget."""
        st.subheader("📊 Content Analytics")

        # Sample analytics data
        analytics_data = {
            "daily_content": [2, 3, 1, 4, 2, 3, 5],
            "quality_scores": [85, 92, 78, 88, 95, 82, 90],
            "categories": ["Educational", "Business", "Technology", "Lifestyle"]
        }

        # Daily content creation chart
        fig = px.line(
            x=list(range(7)),
            y=analytics_data["daily_content"],
            title="Daily Content Creation (Last 7 Days)",
            labels={'x': 'Days Ago', 'y': 'Content Pieces'}
        )
        st.plotly_chart(fig, use_container_width=True)

    def _generate_content_suggestions(self, content_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate AI-powered content suggestions."""
        suggestions = [
            {
                "type": "Introduction Enhancement",
                "content": "Consider starting with a compelling hook that addresses your audience's main pain point."
            },
            {
                "type": "Structure Improvement",
                "content": "Add subheadings to break up long sections and improve readability."
            },
            {
                "type": "Call to Action",
                "content": "Include a clear call to action that tells readers what to do next."
            }
        ]

        return suggestions

    def _calculate_content_quality(self, content_data: Dict[str, Any]) -> int:
        """Calculate overall content quality score."""
        score = 0

        # Title quality
        if content_data.get("title"):
            title_length = len(content_data["title"])
            if 30 <= title_length <= 60:
                score += 20
            elif title_length > 0:
                score += 10

        # Description quality
        if content_data.get("description"):
            desc_length = len(content_data["description"])
            if 100 <= desc_length <= 300:
                score += 20
            elif desc_length > 0:
                score += 10

        # Content quality
        if content_data.get("script"):
            script_length = len(content_data["script"].split())
            if script_length >= 100:
                score += 30
            elif script_length >= 50:
                score += 20
            elif script_length > 0:
                score += 10

        # Metadata completeness
        if content_data.get("tags"):
            score += 15

        if content_data.get("metadata"):
            score += 15

        return min(100, score)

    def _get_quality_breakdown(self, content_data: Dict[str, Any]) -> Dict[str, int]:
        """Get detailed quality breakdown."""
        breakdown = {
            "Title Quality": 0,
            "Content Length": 0,
            "Structure": 0,
            "Metadata": 0
        }

        # Title quality
        if content_data.get("title"):
            title_length = len(content_data["title"])
            if 30 <= title_length <= 60:
                breakdown["Title Quality"] = 100
            elif 20 <= title_length <= 80:
                breakdown["Title Quality"] = 75
            elif title_length > 0:
                breakdown["Title Quality"] = 50

        # Content length
        if content_data.get("script"):
            word_count = len(content_data["script"].split())
            if word_count >= 300:
                breakdown["Content Length"] = 100
            elif word_count >= 150:
                breakdown["Content Length"] = 75
            elif word_count >= 50:
                breakdown["Content Length"] = 50
            elif word_count > 0:
                breakdown["Content Length"] = 25

        # Structure (based on headings and paragraphs)
        if content_data.get("script"):
            script = content_data["script"]
            headings = script.count("#")
            paragraphs = script.count("\n\n")

            if headings >= 2 and paragraphs >= 3:
                breakdown["Structure"] = 100
            elif headings >= 1 or paragraphs >= 2:
                breakdown["Structure"] = 75
            elif script:
                breakdown["Structure"] = 50

        # Metadata completeness
        metadata_score = 0
        if content_data.get("tags"):
            metadata_score += 40
        if content_data.get("description"):
            metadata_score += 30
        if content_data.get("metadata"):
            metadata_score += 30
        breakdown["Metadata"] = metadata_score

        return breakdown

    def _calculate_readability_score(self, text: str) -> int:
        """Calculate simple readability score."""
        if not text:
            return 0

        sentences = text.split('.')
        words = text.split()

        if len(sentences) == 0:
            return 0

        avg_sentence_length = len(words) / len(sentences)

        # Simple scoring based on average sentence length
        if avg_sentence_length <= 15:
            return 90
        elif avg_sentence_length <= 20:
            return 75
        elif avg_sentence_length <= 25:
            return 60
        else:
            return 40

    def _generate_suggested_tags(self, content_data: Dict[str, Any]) -> List[str]:
        """Generate suggested tags based on content."""
        text = f"{content_data.get('title', '')} {content_data.get('script', '')}"

        # Simple keyword extraction (in real implementation, use NLP)
        words = text.lower().split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top words as suggested tags
        return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:10]

    def _analyze_content_comprehensive(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive content analysis."""
        script = content_data.get("script", "")

        if not script:
            return {}

        words = script.split()
        sentences = script.split('.')

        # Word frequency analysis
        word_freq = {}
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        for word in words:
            clean_word = word.lower().strip('.,!?";')
            if len(clean_word) > 2 and clean_word not in common_words:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        # Basic sentiment analysis (simplified)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'disappointing']

        positive_count = sum(1 for word in words if word.lower() in positive_words)
        negative_count = sum(1 for word in words if word.lower() in negative_words)

        sentiment_score = min(100, max(0, 50 + (positive_count - negative_count) * 10))

        # Structure analysis
        headings = script.count('#')
        lists = script.count('-') + script.count('*')
        paragraphs = script.count('\n\n')

        return {
            "readability_grade": self._calculate_readability_score(script),
            "sentiment_score": sentiment_score,
            "keyword_density": len(set(words)) / len(words) if words else 0,
            "engagement_score": min(100, (len(words) // 10) + (headings * 5) + (lists * 2)),
            "word_frequency": dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]),
            "structure_analysis": {
                "headings": headings,
                "lists": lists,
                "paragraphs": paragraphs,
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()])
            },
            "suggestions": self._get_content_suggestions(script, headings, lists, len(words))
        }

    def _get_content_suggestions(self, script: str, headings: int, lists: int, word_count: int) -> List[str]:
        """Generate content improvement suggestions."""
        suggestions = []

        if word_count < 100:
            suggestions.append("Consider expanding your content to at least 100 words for better depth")

        if headings == 0:
            suggestions.append("Add headings to improve content structure and readability")

        if lists == 0 and word_count > 200:
            suggestions.append("Consider adding bullet points or numbered lists to break up long text")

        if not script.count('?'):
            suggestions.append("Include questions to engage your audience")

        if word_count > 500 and headings < 3:
            suggestions.append("For longer content, use more subheadings to improve navigation")

        return suggestions

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences."""
        return {
            "theme": "Light",
            "layout": "Standard",
            "notifications": True,
            "auto_save": True,
            "collaborative_editing": False
        }


# Factory function
def create_ux_manager(database_manager=None, recommendation_engine=None) -> AdvancedUXManager:
    """Create advanced UX manager instance.

    Args:
        database_manager: Database manager instance
        recommendation_engine: Recommendation engine instance

    Returns:
        AdvancedUXManager instance
    """
    return AdvancedUXManager(database_manager, recommendation_engine)
