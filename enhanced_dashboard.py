"""
Enhanced Dashboard Integration with Performance Monitoring

This module integrates the workflow optimizations into the Streamlit dashboard
with real-time performance monitoring and quality metrics.
"""

import streamlit as st
import asyncio
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import our optimization modules
try:
    from workflow_optimization import create_workflow_optimizer
    from content_quality_enhanced import create_enhanced_quality_analyzer
    from content_creation import ContentCreator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def init_session_state():
    """Initialize session state variables."""
    if 'performance_history' not in st.session_state:
        st.session_state.performance_history = []
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'quality_analyzer' not in st.session_state:
        st.session_state.quality_analyzer = None
    if 'optimization_enabled' not in st.session_state:
        st.session_state.optimization_enabled = False

def render_performance_dashboard():
    """Render the performance monitoring dashboard."""
    st.header("🚀 Performance Dashboard")
    st.write("Monitor workflow optimization metrics and content quality in real-time.")

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    if st.session_state.optimizer:
        metrics = st.session_state.optimizer.get_performance_metrics()
        workflow_metrics = metrics['workflow_metrics']
        cache_metrics = metrics['cache_metrics']

        with col1:
            st.metric(
                "Total Requests",
                workflow_metrics['total_requests'],
                delta=None
            )

        with col2:
            st.metric(
                "Parallel Operations",
                workflow_metrics['parallel_operations'],
                delta=None
            )

        with col3:
            cache_hit_rate = metrics['cache_hit_rate']
            st.metric(
                "Cache Hit Rate",
                f"{cache_hit_rate:.1f}%",
                delta=f"+{cache_hit_rate:.1f}%" if cache_hit_rate > 0 else None
            )

        with col4:
            avg_response = workflow_metrics['average_response_time']
            st.metric(
                "Avg Response Time",
                f"{avg_response:.2f}s",
                delta=f"-{max(0, 3.0 - avg_response):.2f}s" if avg_response < 3.0 else None
            )
    else:
        # Show placeholder metrics
        with col1:
            st.metric("Total Requests", "0")
        with col2:
            st.metric("Parallel Operations", "0")
        with col3:
            st.metric("Cache Hit Rate", "0%")
        with col4:
            st.metric("Avg Response Time", "0.00s")

    # Performance history chart
    if st.session_state.performance_history:
        st.subheader("📊 Performance Trends")

        df = pd.DataFrame(st.session_state.performance_history)

        # Response time chart
        fig_response = px.line(
            df,
            x='timestamp',
            y='response_time',
            title="Response Time Over Time",
            labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_response, use_container_width=True)

        # Cache performance chart
        if 'cache_hit_rate' in df.columns:
            fig_cache = px.line(
                df,
                x='timestamp',
                y='cache_hit_rate',
                title="Cache Hit Rate Over Time",
                labels={'cache_hit_rate': 'Cache Hit Rate (%)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig_cache, use_container_width=True)

def render_optimized_workflow():
    """Render the optimized content creation workflow."""
    st.header("⚡ Optimized Content Creation")
    st.write("Experience enhanced performance with parallel processing and intelligent caching.")

    # Topic input
    topic = st.text_input(
        "Content Topic",
        placeholder="e.g., Introduction to Machine Learning",
        help="Enter the main topic for your content"
    )

    # Options
    col1, col2 = st.columns(2)

    with col1:
        include_script = st.checkbox("Generate Script", value=True)
        include_images = st.checkbox("Search Images", value=True)
        include_web_research = st.checkbox("Web Research", value=True)

    with col2:
        image_sources = st.multiselect(
            "Image Sources",
            ["pexels", "pixabay"],
            default=["pexels", "pixabay"]
        )
        images_per_source = st.slider("Images per Source", 1, 10, 3)
        enable_quality_analysis = st.checkbox("Quality Analysis", value=True)

    if st.button("🚀 Create Optimized Content", type="primary"):
        if not topic:
            st.error("Please enter a content topic.")
            return

        if not st.session_state.optimizer:
            st.error("Optimizer not initialized. Please check API keys.")
            return

        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Start timer
            start_time = time.time()

            # Update progress
            progress_bar.progress(10)
            status_text.text("🔄 Initializing content generation...")

            # Run optimized content bundle creation
            async def create_content():
                return await st.session_state.optimizer.optimized_content_bundle(
                    topic=topic,
                    include_images=include_images,
                    include_web_research=include_web_research,
                    include_script=include_script
                )

            # Execute async function
            content_bundle = asyncio.run(create_content())

            progress_bar.progress(70)
            status_text.text("📊 Analyzing content quality...")

            # Quality analysis if enabled
            quality_report = None
            if enable_quality_analysis and st.session_state.quality_analyzer:
                quality_report = st.session_state.quality_analyzer.generate_quality_report(
                    content_bundle['components'],
                    topic=topic
                )

            progress_bar.progress(90)
            status_text.text("📝 Preparing results...")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update performance history
            performance_data = {
                'timestamp': datetime.now(),
                'response_time': processing_time,
                'components': len(content_bundle['components']),
                'topic': topic
            }

            if st.session_state.optimizer:
                metrics = st.session_state.optimizer.get_performance_metrics()
                performance_data['cache_hit_rate'] = metrics['cache_hit_rate']

            st.session_state.performance_history.append(performance_data)

            # Complete progress
            progress_bar.progress(100)
            status_text.text("✅ Content creation completed!")

            # Display results
            st.success(f"🎉 Content created successfully in {processing_time:.2f} seconds!")

            # Show performance improvement
            if processing_time < 5.0:
                st.info(f"⚡ Fast generation! Completed in {processing_time:.2f}s")

            # Display content components
            st.subheader("📦 Generated Components")

            # Script
            if 'script' in content_bundle['components']:
                script_data = content_bundle['components']['script']
                if 'error' not in script_data:
                    with st.expander("📝 Generated Script", expanded=True):
                        st.write(f"**Title:** {script_data.get('title', 'N/A')}")
                        st.write(f"**Duration:** {script_data.get('duration', 'N/A')} minutes")
                        st.write("**Script Content:**")
                        st.write(script_data.get('script', 'No script content available'))
                else:
                    st.error(f"Script generation failed: {script_data['error']}")

            # Images
            if 'images' in content_bundle['components']:
                images_data = content_bundle['components']['images']
                if 'error' not in images_data:
                    with st.expander(f"🖼️ Images ({images_data.get('total_count', 0)} found)", expanded=True):
                        if images_data.get('aggregated_results'):
                            # Display images in grid
                            cols = st.columns(3)
                            for i, img in enumerate(images_data['aggregated_results'][:6]):  # Show first 6
                                with cols[i % 3]:
                                    st.image(img['src'], caption=f"By: {img['photographer']}")
                                    st.caption(f"Source: {img['provider']}")
                        else:
                            st.write("No images found.")
                else:
                    st.error(f"Image search failed: {images_data['error']}")

            # Web Research
            if 'web_research' in content_bundle['components']:
                research_data = content_bundle['components']['web_research']
                if 'error' not in research_data:
                    with st.expander("🔍 Web Research Results"):
                        st.json(research_data)
                else:
                    st.error(f"Web research failed: {research_data['error']}")

            # Quality Analysis
            if quality_report:
                st.subheader("📊 Quality Analysis")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{quality_report['overall_score']}/100")
                with col2:
                    st.metric("Grade", quality_report['overall_grade'])
                with col3:
                    components_count = len(quality_report['components'])
                    st.metric("Components Analyzed", components_count)

                # Quality breakdown
                if quality_report['summary']['strengths']:
                    st.write("**Strengths:**")
                    for strength in quality_report['summary']['strengths']:
                        st.write(f"✅ {strength}")

                if quality_report['summary']['weaknesses']:
                    st.write("**Areas for Improvement:**")
                    for weakness in quality_report['summary']['weaknesses']:
                        st.write(f"🔧 {weakness}")

                if quality_report['summary']['recommendations']:
                    st.write("**Recommendations:**")
                    for rec in quality_report['summary']['recommendations']:
                        st.write(f"💡 {rec}")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Content creation failed: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

def render_batch_operations():
    """Render the batch operations interface."""
    st.header("📚 Batch Operations")
    st.write("Process multiple content requests efficiently with batch optimization.")

    # Batch input
    st.subheader("🎯 Batch Image Search")

    queries_text = st.text_area(
        "Search Queries (one per line)",
        placeholder="machine learning\nartificial intelligence\ndata science\nneural networks",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Image Source", ["pexels", "pixabay"])
    with col2:
        count_per_query = st.slider("Images per Query", 1, 5, 2)

    if st.button("🚀 Start Batch Search"):
        if not queries_text.strip():
            st.error("Please enter search queries.")
            return

        if not st.session_state.optimizer:
            st.error("Optimizer not initialized.")
            return

        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text(f"🔄 Processing {len(queries)} queries...")

            async def batch_search():
                return await st.session_state.optimizer.batch_image_search(
                    queries=queries,
                    source=source,
                    count_per_query=count_per_query
                )

            start_time = time.time()
            results = asyncio.run(batch_search())
            processing_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.text("✅ Batch search completed!")

            # Display results
            st.success(f"🎉 Processed {results['batch_size']} queries in {processing_time:.2f} seconds!")
            st.info(f"📸 Total images found: {results['total_images']}")

            # Results breakdown
            st.subheader("📊 Results Breakdown")

            for query, data in results['queries'].items():
                with st.expander(f"🔍 {query} ({data['count']} images)"):
                    if data['images']:
                        cols = st.columns(min(3, len(data['images'])))
                        for i, img in enumerate(data['images']):
                            with cols[i % 3]:
                                st.image(img['src'], caption=f"By: {img['photographer']}")
                    else:
                        st.write("No images found for this query.")

            # Clear progress
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Batch search failed: {str(e)}")

def setup_optimization_system(api_keys):
    """Setup the optimization system with API keys."""
    try:
        # Create ContentCreator
        creator = ContentCreator(api_keys)

        # Create optimizer and quality analyzer
        optimizer = create_workflow_optimizer(creator)
        quality_analyzer = create_enhanced_quality_analyzer()

        # Store in session state
        st.session_state.optimizer = optimizer
        st.session_state.quality_analyzer = quality_analyzer
        st.session_state.optimization_enabled = True

        return True
    except Exception as e:
        st.error(f"Failed to initialize optimization system: {str(e)}")
        return False

def render_optimization_settings():
    """Render optimization settings and controls."""
    st.sidebar.header("⚙️ Optimization Settings")

    # Enable/disable optimization
    optimize_enabled = st.sidebar.checkbox(
        "Enable Workflow Optimization",
        value=st.session_state.optimization_enabled,
        help="Enable advanced performance optimizations"
    )

    if optimize_enabled and not st.session_state.optimization_enabled:
        # Try to initialize optimization
        from content_dashboard import get_api_key_from_sources

        api_keys = {
            "api_key": get_api_key_from_sources("GEMINI_API_KEY", "API_KEY", "GEMINI_API_KEY"),
            "gemini_api_key": get_api_key_from_sources("GEMINI_API_KEY", "API_KEY", "GEMINI_API_KEY"),
            "model_name": "gemini-2.0-flash",
            "temperature": "0.7",
            "pexels": get_api_key_from_sources("PEXELS_API_KEY", "PEXELS_API_KEY"),
            "pixabay": get_api_key_from_sources("PIXABAY_API_KEY", "PIXABAY_API_KEY"),
            "stability": get_api_key_from_sources("STABILITY_API_KEY", "STABILITY_API_KEY"),
            "assemblyai": get_api_key_from_sources("ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY"),
            "deepgram": get_api_key_from_sources("DEEPGRAM_API_KEY", "DEEPGRAM_API_KEY"),
            "tavily": get_api_key_from_sources("TAVILY_API_KEY", "TAVILY_API_KEY")
        }

        if setup_optimization_system(api_keys):
            st.sidebar.success("✅ Optimization enabled!")
        else:
            st.sidebar.error("❌ Failed to enable optimization")

    # Optimization status
    if st.session_state.optimization_enabled:
        st.sidebar.success("🚀 Optimization Active")

        # Show performance metrics in sidebar
        if st.session_state.optimizer:
            metrics = st.session_state.optimizer.get_performance_metrics()
            st.sidebar.metric("Cache Hit Rate", f"{metrics['cache_hit_rate']:.1f}%")
            st.sidebar.metric("Parallel Ops", metrics['workflow_metrics']['parallel_operations'])

        # Clear cache button
        if st.sidebar.button("🗑️ Clear Cache"):
            if st.session_state.optimizer:
                st.session_state.optimizer.cache.clear()
                st.sidebar.success("Cache cleared!")
    else:
        st.sidebar.warning("⚠️ Optimization Disabled")

# Main app integration
def render_enhanced_dashboard():
    """Render the enhanced dashboard with optimization features."""
    init_session_state()
    render_optimization_settings()

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Performance Dashboard",
        "⚡ Optimized Workflow",
        "📚 Batch Operations",
        "📊 Analytics"
    ])

    with tab1:
        render_performance_dashboard()

    with tab2:
        render_optimized_workflow()

    with tab3:
        render_batch_operations()

    with tab4:
        st.header("📊 Analytics & Reports")
        st.write("Advanced analytics and performance reports.")

        if st.session_state.performance_history:
            # Export performance data
            if st.button("📥 Export Performance Data"):
                df = pd.DataFrame(st.session_state.performance_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            # Performance summary
            df = pd.DataFrame(st.session_state.performance_history)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sessions", len(df))
                st.metric("Avg Response Time", f"{df['response_time'].mean():.2f}s")
            with col2:
                st.metric("Fastest Response", f"{df['response_time'].min():.2f}s")
                if 'cache_hit_rate' in df.columns:
                    st.metric("Best Cache Rate", f"{df['cache_hit_rate'].max():.1f}%")
        else:
            st.info("No performance data available yet. Start using the optimized workflow to see analytics.")

if __name__ == "__main__":
    render_enhanced_dashboard()
