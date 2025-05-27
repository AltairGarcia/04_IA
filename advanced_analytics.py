"""
Advanced Analytics & Reporting System
====================================

A comprehensive system for advanced analytics, A/B testing, ROI analysis,
and deep performance insights for content creation workflows.

Features:
- A/B testing framework for content variations
- ROI analysis and conversion tracking
- Advanced performance metrics and reporting
- Predictive analytics using machine learning
- Custom dashboard creation and visualization
- Automated report generation and distribution
"""

import json
import sqlite3
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import uuid
import statistics
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """A/B test types."""
    CONTENT_VARIATION = "content_variation"
    TEMPLATE_COMPARISON = "template_comparison"
    TIMING_TEST = "timing_test"
    AUDIENCE_SEGMENTATION = "audience_segmentation"
    CHANNEL_OPTIMIZATION = "channel_optimization"

class MetricType(Enum):
    """Performance metric types."""
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    REACH = "reach"
    CLICK_THROUGH = "click_through"
    RETENTION = "retention"
    REVENUE = "revenue"
    COST = "cost"
    ROI = "roi"

class ReportType(Enum):
    """Report types."""
    PERFORMANCE_SUMMARY = "performance_summary"
    AB_TEST_RESULTS = "ab_test_results"
    ROI_ANALYSIS = "roi_analysis"
    TREND_ANALYSIS = "trend_analysis"
    PREDICTIVE_INSIGHTS = "predictive_insights"
    CUSTOM_DASHBOARD = "custom_dashboard"

@dataclass
class ABTestConfig:
    """A/B test configuration."""
    id: str
    name: str
    test_type: TestType
    description: str
    variants: List[Dict[str, Any]]
    target_metric: MetricType
    sample_size_per_variant: int
    confidence_level: float
    test_duration_days: int
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    status: str
    metadata: Dict[str, Any]

@dataclass
class ABTestResult:
    """A/B test result data."""
    test_id: str
    variant_id: str
    participant_id: str
    metric_value: float
    additional_metrics: Dict[str, float]
    timestamp: datetime
    session_data: Dict[str, Any]

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    id: str
    content_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    channel: str
    audience_segment: str
    cost: float
    revenue: float
    metadata: Dict[str, Any]

@dataclass
class ReportConfig:
    """Report configuration."""
    id: str
    name: str
    report_type: ReportType
    parameters: Dict[str, Any]
    schedule: Optional[Dict[str, Any]]
    recipients: List[str]
    created_at: datetime
    last_generated: Optional[datetime]
    is_active: bool

class ABTestingFramework:
    """Advanced A/B testing framework for content optimization."""

    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize A/B testing database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # A/B test configurations
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ab_test_configs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        test_type TEXT NOT NULL,
                        description TEXT,
                        variants TEXT,
                        target_metric TEXT,
                        sample_size_per_variant INTEGER,
                        confidence_level REAL,
                        test_duration_days INTEGER,
                        created_at TIMESTAMP,
                        started_at TIMESTAMP,
                        ended_at TIMESTAMP,
                        status TEXT,
                        metadata TEXT
                    )
                """)

                # A/B test results
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ab_test_results (
                        id TEXT PRIMARY KEY,
                        test_id TEXT,
                        variant_id TEXT,
                        participant_id TEXT,
                        metric_value REAL,
                        additional_metrics TEXT,
                        timestamp TIMESTAMP,
                        session_data TEXT,
                        FOREIGN KEY (test_id) REFERENCES ab_test_configs (id)
                    )
                """)

                # Performance metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id TEXT PRIMARY KEY,
                        content_id TEXT,
                        metric_type TEXT,
                        value REAL,
                        timestamp TIMESTAMP,
                        channel TEXT,
                        audience_segment TEXT,
                        cost REAL,
                        revenue REAL,
                        metadata TEXT
                    )
                """)

                conn.commit()
                logger.info("A/B testing database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing A/B testing database: {e}")
            raise

    def create_ab_test(self, name: str, test_type: TestType, description: str,
                      variants: List[Dict[str, Any]], target_metric: MetricType,
                      sample_size_per_variant: int = 1000, confidence_level: float = 0.95,
                      test_duration_days: int = 14) -> str:
        """Create a new A/B test."""
        test_id = str(uuid.uuid4())

        config = ABTestConfig(
            id=test_id,
            name=name,
            test_type=test_type,
            description=description,
            variants=variants,
            target_metric=target_metric,
            sample_size_per_variant=sample_size_per_variant,
            confidence_level=confidence_level,
            test_duration_days=test_duration_days,
            created_at=datetime.now(),
            started_at=None,
            ended_at=None,
            status="created",
            metadata={}
        )

        self.save_ab_test_config(config)
        return test_id

    def save_ab_test_config(self, config: ABTestConfig):
        """Save A/B test configuration to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO ab_test_configs (
                        id, name, test_type, description, variants, target_metric,
                        sample_size_per_variant, confidence_level, test_duration_days,
                        created_at, started_at, ended_at, status, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config.id, config.name, config.test_type.value, config.description,
                    json.dumps(config.variants), config.target_metric.value,
                    config.sample_size_per_variant, config.confidence_level,
                    config.test_duration_days, config.created_at, config.started_at,
                    config.ended_at, config.status, json.dumps(config.metadata)
                ))

                conn.commit()
                logger.info(f"A/B test config '{config.name}' saved successfully")

        except Exception as e:
            logger.error(f"Error saving A/B test config: {e}")
            raise

    def start_ab_test(self, test_id: str):
        """Start an A/B test."""
        config = self.get_ab_test_config(test_id)
        if not config:
            raise ValueError(f"A/B test {test_id} not found")

        config.status = "running"
        config.started_at = datetime.now()
        self.save_ab_test_config(config)

        logger.info(f"A/B test '{config.name}' started")

    def get_ab_test_config(self, test_id: str) -> Optional[ABTestConfig]:
        """Get A/B test configuration by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM ab_test_configs WHERE id = ?", (test_id,))
                row = cursor.fetchone()

                if row:
                    return ABTestConfig(
                        id=row[0], name=row[1], test_type=TestType(row[2]),
                        description=row[3], variants=json.loads(row[4]),
                        target_metric=MetricType(row[5]), sample_size_per_variant=row[6],
                        confidence_level=row[7], test_duration_days=row[8],
                        created_at=datetime.fromisoformat(row[9]),
                        started_at=datetime.fromisoformat(row[10]) if row[10] else None,
                        ended_at=datetime.fromisoformat(row[11]) if row[11] else None,
                        status=row[12], metadata=json.loads(row[13])
                    )

        except Exception as e:
            logger.error(f"Error getting A/B test config: {e}")

        return None

    def record_ab_test_result(self, test_id: str, variant_id: str, participant_id: str,
                            metric_value: float, additional_metrics: Dict[str, float] = None,
                            session_data: Dict[str, Any] = None):
        """Record an A/B test result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                result_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO ab_test_results (
                        id, test_id, variant_id, participant_id, metric_value,
                        additional_metrics, timestamp, session_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id, test_id, variant_id, participant_id, metric_value,
                    json.dumps(additional_metrics or {}), datetime.now(),
                    json.dumps(session_data or {})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error recording A/B test result: {e}")
            raise

    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results using statistical methods."""
        config = self.get_ab_test_config(test_id)
        if not config:
            raise ValueError(f"A/B test {test_id} not found")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT variant_id, metric_value
                    FROM ab_test_results
                    WHERE test_id = ?
                """, (test_id,))

                results = cursor.fetchall()

                if not results:
                    return {"error": "No results found for this test"}

                # Group results by variant
                variant_data = {}
                for variant_id, metric_value in results:
                    if variant_id not in variant_data:
                        variant_data[variant_id] = []
                    variant_data[variant_id].append(metric_value)

                # Calculate statistics for each variant
                variant_stats = {}
                for variant_id, values in variant_data.items():
                    variant_stats[variant_id] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "confidence_interval": self._calculate_confidence_interval(
                            values, config.confidence_level
                        )
                    }

                # Perform statistical significance testing
                if len(variant_data) == 2:
                    variant_ids = list(variant_data.keys())
                    control_data = variant_data[variant_ids[0]]
                    treatment_data = variant_data[variant_ids[1]]

                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)

                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data) +
                                        (len(treatment_data) - 1) * np.var(treatment_data)) /
                                       (len(control_data) + len(treatment_data) - 2))
                    cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

                    # Determine statistical significance
                    alpha = 1 - config.confidence_level
                    is_significant = p_value < alpha

                    # Calculate relative improvement
                    relative_improvement = ((np.mean(treatment_data) - np.mean(control_data)) /
                                          np.mean(control_data)) * 100

                    statistical_analysis = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "is_significant": is_significant,
                        "confidence_level": config.confidence_level,
                        "effect_size": cohens_d,
                        "relative_improvement": relative_improvement,
                        "winner": variant_ids[1] if is_significant and relative_improvement > 0 else variant_ids[0]
                    }
                else:
                    # Multiple variants - use ANOVA
                    variant_values = list(variant_data.values())
                    f_stat, p_value = stats.f_oneway(*variant_values)

                    statistical_analysis = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "is_significant": p_value < (1 - config.confidence_level),
                        "confidence_level": config.confidence_level,
                        "best_variant": max(variant_stats.keys(),
                                          key=lambda x: variant_stats[x]["mean"])
                    }

                return {
                    "test_id": test_id,
                    "test_name": config.name,
                    "variant_statistics": variant_stats,
                    "statistical_analysis": statistical_analysis,
                    "sample_sizes": {vid: len(vdata) for vid, vdata in variant_data.items()},
                    "total_participants": len(results),
                    "analysis_date": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error analyzing A/B test: {e}")
            return {"error": str(e)}

    def _calculate_confidence_interval(self, data: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for a dataset."""
        if len(data) < 2:
            return (0, 0)

        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence_level) / 2., len(data) - 1)

        return (mean - h, mean + h)

class ROIAnalyzer:
    """ROI analysis and financial performance tracking."""

    def __init__(self, db_path: str = "roi_analytics.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize ROI analytics database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # ROI tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS roi_data (
                        id TEXT PRIMARY KEY,
                        content_id TEXT,
                        campaign_id TEXT,
                        investment REAL,
                        revenue REAL,
                        roi_percentage REAL,
                        timeframe_start TIMESTAMP,
                        timeframe_end TIMESTAMP,
                        channel TEXT,
                        audience_segment TEXT,
                        conversion_count INTEGER,
                        cost_per_conversion REAL,
                        lifetime_value REAL,
                        attribution_model TEXT,
                        metadata TEXT,
                        recorded_at TIMESTAMP
                    )
                """)

                # Cost breakdown table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cost_breakdown (
                        id TEXT PRIMARY KEY,
                        content_id TEXT,
                        cost_type TEXT,
                        amount REAL,
                        description TEXT,
                        date TIMESTAMP,
                        vendor TEXT,
                        metadata TEXT
                    )
                """)

                # Revenue attribution table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS revenue_attribution (
                        id TEXT PRIMARY KEY,
                        content_id TEXT,
                        revenue_source TEXT,
                        amount REAL,
                        attribution_weight REAL,
                        customer_id TEXT,
                        transaction_date TIMESTAMP,
                        metadata TEXT
                    )
                """)

                conn.commit()
                logger.info("ROI analytics database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ROI analytics database: {e}")
            raise

    def track_investment(self, content_id: str, campaign_id: str, cost_type: str,
                        amount: float, description: str = "", vendor: str = ""):
        """Track investment/cost for content or campaign."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cost_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO cost_breakdown (
                        id, content_id, cost_type, amount, description, date, vendor, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cost_id, content_id, cost_type, amount, description,
                    datetime.now(), vendor, json.dumps({})
                ))

                conn.commit()
                logger.info(f"Investment tracked: {cost_type} - ${amount}")

        except Exception as e:
            logger.error(f"Error tracking investment: {e}")
            raise

    def track_revenue(self, content_id: str, revenue_source: str, amount: float,
                     attribution_weight: float = 1.0, customer_id: str = ""):
        """Track revenue attributed to content."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                revenue_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO revenue_attribution (
                        id, content_id, revenue_source, amount, attribution_weight,
                        customer_id, transaction_date, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    revenue_id, content_id, revenue_source, amount, attribution_weight,
                    customer_id, datetime.now(), json.dumps({})
                ))

                conn.commit()
                logger.info(f"Revenue tracked: {revenue_source} - ${amount}")

        except Exception as e:
            logger.error(f"Error tracking revenue: {e}")
            raise

    def calculate_roi(self, content_id: str, timeframe_days: int = 30) -> Dict[str, Any]:
        """Calculate ROI for specific content over a timeframe."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total investment
                cursor.execute("""
                    SELECT SUM(amount) FROM cost_breakdown
                    WHERE content_id = ? AND date BETWEEN ? AND ?
                """, (content_id, start_date, end_date))

                total_investment = cursor.fetchone()[0] or 0

                # Get total revenue
                cursor.execute("""
                    SELECT SUM(amount * attribution_weight) FROM revenue_attribution
                    WHERE content_id = ? AND transaction_date BETWEEN ? AND ?
                """, (content_id, start_date, end_date))

                total_revenue = cursor.fetchone()[0] or 0

                # Calculate ROI
                roi_percentage = ((total_revenue - total_investment) / total_investment * 100) if total_investment > 0 else 0
                profit = total_revenue - total_investment

                # Get cost breakdown
                cursor.execute("""
                    SELECT cost_type, SUM(amount) FROM cost_breakdown
                    WHERE content_id = ? AND date BETWEEN ? AND ?
                    GROUP BY cost_type
                """, (content_id, start_date, end_date))

                cost_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

                # Get revenue breakdown
                cursor.execute("""
                    SELECT revenue_source, SUM(amount * attribution_weight) FROM revenue_attribution
                    WHERE content_id = ? AND transaction_date BETWEEN ? AND ?
                    GROUP BY revenue_source
                """, (content_id, start_date, end_date))

                revenue_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

                # Store ROI calculation
                roi_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO roi_data (
                        id, content_id, investment, revenue, roi_percentage,
                        timeframe_start, timeframe_end, recorded_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    roi_id, content_id, total_investment, total_revenue, roi_percentage,
                    start_date, end_date, datetime.now(), json.dumps({})
                ))

                conn.commit()

                return {
                    "content_id": content_id,
                    "timeframe_days": timeframe_days,
                    "total_investment": total_investment,
                    "total_revenue": total_revenue,
                    "profit": profit,
                    "roi_percentage": roi_percentage,
                    "cost_breakdown": cost_breakdown,
                    "revenue_breakdown": revenue_breakdown,
                    "calculation_date": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return {"error": str(e)}

    def get_roi_trends(self, timeframe_days: int = 90) -> Dict[str, Any]:
        """Get ROI trends across multiple content pieces."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get ROI data over time
                cursor.execute("""
                    SELECT DATE(recorded_at) as date,
                           AVG(roi_percentage) as avg_roi,
                           SUM(investment) as total_investment,
                           SUM(revenue) as total_revenue,
                           COUNT(*) as content_count
                    FROM roi_data
                    WHERE recorded_at BETWEEN ? AND ?
                    GROUP BY DATE(recorded_at)
                    ORDER BY date
                """, (start_date, end_date))

                daily_data = cursor.fetchall()

                # Get top performing content
                cursor.execute("""
                    SELECT content_id, roi_percentage, investment, revenue
                    FROM roi_data
                    WHERE recorded_at BETWEEN ? AND ?
                    ORDER BY roi_percentage DESC
                    LIMIT 10
                """, (start_date, end_date))

                top_performers = cursor.fetchall()

                # Get bottom performing content
                cursor.execute("""
                    SELECT content_id, roi_percentage, investment, revenue
                    FROM roi_data
                    WHERE recorded_at BETWEEN ? AND ?
                    ORDER BY roi_percentage ASC
                    LIMIT 10
                """, (start_date, end_date))

                bottom_performers = cursor.fetchall()

                return {
                    "timeframe_days": timeframe_days,
                    "daily_trends": [
                        {
                            "date": row[0],
                            "avg_roi": row[1],
                            "total_investment": row[2],
                            "total_revenue": row[3],
                            "content_count": row[4]
                        } for row in daily_data
                    ],
                    "top_performers": [
                        {
                            "content_id": row[0],
                            "roi_percentage": row[1],
                            "investment": row[2],
                            "revenue": row[3]
                        } for row in top_performers
                    ],
                    "bottom_performers": [
                        {
                            "content_id": row[0],
                            "roi_percentage": row[1],
                            "investment": row[2],
                            "revenue": row[3]
                        } for row in bottom_performers
                    ],
                    "analysis_date": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error getting ROI trends: {e}")
            return {"error": str(e)}

class PredictiveAnalytics:
    """Machine learning-powered predictive analytics."""

    def __init__(self, ab_testing_db: str = "ab_testing.db", roi_db: str = "roi_analytics.db"):
        self.ab_testing_db = ab_testing_db
        self.roi_db = roi_db
        self.models = {}

    def prepare_performance_data(self) -> pd.DataFrame:
        """Prepare data for machine learning models."""
        try:
            # Combine data from both databases
            ab_data = self._get_ab_testing_data()
            roi_data = self._get_roi_data()

            # Merge and prepare features
            combined_data = pd.merge(ab_data, roi_data, on='content_id', how='outer')

            # Feature engineering
            combined_data['roi_category'] = pd.cut(combined_data['roi_percentage'],
                                                 bins=[-float('inf'), 0, 50, 100, float('inf')],
                                                 labels=['poor', 'fair', 'good', 'excellent'])

            combined_data['investment_category'] = pd.cut(combined_data['investment'],
                                                        bins=[0, 100, 500, 1000, float('inf')],
                                                        labels=['low', 'medium', 'high', 'very_high'])

            return combined_data

        except Exception as e:
            logger.error(f"Error preparing performance data: {e}")
            return pd.DataFrame()

    def _get_ab_testing_data(self) -> pd.DataFrame:
        """Get A/B testing data."""
        try:
            with sqlite3.connect(self.ab_testing_db) as conn:
                query = """
                    SELECT
                        r.test_id as content_id,
                        AVG(r.metric_value) as avg_performance,
                        COUNT(r.id) as sample_size,
                        c.test_type,
                        c.target_metric
                    FROM ab_test_results r
                    JOIN ab_test_configs c ON r.test_id = c.id
                    GROUP BY r.test_id
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting A/B testing data: {e}")
            return pd.DataFrame()

    def _get_roi_data(self) -> pd.DataFrame:
        """Get ROI data."""
        try:
            with sqlite3.connect(self.roi_db) as conn:
                query = """
                    SELECT
                        content_id,
                        AVG(roi_percentage) as roi_percentage,
                        AVG(investment) as investment,
                        AVG(revenue) as revenue,
                        COUNT(*) as measurement_count
                    FROM roi_data
                    GROUP BY content_id
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting ROI data: {e}")
            return pd.DataFrame()

    def train_roi_prediction_model(self) -> Dict[str, Any]:
        """Train a model to predict ROI based on content features."""
        try:
            data = self.prepare_performance_data()

            if data.empty or 'roi_percentage' not in data.columns:
                return {"error": "Insufficient data for model training"}

            # Prepare features
            feature_columns = ['avg_performance', 'sample_size', 'investment']
            features = data[feature_columns].fillna(0)
            target = data['roi_percentage'].fillna(0)

            # Split data
            split_idx = int(len(data) * 0.8)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = target[:split_idx], target[split_idx:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)

            # Train Linear Regression model
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)

            # Evaluate models
            rf_predictions = rf_model.predict(X_test_scaled)
            lr_predictions = lr_model.predict(X_test_scaled)

            rf_mse = mean_squared_error(y_test, rf_predictions)
            lr_mse = mean_squared_error(y_test, lr_predictions)

            rf_r2 = r2_score(y_test, rf_predictions)
            lr_r2 = r2_score(y_test, lr_predictions)

            # Select best model
            best_model = rf_model if rf_r2 > lr_r2 else lr_model
            best_model_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
            best_mse = rf_mse if rf_r2 > lr_r2 else lr_mse
            best_r2 = rf_r2 if rf_r2 > lr_r2 else lr_r2

            # Store models
            self.models['roi_prediction'] = {
                'model': best_model,
                'scaler': scaler,
                'features': feature_columns,
                'name': best_model_name
            }

            # Feature importance (for Random Forest)
            feature_importance = {}
            if best_model_name == "Random Forest":
                importance = best_model.feature_importances_
                feature_importance = dict(zip(feature_columns, importance))

            return {
                "model_name": best_model_name,
                "performance": {
                    "mse": best_mse,
                    "r2_score": best_r2,
                    "rmse": np.sqrt(best_mse)
                },
                "feature_importance": feature_importance,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "trained_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error training ROI prediction model: {e}")
            return {"error": str(e)}

    def predict_roi(self, avg_performance: float, sample_size: int, investment: float) -> Dict[str, Any]:
        """Predict ROI for new content based on features."""
        try:
            if 'roi_prediction' not in self.models:
                return {"error": "ROI prediction model not trained"}

            model_info = self.models['roi_prediction']
            model = model_info['model']
            scaler = model_info['scaler']

            # Prepare input
            input_data = np.array([[avg_performance, sample_size, investment]])
            input_scaled = scaler.transform(input_data)

            # Make prediction
            predicted_roi = model.predict(input_scaled)[0]

            # Calculate confidence interval (approximate)
            if hasattr(model, 'estimators_'):  # Random Forest
                predictions = [estimator.predict(input_scaled)[0] for estimator in model.estimators_]
                std_dev = np.std(predictions)
                confidence_interval = (predicted_roi - 1.96 * std_dev, predicted_roi + 1.96 * std_dev)
            else:
                confidence_interval = None

            return {
                "predicted_roi": predicted_roi,
                "confidence_interval": confidence_interval,
                "input_features": {
                    "avg_performance": avg_performance,
                    "sample_size": sample_size,
                    "investment": investment
                },
                "model_used": model_info['name'],
                "prediction_date": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error predicting ROI: {e}")
            return {"error": str(e)}

    def identify_optimization_opportunities(self) -> Dict[str, Any]:
        """Identify opportunities for content optimization."""
        try:
            data = self.prepare_performance_data()

            if data.empty:
                return {"error": "No data available for analysis"}

            opportunities = []

            # Low ROI, high investment opportunities
            high_investment_low_roi = data[
                (data['investment'] > data['investment'].quantile(0.75)) &
                (data['roi_percentage'] < data['roi_percentage'].quantile(0.25))
            ]

            if not high_investment_low_roi.empty:
                opportunities.append({
                    "type": "cost_optimization",
                    "description": "High investment, low ROI content needs cost optimization",
                    "affected_content": high_investment_low_roi['content_id'].tolist(),
                    "potential_savings": high_investment_low_roi['investment'].sum() * 0.3,
                    "priority": "high"
                })

            # Low performance, good investment opportunities
            low_performance_good_investment = data[
                (data['avg_performance'] < data['avg_performance'].quantile(0.25)) &
                (data['investment'] < data['investment'].quantile(0.5))
            ]

            if not low_performance_good_investment.empty:
                opportunities.append({
                    "type": "performance_improvement",
                    "description": "Low performance content with reasonable investment needs optimization",
                    "affected_content": low_performance_good_investment['content_id'].tolist(),
                    "potential_improvement": "20-50% performance increase possible",
                    "priority": "medium"
                })

            # Small sample size opportunities
            small_sample_size = data[data['sample_size'] < data['sample_size'].quantile(0.25)]

            if not small_sample_size.empty:
                opportunities.append({
                    "type": "sample_size_increase",
                    "description": "Content with small sample sizes needs broader testing",
                    "affected_content": small_sample_size['content_id'].tolist(),
                    "recommendation": "Increase testing duration or audience size",
                    "priority": "low"
                })

            return {
                "total_opportunities": len(opportunities),
                "opportunities": opportunities,
                "data_analyzed": len(data),
                "analysis_date": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
            return {"error": str(e)}

class AdvancedReportingEngine:
    """Advanced reporting and visualization engine."""

    def __init__(self, ab_testing_framework: ABTestingFramework,
                 roi_analyzer: ROIAnalyzer, predictive_analytics: PredictiveAnalytics):
        self.ab_testing = ab_testing_framework
        self.roi_analyzer = roi_analyzer
        self.predictive = predictive_analytics
        self.db_path = "advanced_reports.db"
        self.init_database()

    def init_database(self):
        """Initialize reporting database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS report_configs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        report_type TEXT NOT NULL,
                        parameters TEXT,
                        schedule TEXT,
                        recipients TEXT,
                        created_at TIMESTAMP,
                        last_generated TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS generated_reports (
                        id TEXT PRIMARY KEY,
                        config_id TEXT,
                        report_data TEXT,
                        file_path TEXT,
                        generated_at TIMESTAMP,
                        recipients_sent TEXT,
                        FOREIGN KEY (config_id) REFERENCES report_configs (id)
                    )
                """)

                conn.commit()

        except Exception as e:
            logger.error(f"Error initializing reporting database: {e}")
            raise

    def create_performance_dashboard(self, title: str = "Content Performance Dashboard") -> go.Figure:
        """Create an interactive performance dashboard."""
        try:
            # Get ROI trends
            roi_trends = self.roi_analyzer.get_roi_trends(30)

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ROI Trends Over Time', 'Investment vs Revenue',
                               'Top Performers', 'Performance Distribution'),
                specs=[[{"secondary_y": True}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "histogram"}]]
            )

            if roi_trends and 'daily_trends' in roi_trends:
                daily_data = roi_trends['daily_trends']

                if daily_data:
                    dates = [item['date'] for item in daily_data]
                    roi_values = [item['avg_roi'] for item in daily_data]
                    investments = [item['total_investment'] for item in daily_data]
                    revenues = [item['total_revenue'] for item in daily_data]

                    # ROI trend line
                    fig.add_trace(
                        go.Scatter(x=dates, y=roi_values, name='Average ROI (%)',
                                 line=dict(color='blue')),
                        row=1, col=1
                    )

                    # Investment vs Revenue scatter
                    fig.add_trace(
                        go.Scatter(x=investments, y=revenues, mode='markers',
                                 name='Investment vs Revenue',
                                 marker=dict(size=10, color='green')),
                        row=1, col=2
                    )

                # Top performers
                if 'top_performers' in roi_trends:
                    top_performers = roi_trends['top_performers'][:5]
                    content_ids = [p['content_id'][:8] for p in top_performers]
                    roi_percentages = [p['roi_percentage'] for p in top_performers]

                    fig.add_trace(
                        go.Bar(x=content_ids, y=roi_percentages, name='Top ROI %',
                              marker=dict(color='orange')),
                        row=2, col=1
                    )

                # Performance distribution
                all_roi = [p['roi_percentage'] for p in roi_trends.get('top_performers', []) +
                          roi_trends.get('bottom_performers', [])]

                if all_roi:
                    fig.add_trace(
                        go.Histogram(x=all_roi, name='ROI Distribution',
                                   marker=dict(color='purple')),
                        row=2, col=2
                    )

            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                showlegend=True,
                template="plotly_white"
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return go.Figure()

    def generate_ab_test_report(self, test_id: str) -> Dict[str, Any]:
        """Generate comprehensive A/B test report."""
        try:
            analysis = self.ab_testing.analyze_ab_test(test_id)

            if 'error' in analysis:
                return analysis

            # Create visualization
            fig = go.Figure()

            variant_stats = analysis['variant_statistics']
            variants = list(variant_stats.keys())
            means = [variant_stats[v]['mean'] for v in variants]
            stds = [variant_stats[v]['std'] for v in variants]

            # Add bar chart with error bars
            fig.add_trace(go.Bar(
                x=variants,
                y=means,
                error_y=dict(type='data', array=stds),
                name='Variant Performance'
            ))

            fig.update_layout(
                title=f"A/B Test Results: {analysis['test_name']}",
                xaxis_title="Variants",
                yaxis_title="Performance Metric",
                template="plotly_white"
            )

            # Generate insights
            insights = []
            statistical_analysis = analysis['statistical_analysis']

            if statistical_analysis.get('is_significant'):
                winner = statistical_analysis.get('winner', 'Unknown')
                improvement = statistical_analysis.get('relative_improvement', 0)
                insights.append(f"Statistically significant result found! Variant {winner} "
                               f"performs {improvement:.2f}% better.")
            else:
                insights.append("No statistically significant difference found between variants.")

            effect_size = statistical_analysis.get('effect_size', 0)
            if abs(effect_size) > 0.8:
                insights.append("Large effect size detected - results are practically significant.")
            elif abs(effect_size) > 0.5:
                insights.append("Medium effect size detected - results may have practical value.")
            else:
                insights.append("Small effect size - practical significance may be limited.")

            return {
                "analysis": analysis,
                "visualization": fig,
                "insights": insights,
                "recommendations": self._generate_ab_test_recommendations(analysis),
                "report_generated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating A/B test report: {e}")
            return {"error": str(e)}

    def _generate_ab_test_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on A/B test results."""
        recommendations = []

        statistical_analysis = analysis['statistical_analysis']

        if statistical_analysis.get('is_significant'):
            winner = statistical_analysis.get('winner')
            recommendations.append(f"Implement variant {winner} as the default option.")
            recommendations.append("Consider running a follow-up test to validate results.")
        else:
            recommendations.append("Consider running the test for a longer duration.")
            recommendations.append("Evaluate if the sample size was sufficient.")
            recommendations.append("Consider testing more dramatic variations.")

        # Check sample sizes
        sample_sizes = analysis.get('sample_sizes', {})
        min_sample_size = min(sample_sizes.values()) if sample_sizes else 0

        if min_sample_size < 100:
            recommendations.append("Increase sample size for more reliable results.")

        return recommendations

    def generate_roi_report(self, timeframe_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive ROI report."""
        try:
            roi_trends = self.roi_analyzer.get_roi_trends(timeframe_days)

            # Create ROI visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ROI Trends', 'Investment Distribution',
                               'Revenue Distribution', 'ROI vs Investment'),
                specs=[[{"secondary_y": True}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )

            if roi_trends and 'daily_trends' in roi_trends:
                daily_data = roi_trends['daily_trends']

                if daily_data:
                    dates = [item['date'] for item in daily_data]
                    roi_values = [item['avg_roi'] for item in daily_data]
                    investments = [item['total_investment'] for item in daily_data]
                    revenues = [item['total_revenue'] for item in daily_data]

                    # ROI trend
                    fig.add_trace(
                        go.Scatter(x=dates, y=roi_values, name='ROI %', line=dict(color='blue')),
                        row=1, col=1
                    )

                    # Investment distribution
                    fig.add_trace(
                        go.Histogram(x=investments, name='Investment Distribution',
                                   marker=dict(color='green')),
                        row=1, col=2
                    )

                    # Revenue distribution
                    fig.add_trace(
                        go.Histogram(x=revenues, name='Revenue Distribution',
                                   marker=dict(color='orange')),
                        row=2, col=1
                    )

                    # ROI vs Investment scatter
                    fig.add_trace(
                        go.Scatter(x=investments, y=roi_values, mode='markers',
                                 name='ROI vs Investment', marker=dict(color='purple')),
                        row=2, col=2
                    )

            fig.update_layout(
                title=f"ROI Analysis Report - {timeframe_days} Days",
                height=800,
                template="plotly_white"
            )

            # Generate insights
            insights = self._generate_roi_insights(roi_trends)

            return {
                "roi_data": roi_trends,
                "visualization": fig,
                "insights": insights,
                "optimization_opportunities": self.predictive.identify_optimization_opportunities(),
                "report_generated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating ROI report: {e}")
            return {"error": str(e)}

    def _generate_roi_insights(self, roi_trends: Dict[str, Any]) -> List[str]:
        """Generate insights from ROI data."""
        insights = []

        if not roi_trends or 'daily_trends' not in roi_trends:
            return ["Insufficient data for ROI insights."]

        daily_data = roi_trends['daily_trends']
        top_performers = roi_trends.get('top_performers', [])
        bottom_performers = roi_trends.get('bottom_performers', [])

        if daily_data:
            avg_roi = np.mean([item['avg_roi'] for item in daily_data])
            total_investment = sum([item['total_investment'] for item in daily_data])
            total_revenue = sum([item['total_revenue'] for item in daily_data])

            insights.append(f"Average ROI across all content: {avg_roi:.2f}%")
            insights.append(f"Total investment tracked: ${total_investment:,.2f}")
            insights.append(f"Total revenue generated: ${total_revenue:,.2f}")

            if avg_roi > 100:
                insights.append("Excellent overall ROI performance! Your content is highly profitable.")
            elif avg_roi > 50:
                insights.append("Good ROI performance with room for optimization.")
            elif avg_roi > 0:
                insights.append("Positive ROI but significant improvement opportunities exist.")
            else:
                insights.append("Negative ROI detected - immediate optimization required.")

        if top_performers:
            best_roi = max([p['roi_percentage'] for p in top_performers])
            insights.append(f"Best performing content achieved {best_roi:.2f}% ROI")

        if bottom_performers:
            worst_roi = min([p['roi_percentage'] for p in bottom_performers])
            insights.append(f"Lowest performing content had {worst_roi:.2f}% ROI")

        return insights

# Example usage and testing
if __name__ == "__main__":
    # Initialize systems
    ab_testing = ABTestingFramework()
    roi_analyzer = ROIAnalyzer()
    predictive = PredictiveAnalytics()
    reporting = AdvancedReportingEngine(ab_testing, roi_analyzer, predictive)

    # Example A/B test
    test_id = ab_testing.create_ab_test(
        name="Email Subject Line Test",
        test_type=TestType.CONTENT_VARIATION,
        description="Testing different email subject lines for newsletter",
        variants=[
            {"id": "control", "subject": "Weekly Newsletter #1"},
            {"id": "treatment", "subject": "🚀 Your Weekly Dose of Innovation"}
        ],
        target_metric=MetricType.ENGAGEMENT
    )

    print(f"Created A/B test: {test_id}")

    # Simulate test results
    import random
    for i in range(100):
        # Control group
        ab_testing.record_ab_test_result(
            test_id=test_id,
            variant_id="control",
            participant_id=f"user_{i}",
            metric_value=random.normalvariate(0.15, 0.05)  # 15% open rate
        )

        # Treatment group (slightly better performance)
        ab_testing.record_ab_test_result(
            test_id=test_id,
            variant_id="treatment",
            participant_id=f"user_{i+100}",
            metric_value=random.normalvariate(0.18, 0.05)  # 18% open rate
        )

    # Analyze results
    results = ab_testing.analyze_ab_test(test_id)
    print(f"A/B Test Results: {json.dumps(results, indent=2)}")

    # Generate comprehensive report
    report = reporting.generate_ab_test_report(test_id)
    if 'insights' in report:
        print("Insights:")
        for insight in report['insights']:
            print(f"- {insight}")

    # ROI tracking example
    content_id = "newsletter_001"
    roi_analyzer.track_investment(content_id, "campaign_001", "design", 500, "Newsletter design costs")
    roi_analyzer.track_investment(content_id, "campaign_001", "distribution", 200, "Email platform costs")
    roi_analyzer.track_revenue(content_id, "subscriptions", 1500, 0.8)
    roi_analyzer.track_revenue(content_id, "affiliate", 300, 0.5)

    roi_analysis = roi_analyzer.calculate_roi(content_id)
    print(f"ROI Analysis: {json.dumps(roi_analysis, indent=2)}")

    print("Advanced Analytics & Reporting System demonstration completed!")
