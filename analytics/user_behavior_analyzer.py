"""
User Behavior Analyzer for LangGraph 101 AI Platform

Analyzes user interaction patterns, preferences, and behavior to provide insights
for improving user experience and platform optimization.
"""

import sqlite3
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

from .analytics_logger import get_analytics_logger

@dataclass
class UserProfile:
    """Comprehensive user profile with behavior insights"""
    user_id: str
    total_sessions: int
    avg_session_duration: float
    total_requests: int
    avg_requests_per_session: float
    preferred_models: List[Dict[str, Any]]
    feature_usage: Dict[str, int]
    peak_activity_hours: List[int]
    device_preferences: Dict[str, int]
    error_rate: float
    engagement_score: float
    user_segment: str
    last_activity: str

@dataclass
class BehaviorInsight:
    """Individual behavior insight or pattern"""
    insight_type: str
    title: str
    description: str
    impact_score: float
    affected_users: int
    recommendation: str
    data: Dict[str, Any]

class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns and generates actionable insights"""
    
    def __init__(self):
        self.analytics_logger = get_analytics_logger()
        self.insight_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
    
    def analyze_user_profile(self, user_id: str, days: int = 30) -> Optional[UserProfile]:
        """Generate comprehensive user profile with behavior analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get user events
        events = self.analytics_logger.get_events(
            user_id=user_id,
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            limit=10000
        )
        
        if not events:
            return None
        
        # Analyze sessions
        sessions = self._group_events_by_session(events)
        session_durations = []
        requests_per_session = []
        
        for session_events in sessions.values():
            if len(session_events) >= 2:
                start_time = datetime.fromisoformat(session_events[-1]['timestamp'])
                end_time = datetime.fromisoformat(session_events[0]['timestamp'])
                duration = (end_time - start_time).total_seconds() / 60  # minutes
                session_durations.append(duration)
            
            requests_per_session.append(len(session_events))
        
        # Model preferences
        model_usage = Counter(
            event['model_id'] for event in events 
            if event['model_id']
        )
        preferred_models = [
            {
                'model_id': model_id,
                'usage_count': count,
                'usage_percentage': (count / len(events)) * 100
            }
            for model_id, count in model_usage.most_common(5)
        ]
        
        # Feature usage analysis
        feature_usage = self._analyze_feature_usage(events)
        
        # Peak activity hours
        activity_hours = [
            datetime.fromisoformat(event['timestamp']).hour
            for event in events
        ]
        peak_hours = [
            hour for hour, count in Counter(activity_hours).most_common(3)
        ]
        
        # Device preferences
        device_info = defaultdict(int)
        for event in events:
            if event.get('user_agent'):
                device_type = self._parse_device_type(event['user_agent'])
                device_info[device_type] += 1
        
        # Error rate
        error_count = sum(1 for event in events if event.get('error_code'))
        error_rate = (error_count / len(events)) * 100 if events else 0
        
        # Engagement score (0-100)
        engagement_score = self._calculate_engagement_score(
            len(sessions),
            statistics.mean(session_durations) if session_durations else 0,
            statistics.mean(requests_per_session) if requests_per_session else 0,
            len(feature_usage),
            error_rate
        )
        
        # User segmentation
        user_segment = self._determine_user_segment(
            engagement_score,
            len(sessions),
            statistics.mean(requests_per_session) if requests_per_session else 0
        )
        
        return UserProfile(
            user_id=user_id,
            total_sessions=len(sessions),
            avg_session_duration=statistics.mean(session_durations) if session_durations else 0,
            total_requests=len(events),
            avg_requests_per_session=statistics.mean(requests_per_session) if requests_per_session else 0,
            preferred_models=preferred_models,
            feature_usage=feature_usage,
            peak_activity_hours=peak_hours,
            device_preferences=dict(device_info),
            error_rate=error_rate,
            engagement_score=engagement_score,
            user_segment=user_segment,
            last_activity=events[0]['timestamp'] if events else ""
        )
    
    def _group_events_by_session(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Group events by session ID"""
        sessions = defaultdict(list)
        for event in events:
            session_id = event.get('session_id', 'unknown')
            sessions[session_id].append(event)
        return dict(sessions)
    
    def _analyze_feature_usage(self, events: List[Dict]) -> Dict[str, int]:
        """Analyze feature usage patterns from events"""
        feature_usage = defaultdict(int)
        
        for event in events:
            event_type = event['event_type']
            details = json.loads(event.get('details_json', '{}')) if event.get('details_json') else {}
            
            # Map event types to features
            if event_type == 'model_interaction':
                feature_usage['AI Model Usage'] += 1
            elif event_type == 'content_generation':
                feature_usage['Content Generation'] += 1
            elif event_type == 'template_usage':
                feature_usage['Template Usage'] += 1
            elif event_type == 'export_data':
                feature_usage['Data Export'] += 1
            elif event_type == 'dashboard_view':
                feature_usage['Dashboard'] += 1
            
            # Analyze specific features from details
            if 'feature_name' in details:
                feature_usage[details['feature_name']] += 1
        
        return dict(feature_usage)
    
    def _parse_device_type(self, user_agent: str) -> str:
        """Parse device type from user agent string"""
        user_agent_lower = user_agent.lower()
        
        if 'mobile' in user_agent_lower or 'android' in user_agent_lower or 'iphone' in user_agent_lower:
            return 'Mobile'
        elif 'tablet' in user_agent_lower or 'ipad' in user_agent_lower:
            return 'Tablet'
        else:
            return 'Desktop'
    
    def _calculate_engagement_score(self, 
                                  sessions: int,
                                  avg_session_duration: float,
                                  avg_requests_per_session: float,
                                  feature_diversity: int,
                                  error_rate: float) -> float:
        """Calculate user engagement score (0-100)"""
        # Base scores
        session_score = min(sessions * 2, 30)  # Max 30 points for sessions
        duration_score = min(avg_session_duration / 2, 25)  # Max 25 points for duration
        request_score = min(avg_requests_per_session * 3, 20)  # Max 20 points for requests
        diversity_score = min(feature_diversity * 2, 15)  # Max 15 points for feature diversity
        error_penalty = min(error_rate * 0.5, 10)  # Max 10 point penalty for errors
        
        engagement_score = session_score + duration_score + request_score + diversity_score - error_penalty
        return max(0, min(100, engagement_score))
    
    def _determine_user_segment(self, 
                              engagement_score: float,
                              total_sessions: int,
                              avg_requests_per_session: float) -> str:
        """Determine user segment based on behavior patterns"""
        if engagement_score >= 80 and total_sessions >= 20:
            return "Power User"
        elif engagement_score >= 60 and avg_requests_per_session >= 5:
            return "Active User"
        elif engagement_score >= 40 or total_sessions >= 5:
            return "Regular User"
        elif total_sessions >= 2:
            return "Casual User"
        else:
            return "New User"
    
    def generate_behavior_insights(self, days: int = 7) -> List[BehaviorInsight]:
        """Generate actionable behavior insights for the platform"""
        cache_key = f"behavior_insights_{days}"
        
        # Check cache
        if cache_key in self.insight_cache:
            cached_time, insights = self.insight_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return insights
        
        insights = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all events for analysis
        events = self.analytics_logger.get_events(
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            limit=50000
        )
        
        if not events:
            return insights
        
        # Analyze model usage patterns
        insights.extend(self._analyze_model_usage_patterns(events))
        
        # Analyze feature adoption
        insights.extend(self._analyze_feature_adoption(events))
        
        # Analyze user journey patterns
        insights.extend(self._analyze_user_journey_patterns(events))
        
        # Analyze performance impact on behavior
        insights.extend(self._analyze_performance_behavior_correlation(events))
        
        # Cache results
        self.insight_cache[cache_key] = (datetime.now(), insights)
        
        return insights
    
    def _analyze_model_usage_patterns(self, events: List[Dict]) -> List[BehaviorInsight]:
        """Analyze model usage patterns and preferences"""
        insights = []
        
        # Model usage distribution
        model_usage = Counter(
            event['model_id'] for event in events 
            if event['model_id']
        )
        
        if model_usage:
            total_usage = sum(model_usage.values())
            most_used = model_usage.most_common(1)[0]
            
            # Check for model dominance
            if most_used[1] / total_usage > 0.7:
                insights.append(BehaviorInsight(
                    insight_type="model_dominance",
                    title="Single Model Dominance Detected",
                    description=f"Model '{most_used[0]}' accounts for {most_used[1]/total_usage*100:.1f}% of all usage",
                    impact_score=0.8,
                    affected_users=len(set(event['user_id'] for event in events if event['model_id'] == most_used[0])),
                    recommendation="Consider promoting alternative models or investigating why other models aren't being used",
                    data={"dominant_model": most_used[0], "usage_percentage": most_used[1]/total_usage*100}
                ))
            
            # Check for model switching patterns
            user_model_switches = self._analyze_model_switching(events)
            if user_model_switches:
                avg_switches = statistics.mean(user_model_switches.values())
                if avg_switches > 3:
                    insights.append(BehaviorInsight(
                        insight_type="model_switching",
                        title="High Model Switching Activity",
                        description=f"Users switch models an average of {avg_switches:.1f} times per session",
                        impact_score=0.6,
                        affected_users=len(user_model_switches),
                        recommendation="Consider implementing model recommendations or improving model selection UI",
                        data={"avg_switches_per_user": avg_switches}
                    ))
        
        return insights
    
    def _analyze_feature_adoption(self, events: List[Dict]) -> List[BehaviorInsight]:
        """Analyze feature adoption and usage patterns"""
        insights = []
        
        # Feature usage analysis
        feature_usage = defaultdict(set)  # feature -> set of users
        for event in events:
            if event['event_type'] and event['user_id']:
                feature_usage[event['event_type']].add(event['user_id'])
        
        total_users = len(set(event['user_id'] for event in events if event['user_id']))
        
        # Identify low-adoption features
        for feature, users in feature_usage.items():
            adoption_rate = len(users) / total_users if total_users > 0 else 0
            
            if adoption_rate < 0.2 and len(users) > 5:  # Less than 20% adoption
                insights.append(BehaviorInsight(
                    insight_type="low_feature_adoption",
                    title=f"Low Adoption: {feature}",
                    description=f"Only {adoption_rate*100:.1f}% of users are using {feature}",
                    impact_score=0.7,
                    affected_users=total_users - len(users),
                    recommendation="Consider improving discoverability, documentation, or UX for this feature",
                    data={"feature": feature, "adoption_rate": adoption_rate}
                ))
        
        return insights
    
    def _analyze_user_journey_patterns(self, events: List[Dict]) -> List[BehaviorInsight]:
        """Analyze user journey and flow patterns"""
        insights = []
        
        # Group events by user and session
        user_sessions = defaultdict(lambda: defaultdict(list))
        for event in events:
            if event['user_id'] and event['session_id']:
                user_sessions[event['user_id']][event['session_id']].append(event)
        
        # Analyze drop-off patterns
        short_sessions = 0
        total_sessions = 0
        
        for user_id, sessions in user_sessions.items():
            for session_id, session_events in sessions.items():
                total_sessions += 1
                if len(session_events) <= 2:  # Very short sessions
                    short_sessions += 1
        
        if total_sessions > 0:
            drop_off_rate = short_sessions / total_sessions
            if drop_off_rate > 0.3:  # More than 30% drop-off
                insights.append(BehaviorInsight(
                    insight_type="high_dropoff",
                    title="High Session Drop-off Rate",
                    description=f"{drop_off_rate*100:.1f}% of sessions end with minimal interaction",
                    impact_score=0.9,
                    affected_users=len(user_sessions),
                    recommendation="Investigate onboarding flow and initial user experience",
                    data={"dropoff_rate": drop_off_rate, "short_sessions": short_sessions}
                ))
        
        return insights
    
    def _analyze_performance_behavior_correlation(self, events: List[Dict]) -> List[BehaviorInsight]:
        """Analyze correlation between performance and user behavior"""
        insights = []
        
        # Group events by response time
        fast_responses = []  # < 1 second
        slow_responses = []  # > 3 seconds
        
        for event in events:
            if event.get('response_time_ms'):
                if event['response_time_ms'] < 1000:
                    fast_responses.append(event)
                elif event['response_time_ms'] > 3000:
                    slow_responses.append(event)
        
        # Analyze user behavior correlation with performance
        if fast_responses and slow_responses:
            # Check if slow responses lead to session abandonment
            slow_response_sessions = set(
                event['session_id'] for event in slow_responses
                if event['session_id']
            )
            
            # Count subsequent events after slow responses
            abandonment_rate = self._calculate_abandonment_after_slow_responses(
                events, slow_response_sessions
            )
            
            if abandonment_rate > 0.4:  # More than 40% abandonment
                insights.append(BehaviorInsight(
                    insight_type="performance_abandonment",
                    title="Performance Issues Leading to Abandonment",
                    description=f"{abandonment_rate*100:.1f}% of sessions with slow responses are abandoned",
                    impact_score=0.95,
                    affected_users=len(slow_response_sessions),
                    recommendation="Prioritize performance optimization to reduce user abandonment",
                    data={"abandonment_rate": abandonment_rate, "slow_responses": len(slow_responses)}
                ))
        
        return insights
    
    def _analyze_model_switching(self, events: List[Dict]) -> Dict[str, int]:
        """Analyze model switching patterns per user"""
        user_switches = defaultdict(int)
        user_last_model = {}
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        for event in sorted_events:
            if event['user_id'] and event['model_id']:
                user_id = event['user_id']
                current_model = event['model_id']
                
                if user_id in user_last_model and user_last_model[user_id] != current_model:
                    user_switches[user_id] += 1
                
                user_last_model[user_id] = current_model
        
        return dict(user_switches)
    
    def _calculate_abandonment_after_slow_responses(self, 
                                                   events: List[Dict], 
                                                   slow_sessions: set) -> float:
        """Calculate abandonment rate after slow responses"""
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        session_activity = defaultdict(list)
        for event in sorted_events:
            if event['session_id']:
                session_activity[event['session_id']].append(event)
        
        abandoned_sessions = 0
        total_slow_sessions = len(slow_sessions)
        
        for session_id in slow_sessions:
            session_events = session_activity.get(session_id, [])
            
            # Find first slow response
            slow_event_index = None
            for i, event in enumerate(session_events):
                if event.get('response_time_ms', 0) > 3000:
                    slow_event_index = i
                    break
            
            # Check if there are few events after the slow response
            if slow_event_index is not None:
                events_after_slow = len(session_events) - slow_event_index - 1
                if events_after_slow <= 1:  # Abandoned if ≤1 events after slow response
                    abandoned_sessions += 1
        
        return abandoned_sessions / total_slow_sessions if total_slow_sessions > 0 else 0
    
    def get_user_segments_distribution(self, days: int = 30) -> Dict[str, Any]:
        """Get distribution of user segments"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all unique users in the time period
        events = self.analytics_logger.get_events(
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            limit=50000
        )
        
        unique_users = set(event['user_id'] for event in events if event['user_id'])
        
        segment_distribution = defaultdict(int)
        user_profiles = []
        
        for user_id in unique_users:
            profile = self.analyze_user_profile(user_id, days)
            if profile:
                segment_distribution[profile.user_segment] += 1
                user_profiles.append(profile)
        
        return {
            'distribution': dict(segment_distribution),
            'total_users': len(unique_users),
            'user_profiles': [asdict(profile) for profile in user_profiles[:100]]  # Limit for performance
        }

# Global instance
_user_behavior_analyzer = None

def get_user_behavior_analyzer() -> UserBehaviorAnalyzer:
    """Get the global user behavior analyzer instance"""
    global _user_behavior_analyzer
    if _user_behavior_analyzer is None:
        _user_behavior_analyzer = UserBehaviorAnalyzer()
    return _user_behavior_analyzer
