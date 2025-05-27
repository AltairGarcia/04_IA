"""
Content Calendar Integration System
==================================

A comprehensive system for content calendar management, workflow scheduling,
automated publishing, and content lifecycle management.

Features:
- Visual content calendar with drag-and-drop scheduling
- Automated content generation and publishing
- Multi-channel content distribution
- Content lifecycle management
- Team collaboration and approval workflows
- Integration with external calendar systems
- Automated reminders and notifications
"""

import json
import sqlite3
import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import uuid
import calendar
import pytz
from icalendar import Calendar, Event
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import schedule
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentStatus(Enum):
    """Content status in the workflow."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    CANCELLED = "cancelled"

class Priority(Enum):
    """Content priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ContentType(Enum):
    """Content types for calendar management."""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL_NEWSLETTER = "email_newsletter"
    VIDEO = "video"
    PODCAST = "podcast"
    WEBINAR = "webinar"
    PRESS_RELEASE = "press_release"
    CASE_STUDY = "case_study"
    WHITE_PAPER = "white_paper"
    INFOGRAPHIC = "infographic"

class PublishingChannel(Enum):
    """Publishing channels."""
    WEBSITE = "website"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    EMAIL = "email"
    MEDIUM = "medium"
    WORDPRESS = "wordpress"

class NotificationType(Enum):
    """Notification types."""
    REMINDER = "reminder"
    APPROVAL_REQUEST = "approval_request"
    DEADLINE_WARNING = "deadline_warning"
    PUBLICATION_SUCCESS = "publication_success"
    PUBLICATION_FAILURE = "publication_failure"

@dataclass
class CalendarEvent:
    """Calendar event data structure."""
    id: str
    title: str
    content_type: ContentType
    description: str
    start_date: datetime
    end_date: datetime
    status: ContentStatus
    priority: Priority
    assigned_to: str
    team_members: List[str]
    channels: List[PublishingChannel]
    content_data: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    template_id: Optional[str]
    parent_campaign_id: Optional[str]

@dataclass
class ContentCampaign:
    """Content campaign data structure."""
    id: str
    name: str
    description: str
    start_date: date
    end_date: date
    status: str
    goals: List[str]
    target_audience: str
    content_events: List[str]
    budget: float
    spent: float
    performance_metrics: Dict[str, Any]
    created_at: datetime
    created_by: str

@dataclass
class PublishingRule:
    """Automated publishing rule."""
    id: str
    name: str
    content_type: ContentType
    channels: List[PublishingChannel]
    schedule_pattern: Dict[str, Any]
    template_id: Optional[str]
    auto_generate: bool
    approval_required: bool
    created_at: datetime
    is_active: bool

@dataclass
class TeamMember:
    """Team member data structure."""
    id: str
    name: str
    email: str
    role: str
    permissions: List[str]
    timezone: str
    notification_preferences: Dict[str, bool]
    created_at: datetime

class CalendarDatabase:
    """Database management for calendar system."""

    def __init__(self, db_path: str = "content_calendar.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize calendar database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Calendar events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS calendar_events (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        description TEXT,
                        start_date TIMESTAMP,
                        end_date TIMESTAMP,
                        status TEXT,
                        priority TEXT,
                        assigned_to TEXT,
                        team_members TEXT,
                        channels TEXT,
                        content_data TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        template_id TEXT,
                        parent_campaign_id TEXT
                    )
                """)

                # Content campaigns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_campaigns (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        start_date DATE,
                        end_date DATE,
                        status TEXT,
                        goals TEXT,
                        target_audience TEXT,
                        content_events TEXT,
                        budget REAL,
                        spent REAL,
                        performance_metrics TEXT,
                        created_at TIMESTAMP,
                        created_by TEXT
                    )
                """)

                # Publishing rules table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS publishing_rules (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        content_type TEXT,
                        channels TEXT,
                        schedule_pattern TEXT,
                        template_id TEXT,
                        auto_generate BOOLEAN,
                        approval_required BOOLEAN,
                        created_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                # Team members table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS team_members (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE,
                        role TEXT,
                        permissions TEXT,
                        timezone TEXT,
                        notification_preferences TEXT,
                        created_at TIMESTAMP
                    )
                """)

                # Notifications table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id TEXT PRIMARY KEY,
                        recipient_id TEXT,
                        notification_type TEXT,
                        title TEXT,
                        message TEXT,
                        related_event_id TEXT,
                        sent_at TIMESTAMP,
                        read_at TIMESTAMP,
                        metadata TEXT
                    )
                """)

                # Publishing history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS publishing_history (
                        id TEXT PRIMARY KEY,
                        event_id TEXT,
                        channel TEXT,
                        published_at TIMESTAMP,
                        status TEXT,
                        response_data TEXT,
                        error_message TEXT,
                        FOREIGN KEY (event_id) REFERENCES calendar_events (id)
                    )
                """)

                conn.commit()
                logger.info("Calendar database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing calendar database: {e}")
            raise

class ContentCalendarManager:
    """Main content calendar management system."""

    def __init__(self, db_path: str = "content_calendar.db"):
        self.db = CalendarDatabase(db_path)
        self.db_path = db_path
        self.scheduler_thread = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=5)

    def create_calendar_event(self, title: str, content_type: ContentType, description: str,
                            start_date: datetime, end_date: datetime, assigned_to: str,
                            channels: List[PublishingChannel], priority: Priority = Priority.MEDIUM,
                            template_id: Optional[str] = None, campaign_id: Optional[str] = None) -> str:
        """Create a new calendar event."""
        event_id = str(uuid.uuid4())

        event = CalendarEvent(
            id=event_id,
            title=title,
            content_type=content_type,
            description=description,
            start_date=start_date,
            end_date=end_date,
            status=ContentStatus.DRAFT,
            priority=priority,
            assigned_to=assigned_to,
            team_members=[assigned_to],
            channels=channels,
            content_data={},
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            template_id=template_id,
            parent_campaign_id=campaign_id
        )

        self.save_calendar_event(event)
        return event_id

    def save_calendar_event(self, event: CalendarEvent):
        """Save calendar event to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO calendar_events (
                        id, title, content_type, description, start_date, end_date,
                        status, priority, assigned_to, team_members, channels,
                        content_data, metadata, created_at, updated_at,
                        template_id, parent_campaign_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id, event.title, event.content_type.value, event.description,
                    event.start_date, event.end_date, event.status.value, event.priority.value,
                    event.assigned_to, json.dumps(event.team_members),
                    json.dumps([c.value for c in event.channels]),
                    json.dumps(event.content_data), json.dumps(event.metadata),
                    event.created_at, event.updated_at, event.template_id, event.parent_campaign_id
                ))

                conn.commit()
                logger.info(f"Calendar event '{event.title}' saved successfully")

        except Exception as e:
            logger.error(f"Error saving calendar event: {e}")
            raise

    def get_calendar_event(self, event_id: str) -> Optional[CalendarEvent]:
        """Get calendar event by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,))
                row = cursor.fetchone()

                if row:
                    return CalendarEvent(
                        id=row[0], title=row[1], content_type=ContentType(row[2]),
                        description=row[3], start_date=datetime.fromisoformat(row[4]),
                        end_date=datetime.fromisoformat(row[5]), status=ContentStatus(row[6]),
                        priority=Priority(row[7]), assigned_to=row[8],
                        team_members=json.loads(row[9]),
                        channels=[PublishingChannel(c) for c in json.loads(row[10])],
                        content_data=json.loads(row[11]), metadata=json.loads(row[12]),
                        created_at=datetime.fromisoformat(row[13]),
                        updated_at=datetime.fromisoformat(row[14]),
                        template_id=row[15], parent_campaign_id=row[16]
                    )

        except Exception as e:
            logger.error(f"Error getting calendar event: {e}")

        return None

    def get_calendar_events(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          status: Optional[ContentStatus] = None,
                          assigned_to: Optional[str] = None) -> List[CalendarEvent]:
        """Get calendar events with optional filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM calendar_events WHERE 1=1"
                params = []

                if start_date:
                    query += " AND start_date >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND end_date <= ?"
                    params.append(end_date)

                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                if assigned_to:
                    query += " AND assigned_to = ?"
                    params.append(assigned_to)

                query += " ORDER BY start_date"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    event = CalendarEvent(
                        id=row[0], title=row[1], content_type=ContentType(row[2]),
                        description=row[3], start_date=datetime.fromisoformat(row[4]),
                        end_date=datetime.fromisoformat(row[5]), status=ContentStatus(row[6]),
                        priority=Priority(row[7]), assigned_to=row[8],
                        team_members=json.loads(row[9]),
                        channels=[PublishingChannel(c) for c in json.loads(row[10])],
                        content_data=json.loads(row[11]), metadata=json.loads(row[12]),
                        created_at=datetime.fromisoformat(row[13]),
                        updated_at=datetime.fromisoformat(row[14]),
                        template_id=row[15], parent_campaign_id=row[16]
                    )
                    events.append(event)

                return events

        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []

    def update_event_status(self, event_id: str, status: ContentStatus):
        """Update event status."""
        event = self.get_calendar_event(event_id)
        if event:
            event.status = status
            event.updated_at = datetime.now()
            self.save_calendar_event(event)

            # Send notifications for status changes
            self._send_status_change_notification(event)

    def create_campaign(self, name: str, description: str, start_date: date, end_date: date,
                       goals: List[str], target_audience: str, budget: float, created_by: str) -> str:
        """Create a new content campaign."""
        campaign_id = str(uuid.uuid4())

        campaign = ContentCampaign(
            id=campaign_id,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            status="active",
            goals=goals,
            target_audience=target_audience,
            content_events=[],
            budget=budget,
            spent=0.0,
            performance_metrics={},
            created_at=datetime.now(),
            created_by=created_by
        )

        self.save_campaign(campaign)
        return campaign_id

    def save_campaign(self, campaign: ContentCampaign):
        """Save campaign to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO content_campaigns (
                        id, name, description, start_date, end_date, status,
                        goals, target_audience, content_events, budget, spent,
                        performance_metrics, created_at, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    campaign.id, campaign.name, campaign.description,
                    campaign.start_date, campaign.end_date, campaign.status,
                    json.dumps(campaign.goals), campaign.target_audience,
                    json.dumps(campaign.content_events), campaign.budget, campaign.spent,
                    json.dumps(campaign.performance_metrics), campaign.created_at, campaign.created_by
                ))

                conn.commit()
                logger.info(f"Campaign '{campaign.name}' saved successfully")

        except Exception as e:
            logger.error(f"Error saving campaign: {e}")
            raise

    def get_monthly_calendar(self, year: int, month: int) -> Dict[str, Any]:
        """Get calendar view for a specific month."""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        events = self.get_calendar_events(start_date, end_date)

        # Group events by date
        calendar_data = {}
        for event in events:
            event_date = event.start_date.date()
            date_str = event_date.isoformat()

            if date_str not in calendar_data:
                calendar_data[date_str] = []

            calendar_data[date_str].append({
                "id": event.id,
                "title": event.title,
                "content_type": event.content_type.value,
                "status": event.status.value,
                "priority": event.priority.value,
                "assigned_to": event.assigned_to,
                "channels": [c.value for c in event.channels],
                "start_time": event.start_date.strftime("%H:%M"),
                "end_time": event.end_date.strftime("%H:%M")
            })

        # Generate calendar grid
        cal = calendar.Calendar(firstweekday=0)  # Monday first
        month_days = cal.monthdayscalendar(year, month)

        return {
            "year": year,
            "month": month,
            "month_name": calendar.month_name[month],
            "calendar_grid": month_days,
            "events": calendar_data,
            "total_events": len(events),
            "status_summary": self._get_status_summary(events)
        }

    def _get_status_summary(self, events: List[CalendarEvent]) -> Dict[str, int]:
        """Get summary of event statuses."""
        summary = {}
        for status in ContentStatus:
            summary[status.value] = len([e for e in events if e.status == status])
        return summary

    def _send_status_change_notification(self, event: CalendarEvent):
        """Send notification for status change."""
        try:
            # Create notification
            notification_id = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO notifications (
                        id, recipient_id, notification_type, title, message,
                        related_event_id, sent_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification_id, event.assigned_to, NotificationType.REMINDER.value,
                    f"Status Update: {event.title}",
                    f"Event '{event.title}' status changed to {event.status.value}",
                    event.id, datetime.now(), json.dumps({})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error sending status change notification: {e}")

class AutomationEngine:
    """Content calendar automation engine."""

    def __init__(self, calendar_manager: ContentCalendarManager):
        self.calendar_manager = calendar_manager
        self.db_path = calendar_manager.db_path
        self.scheduler = schedule
        self.running = False
        self.scheduler_thread = None

    def create_publishing_rule(self, name: str, content_type: ContentType,
                             channels: List[PublishingChannel], schedule_pattern: Dict[str, Any],
                             template_id: Optional[str] = None, auto_generate: bool = False,
                             approval_required: bool = True) -> str:
        """Create automated publishing rule."""
        rule_id = str(uuid.uuid4())

        rule = PublishingRule(
            id=rule_id,
            name=name,
            content_type=content_type,
            channels=channels,
            schedule_pattern=schedule_pattern,
            template_id=template_id,
            auto_generate=auto_generate,
            approval_required=approval_required,
            created_at=datetime.now(),
            is_active=True
        )

        self.save_publishing_rule(rule)
        return rule_id

    def save_publishing_rule(self, rule: PublishingRule):
        """Save publishing rule to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO publishing_rules (
                        id, name, content_type, channels, schedule_pattern,
                        template_id, auto_generate, approval_required,
                        created_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.id, rule.name, rule.content_type.value,
                    json.dumps([c.value for c in rule.channels]),
                    json.dumps(rule.schedule_pattern), rule.template_id,
                    rule.auto_generate, rule.approval_required,
                    rule.created_at, rule.is_active
                ))

                conn.commit()
                logger.info(f"Publishing rule '{rule.name}' saved successfully")

        except Exception as e:
            logger.error(f"Error saving publishing rule: {e}")
            raise

    def start_automation(self):
        """Start the automation engine."""
        if self.running:
            return

        self.running = True

        # Schedule daily checks
        self.scheduler.every().hour.do(self._check_scheduled_events)
        self.scheduler.every().day.at("09:00").do(self._send_daily_reminders)
        self.scheduler.every().day.at("17:00").do(self._check_deadlines)

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info("Content calendar automation started")

    def stop_automation(self):
        """Stop the automation engine."""
        self.running = False
        self.scheduler.clear()
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Content calendar automation stopped")

    def _run_scheduler(self):
        """Run the scheduler in a separate thread."""
        while self.running:
            self.scheduler.run_pending()
            time.sleep(60)  # Check every minute

    def _check_scheduled_events(self):
        """Check for events ready to be published."""
        now = datetime.now()

        # Get events scheduled for publication in the next hour
        scheduled_events = self.calendar_manager.get_calendar_events(
            start_date=now,
            end_date=now + timedelta(hours=1),
            status=ContentStatus.SCHEDULED
        )

        for event in scheduled_events:
            if event.start_date <= now:
                self._execute_publication(event)

    def _execute_publication(self, event: CalendarEvent):
        """Execute publication for an event."""
        try:
            logger.info(f"Publishing event: {event.title}")

            # Update status
            self.calendar_manager.update_event_status(event.id, ContentStatus.PUBLISHED)

            # Publish to each channel
            for channel in event.channels:
                self._publish_to_channel(event, channel)

        except Exception as e:
            logger.error(f"Error publishing event {event.id}: {e}")
            self.calendar_manager.update_event_status(event.id, ContentStatus.DRAFT)

    def _publish_to_channel(self, event: CalendarEvent, channel: PublishingChannel):
        """Publish content to a specific channel."""
        try:
            # Record publishing attempt
            history_id = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Simulate publishing (in real implementation, integrate with actual APIs)
                success = True
                response_data = {"status": "success", "published_url": f"https://{channel.value}.com/content/{event.id}"}
                error_message = None

                # For demonstration, simulate some failures
                import random
                if random.random() < 0.1:  # 10% failure rate
                    success = False
                    error_message = f"Failed to publish to {channel.value}: API error"
                    response_data = {"status": "error"}

                cursor.execute("""
                    INSERT INTO publishing_history (
                        id, event_id, channel, published_at, status,
                        response_data, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    history_id, event.id, channel.value, datetime.now(),
                    "success" if success else "error",
                    json.dumps(response_data), error_message
                ))

                conn.commit()

                if success:
                    logger.info(f"Successfully published to {channel.value}")
                else:
                    logger.error(f"Failed to publish to {channel.value}: {error_message}")

        except Exception as e:
            logger.error(f"Error publishing to {channel.value}: {e}")

    def _send_daily_reminders(self):
        """Send daily reminders to team members."""
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)

        # Get events for today and tomorrow
        today_events = self.calendar_manager.get_calendar_events(
            start_date=datetime.combine(today, datetime.min.time()),
            end_date=datetime.combine(today, datetime.max.time())
        )

        tomorrow_events = self.calendar_manager.get_calendar_events(
            start_date=datetime.combine(tomorrow, datetime.min.time()),
            end_date=datetime.combine(tomorrow, datetime.max.time())
        )

        # Group by assigned person
        assignments = {}

        for event in today_events + tomorrow_events:
            if event.assigned_to not in assignments:
                assignments[event.assigned_to] = {"today": [], "tomorrow": []}

            if event.start_date.date() == today:
                assignments[event.assigned_to]["today"].append(event)
            else:
                assignments[event.assigned_to]["tomorrow"].append(event)

        # Send reminders
        for person, events in assignments.items():
            self._send_reminder_notification(person, events)

    def _send_reminder_notification(self, person: str, events: Dict[str, List[CalendarEvent]]):
        """Send reminder notification to a person."""
        try:
            today_count = len(events["today"])
            tomorrow_count = len(events["tomorrow"])

            if today_count == 0 and tomorrow_count == 0:
                return

            message = f"Daily Reminder:\n"
            if today_count > 0:
                message += f"- {today_count} event(s) today\n"
            if tomorrow_count > 0:
                message += f"- {tomorrow_count} event(s) tomorrow\n"

            notification_id = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO notifications (
                        id, recipient_id, notification_type, title, message,
                        sent_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification_id, person, NotificationType.REMINDER.value,
                    "Daily Content Reminder", message, datetime.now(), json.dumps({})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error sending reminder notification: {e}")

    def _check_deadlines(self):
        """Check for approaching deadlines."""
        now = datetime.now()
        warning_time = now + timedelta(hours=24)  # 24-hour warning

        # Get events with approaching deadlines
        upcoming_events = self.calendar_manager.get_calendar_events(
            start_date=now,
            end_date=warning_time
        )

        for event in upcoming_events:
            if event.status in [ContentStatus.DRAFT, ContentStatus.IN_REVIEW]:
                time_remaining = event.start_date - now
                hours_remaining = time_remaining.total_seconds() / 3600

                if hours_remaining <= 24:
                    self._send_deadline_warning(event, hours_remaining)

    def _send_deadline_warning(self, event: CalendarEvent, hours_remaining: float):
        """Send deadline warning notification."""
        try:
            message = f"Deadline Warning: '{event.title}' is due in {hours_remaining:.1f} hours. Current status: {event.status.value}"

            notification_id = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO notifications (
                        id, recipient_id, notification_type, title, message,
                        related_event_id, sent_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification_id, event.assigned_to, NotificationType.DEADLINE_WARNING.value,
                    f"Deadline Warning: {event.title}", message, event.id,
                    datetime.now(), json.dumps({})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error sending deadline warning: {e}")

class CalendarIntegration:
    """Integration with external calendar systems."""

    def __init__(self, calendar_manager: ContentCalendarManager):
        self.calendar_manager = calendar_manager

    def export_to_icalendar(self, start_date: datetime, end_date: datetime) -> str:
        """Export calendar events to iCalendar format."""
        try:
            cal = Calendar()
            cal.add('prodid', '-//Content Calendar//mxm.dk//')
            cal.add('version', '2.0')

            events = self.calendar_manager.get_calendar_events(start_date, end_date)

            for event in events:
                ical_event = Event()
                ical_event.add('uid', event.id)
                ical_event.add('summary', event.title)
                ical_event.add('description', event.description)
                ical_event.add('dtstart', event.start_date)
                ical_event.add('dtend', event.end_date)
                ical_event.add('status', event.status.value.upper())
                ical_event.add('priority', self._map_priority_to_ical(event.priority))

                # Add custom properties
                ical_event.add('x-content-type', event.content_type.value)
                ical_event.add('x-assigned-to', event.assigned_to)
                ical_event.add('x-channels', ','.join([c.value for c in event.channels]))

                cal.add_component(ical_event)

            return cal.to_ical().decode('utf-8')

        except Exception as e:
            logger.error(f"Error exporting to iCalendar: {e}")
            return ""

    def _map_priority_to_ical(self, priority: Priority) -> int:
        """Map internal priority to iCalendar priority scale (1-9)."""
        mapping = {
            Priority.LOW: 9,
            Priority.MEDIUM: 5,
            Priority.HIGH: 3,
            Priority.URGENT: 1
        }
        return mapping.get(priority, 5)

    def sync_with_google_calendar(self, calendar_id: str, credentials_path: str):
        """Sync with Google Calendar (placeholder for actual implementation)."""
        # In a real implementation, this would use the Google Calendar API
        logger.info(f"Syncing with Google Calendar: {calendar_id}")

        # Placeholder for Google Calendar integration
        # This would require:
        # 1. Google Calendar API credentials
        # 2. OAuth authentication
        # 3. Event creation/update/deletion logic

        return {"status": "success", "synced_events": 0}

    def sync_with_outlook(self, calendar_id: str, credentials: Dict[str, str]):
        """Sync with Outlook Calendar (placeholder for actual implementation)."""
        # In a real implementation, this would use the Microsoft Graph API
        logger.info(f"Syncing with Outlook Calendar: {calendar_id}")

        # Placeholder for Outlook Calendar integration
        return {"status": "success", "synced_events": 0}

class ContentCalendarSystem:
    """Main content calendar system."""

    def __init__(self, db_path: str = "content_calendar.db"):
        self.calendar_manager = ContentCalendarManager(db_path)
        self.automation_engine = AutomationEngine(self.calendar_manager)
        self.calendar_integration = CalendarIntegration(self.calendar_manager)

    def start(self):
        """Start the content calendar system."""
        self.automation_engine.start_automation()
        logger.info("Content Calendar System started")

    def stop(self):
        """Stop the content calendar system."""
        self.automation_engine.stop_automation()
        logger.info("Content Calendar System stopped")

    def create_weekly_blog_schedule(self, assigned_to: str, start_date: date, weeks: int = 4):
        """Helper method to create a weekly blog posting schedule."""
        events_created = []

        for week in range(weeks):
            # Schedule blog post for every Tuesday
            blog_date = start_date + timedelta(weeks=week, days=1)  # Tuesday
            blog_datetime = datetime.combine(blog_date, datetime.min.time().replace(hour=10))

            event_id = self.calendar_manager.create_calendar_event(
                title=f"Weekly Blog Post - Week {week + 1}",
                content_type=ContentType.BLOG_POST,
                description=f"Weekly blog post for week {week + 1}",
                start_date=blog_datetime,
                end_date=blog_datetime + timedelta(hours=2),
                assigned_to=assigned_to,
                channels=[PublishingChannel.WEBSITE, PublishingChannel.LINKEDIN],
                priority=Priority.MEDIUM
            )
            events_created.append(event_id)

        return events_created

    def create_social_media_campaign(self, campaign_name: str, start_date: date,
                                   end_date: date, assigned_to: str, daily_posts: int = 2):
        """Helper method to create a social media campaign schedule."""
        campaign_id = self.calendar_manager.create_campaign(
            name=campaign_name,
            description=f"Social media campaign: {campaign_name}",
            start_date=start_date,
            end_date=end_date,
            goals=["Increase engagement", "Drive traffic", "Build brand awareness"],
            target_audience="General audience",
            budget=1000.0,
            created_by=assigned_to
        )

        events_created = []
        current_date = start_date

        while current_date <= end_date:
            for post_num in range(daily_posts):
                post_time = datetime.combine(current_date, datetime.min.time().replace(
                    hour=9 + (post_num * 4)  # 9 AM, 1 PM, etc.
                ))

                event_id = self.calendar_manager.create_calendar_event(
                    title=f"{campaign_name} - Post {post_num + 1}",
                    content_type=ContentType.SOCIAL_MEDIA,
                    description=f"Social media post for {campaign_name}",
                    start_date=post_time,
                    end_date=post_time + timedelta(minutes=30),
                    assigned_to=assigned_to,
                    channels=[PublishingChannel.FACEBOOK, PublishingChannel.TWITTER, PublishingChannel.INSTAGRAM],
                    priority=Priority.MEDIUM,
                    campaign_id=campaign_id
                )
                events_created.append(event_id)

            current_date += timedelta(days=1)

        return {"campaign_id": campaign_id, "events": events_created}

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for the content calendar."""
        now = datetime.now()

        # Get upcoming events (next 7 days)
        upcoming_events = self.calendar_manager.get_calendar_events(
            start_date=now,
            end_date=now + timedelta(days=7)
        )

        # Get overdue events
        overdue_events = self.calendar_manager.get_calendar_events(
            end_date=now,
            status=ContentStatus.DRAFT
        )

        # Get events by status
        all_events = self.calendar_manager.get_calendar_events(
            start_date=now - timedelta(days=30),
            end_date=now + timedelta(days=30)
        )

        status_counts = {}
        for status in ContentStatus:
            status_counts[status.value] = len([e for e in all_events if e.status == status])

        # Get events by content type
        type_counts = {}
        for content_type in ContentType:
            type_counts[content_type.value] = len([e for e in all_events if e.content_type == content_type])

        return {
            "upcoming_events": len(upcoming_events),
            "overdue_events": len(overdue_events),
            "total_events_month": len(all_events),
            "status_breakdown": status_counts,
            "content_type_breakdown": type_counts,
            "recent_events": [
                {
                    "id": e.id,
                    "title": e.title,
                    "status": e.status.value,
                    "start_date": e.start_date.isoformat(),
                    "assigned_to": e.assigned_to
                } for e in upcoming_events[:10]
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the content calendar system
    calendar_system = ContentCalendarSystem()
    calendar_system.start()

    try:
        # Create a sample team member
        team_member = "john.doe@example.com"

        # Create a weekly blog schedule
        start_date = date.today() + timedelta(days=1)
        blog_events = calendar_system.create_weekly_blog_schedule(
            assigned_to=team_member,
            start_date=start_date,
            weeks=4
        )
        print(f"Created {len(blog_events)} blog post events")

        # Create a social media campaign
        campaign_data = calendar_system.create_social_media_campaign(
            campaign_name="Product Launch Campaign",
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            assigned_to=team_member,
            daily_posts=2
        )
        print(f"Created social media campaign with {len(campaign_data['events'])} events")

        # Get monthly calendar view
        today = date.today()
        monthly_calendar = calendar_system.calendar_manager.get_monthly_calendar(
            today.year, today.month
        )
        print(f"Monthly calendar has {monthly_calendar['total_events']} events")

        # Get dashboard data
        dashboard = calendar_system.get_dashboard_data()
        print(f"Dashboard: {dashboard['upcoming_events']} upcoming events, "
              f"{dashboard['overdue_events']} overdue events")

        # Export to iCalendar format
        ical_data = calendar_system.calendar_integration.export_to_icalendar(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30)
        )
        print(f"iCalendar export: {len(ical_data)} characters")

        # Create automation rule
        rule_id = calendar_system.automation_engine.create_publishing_rule(
            name="Daily Social Media Posts",
            content_type=ContentType.SOCIAL_MEDIA,
            channels=[PublishingChannel.FACEBOOK, PublishingChannel.TWITTER],
            schedule_pattern={"type": "daily", "time": "09:00"},
            auto_generate=True,
            approval_required=False
        )
        print(f"Created automation rule: {rule_id}")

        print("Content Calendar Integration System demonstration completed!")

    finally:
        calendar_system.stop()
