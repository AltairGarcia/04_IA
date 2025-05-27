"""
Content Templates & Automation System
====================================

A comprehensive system for managing content templates, automating workflows,
and scheduling content creation tasks.

Features:
- Pre-built templates for different content types
- Template customization and versioning
- Automated workflow scheduling
- Integration with existing content creation system
- Template analytics and performance tracking
"""

import json
import sqlite3
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import threading
import uuid
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content types supported by the template system."""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL_NEWSLETTER = "email_newsletter"
    PRODUCT_DESCRIPTION = "product_description"
    LANDING_PAGE = "landing_page"
    VIDEO_SCRIPT = "video_script"
    PODCAST_SCRIPT = "podcast_script"
    PRESS_RELEASE = "press_release"
    CASE_STUDY = "case_study"
    WHITE_PAPER = "white_paper"

class TemplateCategory(Enum):
    """Template categories for organization."""
    MARKETING = "marketing"
    EDUCATION = "education"
    ECOMMERCE = "ecommerce"
    TECHNOLOGY = "technology"
    HEALTH = "health"
    FINANCE = "finance"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    BUSINESS = "business"
    PERSONAL = "personal"

class AutomationTrigger(Enum):
    """Automation trigger types."""
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"

@dataclass
class ContentTemplate:
    """Content template data structure."""
    id: str
    name: str
    content_type: ContentType
    category: TemplateCategory
    title_template: str
    content_template: str
    description: str
    variables: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str
    tags: List[str]
    usage_count: int = 0
    rating: float = 0.0
    is_active: bool = True

@dataclass
class AutomationRule:
    """Automation rule data structure."""
    id: str
    name: str
    template_id: str
    trigger_type: AutomationTrigger
    trigger_config: Dict[str, Any]
    variables: Dict[str, Any]
    schedule_config: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    run_count: int = 0

@dataclass
class ScheduledTask:
    """Scheduled task data structure."""
    id: str
    rule_id: str
    template_id: str
    variables: Dict[str, Any]
    scheduled_time: datetime
    status: str
    result: Optional[Dict[str, Any]]
    created_at: datetime
    executed_at: Optional[datetime]

class TemplateManager:
    """Manages content templates and their operations."""

    def __init__(self, db_path: str = "content_templates.db"):
        self.db_path = db_path
        self.templates: Dict[str, ContentTemplate] = {}
        self.init_database()
        self.load_default_templates()

    def init_database(self):
        """Initialize the database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Templates table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS templates (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        title_template TEXT NOT NULL,
                        content_template TEXT NOT NULL,
                        description TEXT,
                        variables TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        version TEXT,
                        tags TEXT,
                        usage_count INTEGER DEFAULT 0,
                        rating REAL DEFAULT 0.0,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                # Template versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS template_versions (
                        id TEXT PRIMARY KEY,
                        template_id TEXT,
                        version TEXT,
                        title_template TEXT,
                        content_template TEXT,
                        variables TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP,
                        FOREIGN KEY (template_id) REFERENCES templates (id)
                    )
                """)

                # Template usage analytics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS template_usage (
                        id TEXT PRIMARY KEY,
                        template_id TEXT,
                        used_at TIMESTAMP,
                        variables_used TEXT,
                        success BOOLEAN,
                        performance_metrics TEXT,
                        FOREIGN KEY (template_id) REFERENCES templates (id)
                    )
                """)

                conn.commit()
                logger.info("Template database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing template database: {e}")
            raise

    def load_default_templates(self):
        """Load default templates into the system."""
        default_templates = [
            {
                "name": "Blog Post - How-to Guide",
                "content_type": ContentType.BLOG_POST,
                "category": TemplateCategory.EDUCATION,
                "title_template": "How to {action} in {timeframe}: A Complete Guide",
                "content_template": """
# How to {action} in {timeframe}: A Complete Guide

## Introduction
{introduction}

## Why {action} Matters
{importance}

## Step-by-Step Guide

### Step 1: {step1_title}
{step1_content}

### Step 2: {step2_title}
{step2_content}

### Step 3: {step3_title}
{step3_content}

## Common Mistakes to Avoid
{mistakes}

## Conclusion
{conclusion}

## Call to Action
{cta}
                """.strip(),
                "description": "Template for creating comprehensive how-to guides",
                "variables": ["action", "timeframe", "introduction", "importance", "step1_title", "step1_content", "step2_title", "step2_content", "step3_title", "step3_content", "mistakes", "conclusion", "cta"],
                "tags": ["how-to", "guide", "educational", "tutorial"]
            },
            {
                "name": "Social Media - Product Launch",
                "content_type": ContentType.SOCIAL_MEDIA,
                "category": TemplateCategory.MARKETING,
                "title_template": "🚀 Introducing {product_name} - {tagline}",
                "content_template": """
🚀 Exciting news! We're thrilled to introduce {product_name} - {tagline}

✨ Key Features:
{features}

🎯 Perfect for: {target_audience}

💰 Special Launch Offer: {offer}

👉 Learn more: {link}

#{hashtag1} #{hashtag2} #{hashtag3} #ProductLaunch #Innovation
                """.strip(),
                "description": "Template for product launch social media posts",
                "variables": ["product_name", "tagline", "features", "target_audience", "offer", "link", "hashtag1", "hashtag2", "hashtag3"],
                "tags": ["social-media", "product-launch", "marketing", "announcement"]
            },
            {
                "name": "Email Newsletter - Weekly Update",
                "content_type": ContentType.EMAIL_NEWSLETTER,
                "category": TemplateCategory.MARKETING,
                "title_template": "{company_name} Weekly Update - {week_highlight}",
                "content_template": """
Subject: {company_name} Weekly Update - {week_highlight}

Hi {subscriber_name},

Hope you're having a great week! Here's what's been happening at {company_name}:

## This Week's Highlights
{highlights}

## Featured Content
{featured_content}

## Upcoming Events
{upcoming_events}

## Community Spotlight
{community_spotlight}

## Quick Tips
{tips}

That's all for this week! Have a fantastic weekend.

Best regards,
{sender_name}
{company_name} Team

---
You're receiving this email because you subscribed to our newsletter.
[Unsubscribe] | [Update Preferences] | [Forward to a Friend]
                """.strip(),
                "description": "Template for weekly newsletter emails",
                "variables": ["company_name", "week_highlight", "subscriber_name", "highlights", "featured_content", "upcoming_events", "community_spotlight", "tips", "sender_name"],
                "tags": ["newsletter", "email", "weekly", "updates"]
            },
            {
                "name": "Product Description - E-commerce",
                "content_type": ContentType.PRODUCT_DESCRIPTION,
                "category": TemplateCategory.ECOMMERCE,
                "title_template": "{product_name} - {key_benefit}",
                "content_template": """
## {product_name}

{product_description}

### Key Features:
{features}

### Benefits:
{benefits}

### Specifications:
{specifications}

### What's Included:
{included_items}

### Perfect For:
{ideal_for}

### Guarantee:
{guarantee}

**Price: {price}**
**Availability: {availability}**

[Add to Cart] [Add to Wishlist] [Share]
                """.strip(),
                "description": "Template for e-commerce product descriptions",
                "variables": ["product_name", "key_benefit", "product_description", "features", "benefits", "specifications", "included_items", "ideal_for", "guarantee", "price", "availability"],
                "tags": ["product", "ecommerce", "description", "sales"]
            },
            {
                "name": "Landing Page - Service Offering",
                "content_type": ContentType.LANDING_PAGE,
                "category": TemplateCategory.MARKETING,
                "title_template": "{service_name} - {value_proposition}",
                "content_template": """
# {service_name}
## {value_proposition}

### The Problem
{problem_description}

### Our Solution
{solution_description}

### How It Works
{process_steps}

### Benefits
{benefits_list}

### Success Stories
{testimonials}

### Pricing
{pricing_info}

### Get Started Today
{cta_section}

### Frequently Asked Questions
{faq_section}

### Contact Us
{contact_info}
                """.strip(),
                "description": "Template for service offering landing pages",
                "variables": ["service_name", "value_proposition", "problem_description", "solution_description", "process_steps", "benefits_list", "testimonials", "pricing_info", "cta_section", "faq_section", "contact_info"],
                "tags": ["landing-page", "service", "marketing", "conversion"]
            }
        ]

        for template_data in default_templates:
            if not self.template_exists(template_data["name"]):
                template = ContentTemplate(
                    id=str(uuid.uuid4()),
                    name=template_data["name"],
                    content_type=template_data["content_type"],
                    category=template_data["category"],
                    title_template=template_data["title_template"],
                    content_template=template_data["content_template"],
                    description=template_data["description"],
                    variables=template_data["variables"],
                    metadata={},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version="1.0",
                    tags=template_data["tags"]
                )
                self.save_template(template)

    def template_exists(self, name: str) -> bool:
        """Check if a template with the given name exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM templates WHERE name = ?", (name,))
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking template existence: {e}")
            return False

    def save_template(self, template: ContentTemplate):
        """Save a template to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO templates (
                        id, name, content_type, category, title_template, content_template,
                        description, variables, metadata, created_at, updated_at, version,
                        tags, usage_count, rating, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    template.id, template.name, template.content_type.value,
                    template.category.value, template.title_template, template.content_template,
                    template.description, json.dumps(template.variables), json.dumps(template.metadata),
                    template.created_at, template.updated_at, template.version,
                    json.dumps(template.tags), template.usage_count, template.rating, template.is_active
                ))

                conn.commit()
                self.templates[template.id] = template
                logger.info(f"Template '{template.name}' saved successfully")

        except Exception as e:
            logger.error(f"Error saving template: {e}")
            raise

    def get_template(self, template_id: str) -> Optional[ContentTemplate]:
        """Get a template by ID."""
        if template_id in self.templates:
            return self.templates[template_id]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM templates WHERE id = ?", (template_id,))
                row = cursor.fetchone()

                if row:
                    template = ContentTemplate(
                        id=row[0], name=row[1], content_type=ContentType(row[2]),
                        category=TemplateCategory(row[3]), title_template=row[4],
                        content_template=row[5], description=row[6],
                        variables=json.loads(row[7]), metadata=json.loads(row[8]),
                        created_at=datetime.fromisoformat(row[9]),
                        updated_at=datetime.fromisoformat(row[10]),
                        version=row[11], tags=json.loads(row[12]),
                        usage_count=row[13], rating=row[14], is_active=bool(row[15])
                    )
                    self.templates[template_id] = template
                    return template

        except Exception as e:
            logger.error(f"Error getting template: {e}")

        return None

    def list_templates(self, content_type: Optional[ContentType] = None,
                      category: Optional[TemplateCategory] = None,
                      tags: Optional[List[str]] = None) -> List[ContentTemplate]:
        """List templates with optional filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM templates WHERE is_active = 1"
                params = []

                if content_type:
                    query += " AND content_type = ?"
                    params.append(content_type.value)

                if category:
                    query += " AND category = ?"
                    params.append(category.value)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                templates = []
                for row in rows:
                    template = ContentTemplate(
                        id=row[0], name=row[1], content_type=ContentType(row[2]),
                        category=TemplateCategory(row[3]), title_template=row[4],
                        content_template=row[5], description=row[6],
                        variables=json.loads(row[7]), metadata=json.loads(row[8]),
                        created_at=datetime.fromisoformat(row[9]),
                        updated_at=datetime.fromisoformat(row[10]),
                        version=row[11], tags=json.loads(row[12]),
                        usage_count=row[13], rating=row[14], is_active=bool(row[15])
                    )

                    # Filter by tags if specified
                    if tags:
                        if any(tag in template.tags for tag in tags):
                            templates.append(template)
                    else:
                        templates.append(template)

                return templates

        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []

    def generate_content(self, template_id: str, variables: Dict[str, str]) -> Dict[str, str]:
        """Generate content using a template and variables."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")

        try:
            # Generate title
            title = template.title_template
            for var, value in variables.items():
                title = title.replace(f"{{{var}}}", str(value))

            # Generate content
            content = template.content_template
            for var, value in variables.items():
                content = content.replace(f"{{{var}}}", str(value))

            # Update usage count
            template.usage_count += 1
            template.updated_at = datetime.now()
            self.save_template(template)

            # Log usage
            self.log_template_usage(template_id, variables, True)

            return {
                "title": title,
                "content": content,
                "template_id": template_id,
                "template_name": template.name,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating content: {e}")
            self.log_template_usage(template_id, variables, False)
            raise

    def log_template_usage(self, template_id: str, variables: Dict[str, str], success: bool):
        """Log template usage for analytics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO template_usage (
                        id, template_id, used_at, variables_used, success, performance_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()), template_id, datetime.now(),
                    json.dumps(variables), success, json.dumps({})
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Error logging template usage: {e}")

    def get_template_analytics(self, template_id: str) -> Dict[str, Any]:
        """Get analytics for a specific template."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get template info
                template = self.get_template(template_id)
                if not template:
                    return {}

                # Get usage statistics
                cursor.execute("""
                    SELECT COUNT(*) as total_uses,
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_uses,
                           MIN(used_at) as first_used,
                           MAX(used_at) as last_used
                    FROM template_usage WHERE template_id = ?
                """, (template_id,))

                stats = cursor.fetchone()

                # Get recent usage
                cursor.execute("""
                    SELECT used_at, variables_used, success
                    FROM template_usage
                    WHERE template_id = ?
                    ORDER BY used_at DESC
                    LIMIT 10
                """, (template_id,))

                recent_usage = cursor.fetchall()

                return {
                    "template_name": template.name,
                    "total_uses": stats[0] or 0,
                    "successful_uses": stats[1] or 0,
                    "success_rate": (stats[1] / stats[0] * 100) if stats[0] > 0 else 0,
                    "first_used": stats[2],
                    "last_used": stats[3],
                    "rating": template.rating,
                    "recent_usage": [
                        {
                            "used_at": usage[0],
                            "variables": json.loads(usage[1]),
                            "success": bool(usage[2])
                        } for usage in recent_usage
                    ]
                }

        except Exception as e:
            logger.error(f"Error getting template analytics: {e}")
            return {}


class AutomationEngine:
    """Handles automation rules and scheduled content generation."""

    def __init__(self, template_manager: TemplateManager, db_path: str = "content_automation.db"):
        self.template_manager = template_manager
        self.db_path = db_path
        self.automation_rules: Dict[str, AutomationRule] = {}
        self.scheduler_thread = None
        self.running = False
        self.init_database()

    def init_database(self):
        """Initialize automation database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Automation rules table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS automation_rules (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        template_id TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        trigger_config TEXT,
                        variables TEXT,
                        schedule_config TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP,
                        last_run TIMESTAMP,
                        next_run TIMESTAMP,
                        run_count INTEGER DEFAULT 0
                    )
                """)

                # Scheduled tasks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_tasks (
                        id TEXT PRIMARY KEY,
                        rule_id TEXT,
                        template_id TEXT,
                        variables TEXT,
                        scheduled_time TIMESTAMP,
                        status TEXT,
                        result TEXT,
                        created_at TIMESTAMP,
                        executed_at TIMESTAMP,
                        FOREIGN KEY (rule_id) REFERENCES automation_rules (id)
                    )
                """)

                conn.commit()
                logger.info("Automation database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing automation database: {e}")
            raise

    def create_automation_rule(self, name: str, template_id: str,
                             trigger_type: AutomationTrigger,
                             trigger_config: Dict[str, Any],
                             variables: Dict[str, Any],
                             schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new automation rule."""
        rule_id = str(uuid.uuid4())

        rule = AutomationRule(
            id=rule_id,
            name=name,
            template_id=template_id,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            variables=variables,
            schedule_config=schedule_config,
            is_active=True,
            created_at=datetime.now(),
            last_run=None,
            next_run=self._calculate_next_run(schedule_config)
        )

        self.save_automation_rule(rule)
        return rule_id

    def save_automation_rule(self, rule: AutomationRule):
        """Save automation rule to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO automation_rules (
                        id, name, template_id, trigger_type, trigger_config,
                        variables, schedule_config, is_active, created_at,
                        last_run, next_run, run_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.id, rule.name, rule.template_id, rule.trigger_type.value,
                    json.dumps(rule.trigger_config), json.dumps(rule.variables),
                    json.dumps(rule.schedule_config), rule.is_active,
                    rule.created_at, rule.last_run, rule.next_run, rule.run_count
                ))

                conn.commit()
                self.automation_rules[rule.id] = rule
                logger.info(f"Automation rule '{rule.name}' saved successfully")

        except Exception as e:
            logger.error(f"Error saving automation rule: {e}")
            raise

    def _calculate_next_run(self, schedule_config: Optional[Dict[str, Any]]) -> Optional[datetime]:
        """Calculate next run time based on schedule configuration."""
        if not schedule_config:
            return None

        schedule_type = schedule_config.get("type", "")
        now = datetime.now()

        if schedule_type == "daily":
            time_str = schedule_config.get("time", "09:00")
            hour, minute = map(int, time_str.split(":"))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        elif schedule_type == "weekly":
            day_of_week = schedule_config.get("day", 0)  # 0 = Monday
            time_str = schedule_config.get("time", "09:00")
            hour, minute = map(int, time_str.split(":"))

            days_ahead = day_of_week - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7

            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return next_run

        elif schedule_type == "monthly":
            day = schedule_config.get("day", 1)
            time_str = schedule_config.get("time", "09:00")
            hour, minute = map(int, time_str.split(":"))

            next_month = now.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)

            try:
                next_run = next_month.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            except ValueError:
                # Handle case where day doesn't exist in month (e.g., Feb 30)
                next_run = next_month.replace(day=28, hour=hour, minute=minute, second=0, microsecond=0)

            return next_run

        return None

    def start_scheduler(self):
        """Start the automation scheduler."""
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Automation scheduler started")

    def stop_scheduler(self):
        """Stop the automation scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Automation scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                self._check_and_execute_scheduled_tasks()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)

    def _check_and_execute_scheduled_tasks(self):
        """Check for and execute scheduled tasks."""
        now = datetime.now()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM automation_rules
                    WHERE is_active = 1 AND next_run <= ?
                """, (now,))

                rules = cursor.fetchall()

                for rule_data in rules:
                    rule = AutomationRule(
                        id=rule_data[0], name=rule_data[1], template_id=rule_data[2],
                        trigger_type=AutomationTrigger(rule_data[3]),
                        trigger_config=json.loads(rule_data[4]),
                        variables=json.loads(rule_data[5]),
                        schedule_config=json.loads(rule_data[6]) if rule_data[6] else None,
                        is_active=bool(rule_data[7]),
                        created_at=datetime.fromisoformat(rule_data[8]),
                        last_run=datetime.fromisoformat(rule_data[9]) if rule_data[9] else None,
                        next_run=datetime.fromisoformat(rule_data[10]) if rule_data[10] else None,
                        run_count=rule_data[11]
                    )

                    self._execute_automation_rule(rule)

        except Exception as e:
            logger.error(f"Error checking scheduled tasks: {e}")

    def _execute_automation_rule(self, rule: AutomationRule):
        """Execute an automation rule."""
        try:
            logger.info(f"Executing automation rule: {rule.name}")

            # Generate content using the template
            result = self.template_manager.generate_content(rule.template_id, rule.variables)

            # Create scheduled task record
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                rule_id=rule.id,
                template_id=rule.template_id,
                variables=rule.variables,
                scheduled_time=rule.next_run,
                status="completed",
                result=result,
                created_at=datetime.now(),
                executed_at=datetime.now()
            )

            self._save_scheduled_task(task)

            # Update rule
            rule.last_run = datetime.now()
            rule.next_run = self._calculate_next_run(rule.schedule_config)
            rule.run_count += 1
            self.save_automation_rule(rule)

            logger.info(f"Automation rule '{rule.name}' executed successfully")

        except Exception as e:
            logger.error(f"Error executing automation rule '{rule.name}': {e}")

            # Create failed task record
            task = ScheduledTask(
                id=str(uuid.uuid4()),
                rule_id=rule.id,
                template_id=rule.template_id,
                variables=rule.variables,
                scheduled_time=rule.next_run,
                status="failed",
                result={"error": str(e)},
                created_at=datetime.now(),
                executed_at=datetime.now()
            )

            self._save_scheduled_task(task)

    def _save_scheduled_task(self, task: ScheduledTask):
        """Save scheduled task to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO scheduled_tasks (
                        id, rule_id, template_id, variables, scheduled_time,
                        status, result, created_at, executed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id, task.rule_id, task.template_id,
                    json.dumps(task.variables), task.scheduled_time,
                    task.status, json.dumps(task.result),
                    task.created_at, task.executed_at
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error saving scheduled task: {e}")

    def get_automation_rules(self) -> List[AutomationRule]:
        """Get all automation rules."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM automation_rules")
                rows = cursor.fetchall()

                rules = []
                for row in rows:
                    rule = AutomationRule(
                        id=row[0], name=row[1], template_id=row[2],
                        trigger_type=AutomationTrigger(row[3]),
                        trigger_config=json.loads(row[4]),
                        variables=json.loads(row[5]),
                        schedule_config=json.loads(row[6]) if row[6] else None,
                        is_active=bool(row[7]),
                        created_at=datetime.fromisoformat(row[8]),
                        last_run=datetime.fromisoformat(row[9]) if row[9] else None,
                        next_run=datetime.fromisoformat(row[10]) if row[10] else None,
                        run_count=row[11]
                    )
                    rules.append(rule)

                return rules

        except Exception as e:
            logger.error(f"Error getting automation rules: {e}")
            return []

    def get_scheduled_tasks(self, limit: int = 50) -> List[ScheduledTask]:
        """Get recent scheduled tasks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM scheduled_tasks
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

                rows = cursor.fetchall()

                tasks = []
                for row in rows:
                    task = ScheduledTask(
                        id=row[0], rule_id=row[1], template_id=row[2],
                        variables=json.loads(row[3]),
                        scheduled_time=datetime.fromisoformat(row[4]),
                        status=row[5],
                        result=json.loads(row[6]) if row[6] else None,
                        created_at=datetime.fromisoformat(row[7]),
                        executed_at=datetime.fromisoformat(row[8]) if row[8] else None
                    )
                    tasks.append(task)

                return tasks

        except Exception as e:
            logger.error(f"Error getting scheduled tasks: {e}")
            return []


class ContentTemplateSystem:
    """Main content template and automation system."""

    def __init__(self, template_db_path: str = "content_templates.db",
                 automation_db_path: str = "content_automation.db"):
        self.template_manager = TemplateManager(template_db_path)
        self.automation_engine = AutomationEngine(self.template_manager, automation_db_path)

    def start(self):
        """Start the template system."""
        self.automation_engine.start_scheduler()
        logger.info("Content Template System started")

    def stop(self):
        """Stop the template system."""
        self.automation_engine.stop_scheduler()
        logger.info("Content Template System stopped")

    def create_daily_blog_automation(self, template_id: str, variables: Dict[str, Any], time: str = "09:00"):
        """Helper method to create daily blog post automation."""
        return self.automation_engine.create_automation_rule(
            name=f"Daily Blog Post - {datetime.now().strftime('%Y-%m-%d')}",
            template_id=template_id,
            trigger_type=AutomationTrigger.SCHEDULED,
            trigger_config={},
            variables=variables,
            schedule_config={
                "type": "daily",
                "time": time
            }
        )

    def create_weekly_newsletter_automation(self, template_id: str, variables: Dict[str, Any],
                                          day: int = 0, time: str = "10:00"):
        """Helper method to create weekly newsletter automation."""
        return self.automation_engine.create_automation_rule(
            name=f"Weekly Newsletter - {datetime.now().strftime('%Y-%m-%d')}",
            template_id=template_id,
            trigger_type=AutomationTrigger.SCHEDULED,
            trigger_config={},
            variables=variables,
            schedule_config={
                "type": "weekly",
                "day": day,  # 0 = Monday
                "time": time
            }
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        templates = self.template_manager.list_templates()
        rules = self.automation_engine.get_automation_rules()
        recent_tasks = self.automation_engine.get_scheduled_tasks(10)

        return {
            "templates": {
                "total": len(templates),
                "active": len([t for t in templates if t.is_active]),
                "by_type": {ct.value: len([t for t in templates if t.content_type == ct])
                           for ct in ContentType},
                "by_category": {cc.value: len([t for t in templates if t.category == cc])
                               for cc in TemplateCategory}
            },
            "automation": {
                "total_rules": len(rules),
                "active_rules": len([r for r in rules if r.is_active]),
                "recent_executions": len(recent_tasks),
                "successful_executions": len([t for t in recent_tasks if t.status == "completed"]),
                "scheduler_running": self.automation_engine.running
            },
            "system": {
                "startup_time": datetime.now().isoformat(),
                "database_status": "connected"
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    system = ContentTemplateSystem()
    system.start()

    try:
        # Get available templates
        templates = system.template_manager.list_templates()
        print(f"Available templates: {len(templates)}")

        for template in templates[:3]:  # Show first 3 templates
            print(f"- {template.name} ({template.content_type.value})")

        # Create automation example
        if templates:
            blog_template = None
            for template in templates:
                if template.content_type == ContentType.BLOG_POST:
                    blog_template = template
                    break

            if blog_template:
                # Create daily automation
                rule_id = system.create_daily_blog_automation(
                    template_id=blog_template.id,
                    variables={
                        "action": "learn Python",
                        "timeframe": "30 days",
                        "introduction": "Python is a powerful programming language...",
                        "importance": "Learning Python opens up many opportunities...",
                        "step1_title": "Install Python",
                        "step1_content": "Download Python from python.org...",
                        "step2_title": "Learn the Basics",
                        "step2_content": "Start with variables and data types...",
                        "step3_title": "Practice Projects",
                        "step3_content": "Build small projects to apply your knowledge...",
                        "mistakes": "Common mistakes include...",
                        "conclusion": "With consistent practice...",
                        "cta": "Start your Python journey today!"
                    }
                )
                print(f"Created automation rule: {rule_id}")

        # Get system status
        status = system.get_system_status()
        print(f"System status: {json.dumps(status, indent=2)}")

    finally:
        system.stop()
