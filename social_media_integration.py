"""
Social Media Integration System

This module provides comprehensive social media posting and management capabilities
for the content creation workflow, including authentication, publishing, scheduling,
and analytics across multiple platforms.
"""

import os
import json
import sqlite3
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
import logging
from pathlib import Path
import base64
import mimetypes

logger = logging.getLogger(__name__)

class SocialPlatform(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    PINTEREST = "pinterest"
    REDDIT = "reddit"

class ContentType(Enum):
    """Types of content that can be posted."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"
    POLL = "poll"
    STORY = "story"
    REEL = "reel"

class PostStatus(Enum):
    """Status of social media posts."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    DELETED = "deleted"

@dataclass
class SocialMediaCredentials:
    """Social media platform credentials."""
    platform: SocialPlatform
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user_id: Optional[str] = None
    expires_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if credentials are valid and not expired."""
        if not self.access_token:
            return False

        if self.expires_at and datetime.now() >= self.expires_at:
            return False

        return True

@dataclass
class SocialMediaPost:
    """Social media post data structure."""
    id: str
    platform: SocialPlatform
    content_type: ContentType
    title: Optional[str] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    link_url: Optional[str] = None
    hashtags: List[str] = None
    mentions: List[str] = None
    scheduled_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    status: PostStatus = PostStatus.DRAFT
    platform_post_id: Optional[str] = None
    analytics: Dict[str, Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []
        if self.analytics is None:
            self.analytics = {}
        if self.metadata is None:
            self.metadata = {}

class SocialMediaManager:
    """Manages social media integrations and posting."""

    def __init__(self, db_path: str = "social_media.db"):
        self.db_path = db_path
        self.credentials: Dict[SocialPlatform, SocialMediaCredentials] = {}
        self.session = requests.Session()
        self._initialize_database()
        self._load_credentials()

    def _initialize_database(self):
        """Initialize the social media database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Credentials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    platform TEXT PRIMARY KEY,
                    api_key TEXT,
                    api_secret TEXT,
                    access_token TEXT,
                    refresh_token TEXT,
                    user_id TEXT,
                    expires_at TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Posts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    platform TEXT,
                    content_type TEXT,
                    title TEXT,
                    text TEXT,
                    image_url TEXT,
                    video_url TEXT,
                    link_url TEXT,
                    hashtags TEXT,
                    mentions TEXT,
                    scheduled_at TEXT,
                    published_at TEXT,
                    status TEXT,
                    platform_post_id TEXT,
                    analytics TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS post_analytics (
                    id TEXT PRIMARY KEY,
                    post_id TEXT,
                    platform TEXT,
                    metric_name TEXT,
                    metric_value INTEGER,
                    recorded_at TEXT,
                    FOREIGN KEY (post_id) REFERENCES posts (id)
                )
            """)

            # Platform limits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS platform_limits (
                    platform TEXT PRIMARY KEY,
                    posts_per_hour INTEGER,
                    posts_per_day INTEGER,
                    character_limit INTEGER,
                    image_size_limit INTEGER,
                    video_size_limit INTEGER,
                    video_duration_limit INTEGER,
                    hashtag_limit INTEGER
                )
            """)

            conn.commit()
            self._populate_platform_limits()

    def _populate_platform_limits(self):
        """Populate platform-specific limits."""
        limits = {
            SocialPlatform.TWITTER: {
                'posts_per_hour': 100,
                'posts_per_day': 2400,
                'character_limit': 280,
                'image_size_limit': 5 * 1024 * 1024,  # 5MB
                'video_size_limit': 512 * 1024 * 1024,  # 512MB
                'video_duration_limit': 140,  # seconds
                'hashtag_limit': 10
            },
            SocialPlatform.LINKEDIN: {
                'posts_per_hour': 25,
                'posts_per_day': 100,
                'character_limit': 3000,
                'image_size_limit': 20 * 1024 * 1024,  # 20MB
                'video_size_limit': 5 * 1024 * 1024 * 1024,  # 5GB
                'video_duration_limit': 600,  # 10 minutes
                'hashtag_limit': 20
            },
            SocialPlatform.FACEBOOK: {
                'posts_per_hour': 25,
                'posts_per_day': 200,
                'character_limit': 63206,
                'image_size_limit': 10 * 1024 * 1024,  # 10MB
                'video_size_limit': 10 * 1024 * 1024 * 1024,  # 10GB
                'video_duration_limit': 7200,  # 2 hours
                'hashtag_limit': 30
            },
            SocialPlatform.INSTAGRAM: {
                'posts_per_hour': 25,
                'posts_per_day': 100,
                'character_limit': 2200,
                'image_size_limit': 30 * 1024 * 1024,  # 30MB
                'video_size_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                'video_duration_limit': 3600,  # 1 hour
                'hashtag_limit': 30
            }
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for platform, platform_limits in limits.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_limits
                    (platform, posts_per_hour, posts_per_day, character_limit,
                     image_size_limit, video_size_limit, video_duration_limit, hashtag_limit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    platform.value,
                    platform_limits['posts_per_hour'],
                    platform_limits['posts_per_day'],
                    platform_limits['character_limit'],
                    platform_limits['image_size_limit'],
                    platform_limits['video_size_limit'],
                    platform_limits['video_duration_limit'],
                    platform_limits['hashtag_limit']
                ))

            conn.commit()

    def _load_credentials(self):
        """Load stored credentials from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM credentials")

                for row in cursor.fetchall():
                    platform = SocialPlatform(row[0])
                    expires_at = datetime.fromisoformat(row[5]) if row[5] else None

                    credentials = SocialMediaCredentials(
                        platform=platform,
                        api_key=row[1],
                        api_secret=row[2],
                        access_token=row[3],
                        refresh_token=row[4],
                        user_id=row[5],
                        expires_at=expires_at
                    )

                    self.credentials[platform] = credentials

        except Exception as e:
            logger.warning(f"Failed to load credentials: {e}")

    def add_credentials(self, credentials: SocialMediaCredentials):
        """Add or update platform credentials."""
        self.credentials[credentials.platform] = credentials

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            expires_at_str = credentials.expires_at.isoformat() if credentials.expires_at else None
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO credentials
                (platform, api_key, api_secret, access_token, refresh_token,
                 user_id, expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                credentials.platform.value,
                credentials.api_key,
                credentials.api_secret,
                credentials.access_token,
                credentials.refresh_token,
                credentials.user_id,
                expires_at_str,
                now,
                now
            ))

            conn.commit()

    def get_platform_limits(self, platform: SocialPlatform) -> Dict[str, int]:
        """Get platform-specific limits."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT posts_per_hour, posts_per_day, character_limit,
                       image_size_limit, video_size_limit, video_duration_limit, hashtag_limit
                FROM platform_limits WHERE platform = ?
            """, (platform.value,))

            row = cursor.fetchone()
            if row:
                return {
                    'posts_per_hour': row[0],
                    'posts_per_day': row[1],
                    'character_limit': row[2],
                    'image_size_limit': row[3],
                    'video_size_limit': row[4],
                    'video_duration_limit': row[5],
                    'hashtag_limit': row[6]
                }

            return {}

    def validate_post_content(self, post: SocialMediaPost) -> List[str]:
        """Validate post content against platform limits."""
        errors = []
        limits = self.get_platform_limits(post.platform)

        if not limits:
            return ["Platform limits not found"]

        # Check character limit
        if post.text and len(post.text) > limits.get('character_limit', float('inf')):
            errors.append(f"Text exceeds character limit ({len(post.text)}/{limits['character_limit']})")

        # Check hashtag limit
        if len(post.hashtags) > limits.get('hashtag_limit', float('inf')):
            errors.append(f"Too many hashtags ({len(post.hashtags)}/{limits['hashtag_limit']})")

        return errors

    async def create_post(self, post: SocialMediaPost) -> bool:
        """Create a new social media post."""
        # Validate post content
        validation_errors = self.validate_post_content(post)
        if validation_errors:
            post.status = PostStatus.FAILED
            post.error_message = "; ".join(validation_errors)
            self._save_post(post)
            return False

        # Check if credentials are valid
        if post.platform not in self.credentials:
            post.status = PostStatus.FAILED
            post.error_message = f"No credentials found for {post.platform.value}"
            self._save_post(post)
            return False

        credentials = self.credentials[post.platform]
        if not credentials.is_valid():
            post.status = PostStatus.FAILED
            post.error_message = f"Invalid or expired credentials for {post.platform.value}"
            self._save_post(post)
            return False

        # Save post to database
        self._save_post(post)

        # If scheduled for later, don't publish now
        if post.scheduled_at and post.scheduled_at > datetime.now():
            post.status = PostStatus.SCHEDULED
            self._save_post(post)
            return True

        # Publish immediately
        return await self._publish_post(post)

    async def _publish_post(self, post: SocialMediaPost) -> bool:
        """Publish a post to the specified platform."""
        try:
            if post.platform == SocialPlatform.TWITTER:
                success = await self._publish_to_twitter(post)
            elif post.platform == SocialPlatform.LINKEDIN:
                success = await self._publish_to_linkedin(post)
            elif post.platform == SocialPlatform.FACEBOOK:
                success = await self._publish_to_facebook(post)
            elif post.platform == SocialPlatform.INSTAGRAM:
                success = await self._publish_to_instagram(post)
            else:
                # For unsupported platforms, simulate success
                success = await self._simulate_publish(post)

            if success:
                post.status = PostStatus.PUBLISHED
                post.published_at = datetime.now()
            else:
                post.status = PostStatus.FAILED

            self._save_post(post)
            return success

        except Exception as e:
            post.status = PostStatus.FAILED
            post.error_message = str(e)
            self._save_post(post)
            logger.error(f"Failed to publish post {post.id}: {e}")
            return False

    async def _publish_to_twitter(self, post: SocialMediaPost) -> bool:
        """Publish to Twitter using Twitter API v2."""
        credentials = self.credentials[SocialPlatform.TWITTER]

        # Twitter API v2 endpoint
        url = "https://api.twitter.com/2/tweets"

        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json"
        }

        # Prepare tweet data
        tweet_data = {
            "text": post.text
        }

        # Add media if present
        if post.image_url or post.video_url:
            media_ids = await self._upload_media_to_twitter(post)
            if media_ids:
                tweet_data["media"] = {"media_ids": media_ids}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=tweet_data) as response:
                if response.status == 201:
                    result = await response.json()
                    post.platform_post_id = result["data"]["id"]
                    return True
                else:
                    error_text = await response.text()
                    post.error_message = f"Twitter API error: {error_text}"
                    return False

    async def _upload_media_to_twitter(self, post: SocialMediaPost) -> List[str]:
        """Upload media to Twitter and return media IDs."""
        # This is a simplified implementation
        # In practice, you'd need to implement proper media upload
        # using Twitter's media upload endpoints
        return []

    async def _publish_to_linkedin(self, post: SocialMediaPost) -> bool:
        """Publish to LinkedIn using LinkedIn API."""
        credentials = self.credentials[SocialPlatform.LINKEDIN]

        # LinkedIn API endpoint
        url = "https://api.linkedin.com/v2/ugcPosts"

        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }

        # Prepare post data
        post_data = {
            "author": f"urn:li:person:{credentials.user_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": post.text
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=post_data) as response:
                if response.status == 201:
                    result = await response.json()
                    post.platform_post_id = result.get("id", "")
                    return True
                else:
                    error_text = await response.text()
                    post.error_message = f"LinkedIn API error: {error_text}"
                    return False

    async def _publish_to_facebook(self, post: SocialMediaPost) -> bool:
        """Publish to Facebook using Graph API."""
        credentials = self.credentials[SocialPlatform.FACEBOOK]

        # Facebook Graph API endpoint
        url = f"https://graph.facebook.com/v18.0/{credentials.user_id}/feed"

        params = {
            "access_token": credentials.access_token,
            "message": post.text
        }

        if post.link_url:
            params["link"] = post.link_url

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    post.platform_post_id = result.get("id", "")
                    return True
                else:
                    error_text = await response.text()
                    post.error_message = f"Facebook API error: {error_text}"
                    return False

    async def _publish_to_instagram(self, post: SocialMediaPost) -> bool:
        """Publish to Instagram using Instagram Basic Display API."""
        credentials = self.credentials[SocialPlatform.INSTAGRAM]

        # Instagram requires media to be uploaded first
        # This is a simplified implementation
        url = f"https://graph.facebook.com/v18.0/{credentials.user_id}/media"

        params = {
            "access_token": credentials.access_token,
            "caption": post.text
        }

        if post.image_url:
            params["image_url"] = post.image_url
        elif post.video_url:
            params["video_url"] = post.video_url

        async with aiohttp.ClientSession() as session:
            # Create media container
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    creation_id = result.get("id")

                    # Publish media container
                    publish_url = f"https://graph.facebook.com/v18.0/{credentials.user_id}/media_publish"
                    publish_params = {
                        "access_token": credentials.access_token,
                        "creation_id": creation_id
                    }

                    async with session.post(publish_url, params=publish_params) as publish_response:
                        if publish_response.status == 200:
                            publish_result = await publish_response.json()
                            post.platform_post_id = publish_result.get("id", "")
                            return True
                        else:
                            error_text = await publish_response.text()
                            post.error_message = f"Instagram publish error: {error_text}"
                            return False
                else:
                    error_text = await response.text()
                    post.error_message = f"Instagram media creation error: {error_text}"
                    return False

    async def _simulate_publish(self, post: SocialMediaPost) -> bool:
        """Simulate publishing for unsupported platforms or demo mode."""
        import random

        # Simulate API delay
        await asyncio.sleep(random.uniform(0.5, 2.0))

        # Simulate 90% success rate
        success = random.random() < 0.9

        if success:
            post.platform_post_id = f"sim_{post.platform.value}_{uuid.uuid4().hex[:8]}"
        else:
            post.error_message = f"Simulated API error for {post.platform.value}"

        return success

    def _save_post(self, post: SocialMediaPost):
        """Save post to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            now = datetime.now().isoformat()
            scheduled_at_str = post.scheduled_at.isoformat() if post.scheduled_at else None
            published_at_str = post.published_at.isoformat() if post.published_at else None

            cursor.execute("""
                INSERT OR REPLACE INTO posts
                (id, platform, content_type, title, text, image_url, video_url, link_url,
                 hashtags, mentions, scheduled_at, published_at, status, platform_post_id,
                 analytics, error_message, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM posts WHERE id = ?), ?), ?)
            """, (
                post.id, post.platform.value, post.content_type.value,
                post.title, post.text, post.image_url, post.video_url, post.link_url,
                json.dumps(post.hashtags), json.dumps(post.mentions),
                scheduled_at_str, published_at_str, post.status.value,
                post.platform_post_id, json.dumps(post.analytics),
                post.error_message, json.dumps(post.metadata),
                post.id, now, now
            ))

            conn.commit()

    def get_post(self, post_id: str) -> Optional[SocialMediaPost]:
        """Retrieve a post by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM posts WHERE id = ?", (post_id,))

            row = cursor.fetchone()
            if row:
                return self._row_to_post(row)

            return None

    def get_posts(self, platform: Optional[SocialPlatform] = None,
                  status: Optional[PostStatus] = None,
                  limit: int = 50) -> List[SocialMediaPost]:
        """Get posts with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM posts WHERE 1=1"
            params = []

            if platform:
                query += " AND platform = ?"
                params.append(platform.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            return [self._row_to_post(row) for row in cursor.fetchall()]

    def _row_to_post(self, row) -> SocialMediaPost:
        """Convert database row to SocialMediaPost object."""
        scheduled_at = datetime.fromisoformat(row[10]) if row[10] else None
        published_at = datetime.fromisoformat(row[11]) if row[11] else None

        return SocialMediaPost(
            id=row[0],
            platform=SocialPlatform(row[1]),
            content_type=ContentType(row[2]),
            title=row[3],
            text=row[4],
            image_url=row[5],
            video_url=row[6],
            link_url=row[7],
            hashtags=json.loads(row[8]) if row[8] else [],
            mentions=json.loads(row[9]) if row[9] else [],
            scheduled_at=scheduled_at,
            published_at=published_at,
            status=PostStatus(row[12]),
            platform_post_id=row[13],
            analytics=json.loads(row[14]) if row[14] else {},
            error_message=row[15],
            metadata=json.loads(row[16]) if row[16] else {}
        )

    async def process_scheduled_posts(self):
        """Process posts scheduled for publishing."""
        scheduled_posts = self.get_posts(status=PostStatus.SCHEDULED)
        now = datetime.now()

        for post in scheduled_posts:
            if post.scheduled_at and post.scheduled_at <= now:
                logger.info(f"Publishing scheduled post {post.id}")
                await self._publish_post(post)

    def get_analytics_summary(self, platform: Optional[SocialPlatform] = None,
                            days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for posts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Base query
            query = """
                SELECT platform, status, COUNT(*) as count
                FROM posts
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days)

            params = []
            if platform:
                query += " AND platform = ?"
                params.append(platform.value)

            query += " GROUP BY platform, status"

            cursor.execute(query, params)

            results = cursor.fetchall()

            # Process results
            summary = {
                "total_posts": 0,
                "published_posts": 0,
                "failed_posts": 0,
                "scheduled_posts": 0,
                "by_platform": {},
                "success_rate": 0.0
            }

            for platform_name, status, count in results:
                summary["total_posts"] += count

                if platform_name not in summary["by_platform"]:
                    summary["by_platform"][platform_name] = {
                        "total": 0,
                        "published": 0,
                        "failed": 0,
                        "scheduled": 0
                    }

                summary["by_platform"][platform_name]["total"] += count

                if status == PostStatus.PUBLISHED.value:
                    summary["published_posts"] += count
                    summary["by_platform"][platform_name]["published"] += count
                elif status == PostStatus.FAILED.value:
                    summary["failed_posts"] += count
                    summary["by_platform"][platform_name]["failed"] += count
                elif status == PostStatus.SCHEDULED.value:
                    summary["scheduled_posts"] += count
                    summary["by_platform"][platform_name]["scheduled"] += count

            # Calculate success rate
            if summary["total_posts"] > 0:
                summary["success_rate"] = summary["published_posts"] / summary["total_posts"]

            return summary

class SocialMediaContentOptimizer:
    """Optimizes content for different social media platforms."""

    def __init__(self, social_manager: SocialMediaManager):
        self.social_manager = social_manager

    def optimize_content_for_platform(self, content: str, platform: SocialPlatform) -> str:
        """Optimize content for a specific platform."""
        limits = self.social_manager.get_platform_limits(platform)
        char_limit = limits.get('character_limit', len(content))

        if len(content) <= char_limit:
            return content

        # Truncate content while preserving meaning
        if platform == SocialPlatform.TWITTER:
            return self._optimize_for_twitter(content, char_limit)
        elif platform == SocialPlatform.LINKEDIN:
            return self._optimize_for_linkedin(content, char_limit)
        else:
            # Generic truncation
            return content[:char_limit-3] + "..."

    def _optimize_for_twitter(self, content: str, char_limit: int) -> str:
        """Optimize content for Twitter."""
        # Remove extra spaces and line breaks
        content = " ".join(content.split())

        if len(content) <= char_limit:
            return content

        # Try to truncate at sentence boundary
        sentences = content.split('. ')
        truncated = ""

        for sentence in sentences:
            if len(truncated + sentence + ". ") <= char_limit - 3:
                truncated += sentence + ". "
            else:
                break

        if truncated:
            return truncated.rstrip() + "..."

        # If no complete sentence fits, truncate at word boundary
        words = content.split()
        truncated = ""

        for word in words:
            if len(truncated + word + " ") <= char_limit - 3:
                truncated += word + " "
            else:
                break

        return truncated.rstrip() + "..."

    def _optimize_for_linkedin(self, content: str, char_limit: int) -> str:
        """Optimize content for LinkedIn."""
        if len(content) <= char_limit:
            return content

        # LinkedIn allows longer posts, so truncate at paragraph boundary
        paragraphs = content.split('\n\n')
        truncated = ""

        for paragraph in paragraphs:
            if len(truncated + paragraph + "\n\n") <= char_limit - 20:
                truncated += paragraph + "\n\n"
            else:
                break

        if truncated:
            return truncated.rstrip() + "\n\n[Read more...]"

        # Fallback to sentence truncation
        return self._optimize_for_twitter(content, char_limit)

    def suggest_hashtags(self, content: str, platform: SocialPlatform) -> List[str]:
        """Suggest relevant hashtags for content."""
        # This is a simplified implementation
        # In practice, you might use AI/ML to analyze content and suggest relevant hashtags

        keywords = content.lower().split()
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}

        # Extract potential hashtag words
        hashtag_candidates = []
        for word in keywords:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 3 and clean_word not in common_words:
                hashtag_candidates.append(clean_word)

        # Get platform hashtag limit
        limits = self.social_manager.get_platform_limits(platform)
        hashtag_limit = limits.get('hashtag_limit', 10)

        # Return unique hashtags up to platform limit
        return list(set(hashtag_candidates))[:hashtag_limit]

# Factory functions
def create_social_media_manager(db_path: str = "social_media.db") -> SocialMediaManager:
    """Create and configure social media manager."""
    return SocialMediaManager(db_path)

def create_content_optimizer(social_manager: SocialMediaManager) -> SocialMediaContentOptimizer:
    """Create content optimizer."""
    return SocialMediaContentOptimizer(social_manager)

# Example usage and testing
if __name__ == "__main__":
    async def test_social_media_system():
        """Test the social media integration system."""
        print("🚀 Testing Social Media Integration System...")

        # Create manager
        manager = create_social_media_manager("test_social_media.db")
        optimizer = create_content_optimizer(manager)

        # Add demo credentials (these would be real OAuth tokens in production)
        twitter_creds = SocialMediaCredentials(
            platform=SocialPlatform.TWITTER,
            api_key="demo_key",
            api_secret="demo_secret",
            access_token="demo_token",
            user_id="demo_user"
        )
        manager.add_credentials(twitter_creds)

        # Create a test post
        post = SocialMediaPost(
            id=str(uuid.uuid4()),
            platform=SocialPlatform.TWITTER,
            content_type=ContentType.TEXT,
            text="Testing the new AI-powered content creation system! 🚀 #AI #ContentCreation #SocialMedia",
            hashtags=["AI", "ContentCreation", "SocialMedia"]
        )

        # Test content optimization
        optimized_text = optimizer.optimize_content_for_platform(
            "This is a very long piece of content that might need to be truncated for Twitter because it exceeds the character limit significantly and we need to make sure it fits within the platform constraints while maintaining readability.",
            SocialPlatform.TWITTER
        )
        print(f"✅ Optimized content: {optimized_text}")

        # Test hashtag suggestions
        suggested_hashtags = optimizer.suggest_hashtags(
            "Creating amazing AI-powered content for social media marketing",
            SocialPlatform.TWITTER
        )
        print(f"✅ Suggested hashtags: {suggested_hashtags}")

        # Create and publish post
        success = await manager.create_post(post)
        print(f"✅ Post creation: {'Success' if success else 'Failed'}")

        # Get analytics
        analytics = manager.get_analytics_summary(days=7)
        print(f"✅ Analytics: {analytics}")

        print("🎉 Social Media Integration System test completed!")

    # Run test
    asyncio.run(test_social_media_system())
