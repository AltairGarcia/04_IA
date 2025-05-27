"""
Export and Data Portability System

This module provides comprehensive data export and import capabilities
including multiple formats, batch operations, scheduled exports, and
data transformation for the content creation platform.
"""

import os
import json
import csv
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, IO
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from pathlib import Path
import logging
import uuid
import base64
import mimetypes
from io import BytesIO, StringIO
import yaml
import pickle
import asyncio
import aiofiles
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import markdown

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    EXCEL = "excel"
    ZIP = "zip"
    BACKUP = "backup"
    SQL = "sql"

class ExportScope(Enum):
    """Scope of data to export."""
    ALL = "all"
    CONTENT = "content"
    ANALYTICS = "analytics"
    USERS = "users"
    SETTINGS = "settings"
    CUSTOM = "custom"

class ExportStatus(Enum):
    """Export job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExportJob:
    """Export job configuration and status."""
    id: str
    name: str
    format: ExportFormat
    scope: ExportScope
    filters: Dict[str, Any]
    include_media: bool = False
    compress: bool = False
    password_protected: bool = False
    scheduled: bool = False
    schedule_pattern: Optional[str] = None
    status: ExportStatus = ExportStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

class DataExportManager:
    """Manages data export and import operations."""

    def __init__(self, export_dir: str = "exports", db_path: str = "content_creation.db"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.db_path = db_path
        self.temp_dir = self.export_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize export database
        self._initialize_export_database()

    def _initialize_export_database(self):
        """Initialize export tracking database."""
        export_db_path = self.export_dir / "exports.db"

        with sqlite3.connect(export_db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS export_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    format TEXT,
                    scope TEXT,
                    filters TEXT,
                    include_media BOOLEAN,
                    compress BOOLEAN,
                    password_protected BOOLEAN,
                    scheduled BOOLEAN,
                    schedule_pattern TEXT,
                    status TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    error_message TEXT,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS export_schedules (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    pattern TEXT,
                    next_run TEXT,
                    last_run TEXT,
                    is_active BOOLEAN,
                    FOREIGN KEY (job_id) REFERENCES export_jobs (id)
                )
            """)

            conn.commit()

    async def create_export_job(self, job: ExportJob) -> bool:
        """Create and start an export job."""
        try:
            # Save job to database
            self._save_export_job(job)

            # Start export process
            job.status = ExportStatus.PROCESSING
            job.started_at = datetime.now()
            self._save_export_job(job)

            # Export data based on format
            if job.format == ExportFormat.JSON:
                success = await self._export_json(job)
            elif job.format == ExportFormat.CSV:
                success = await self._export_csv(job)
            elif job.format == ExportFormat.XML:
                success = await self._export_xml(job)
            elif job.format == ExportFormat.YAML:
                success = await self._export_yaml(job)
            elif job.format == ExportFormat.PDF:
                success = await self._export_pdf(job)
            elif job.format == ExportFormat.HTML:
                success = await self._export_html(job)
            elif job.format == ExportFormat.MARKDOWN:
                success = await self._export_markdown(job)
            elif job.format == ExportFormat.EXCEL:
                success = await self._export_excel(job)
            elif job.format == ExportFormat.ZIP:
                success = await self._export_zip(job)
            elif job.format == ExportFormat.BACKUP:
                success = await self._export_backup(job)
            elif job.format == ExportFormat.SQL:
                success = await self._export_sql(job)
            else:
                job.error_message = f"Unsupported export format: {job.format.value}"
                success = False

            # Update job status
            job.completed_at = datetime.now()
            if success:
                job.status = ExportStatus.COMPLETED
                # Get file size
                if job.file_path and os.path.exists(job.file_path):
                    job.file_size = os.path.getsize(job.file_path)
            else:
                job.status = ExportStatus.FAILED

            self._save_export_job(job)
            return success

        except Exception as e:
            job.status = ExportStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._save_export_job(job)
            logger.error(f"Export job {job.id} failed: {e}")
            return False

    async def _export_json(self, job: ExportJob) -> bool:
        """Export data as JSON."""
        try:
            data = await self._gather_export_data(job)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.json"
            file_path = self.export_dir / filename

            # Write JSON file
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, default=str, ensure_ascii=False))

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"JSON export failed: {e}"
            return False

    async def _export_csv(self, job: ExportJob) -> bool:
        """Export data as CSV."""
        try:
            data = await self._gather_export_data(job)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if job.scope == ExportScope.ALL:
                # Create ZIP file with multiple CSV files
                zip_filename = f"{job.name}_{timestamp}.zip"
                zip_path = self.export_dir / zip_filename

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for table_name, table_data in data.items():
                        if isinstance(table_data, list) and table_data:
                            csv_content = self._list_to_csv(table_data)
                            zipf.writestr(f"{table_name}.csv", csv_content)

                job.file_path = str(zip_path)
            else:
                # Single CSV file
                csv_filename = f"{job.name}_{timestamp}.csv"
                csv_path = self.export_dir / csv_filename

                # Flatten data if needed
                if isinstance(data, dict) and len(data) == 1:
                    table_data = list(data.values())[0]
                else:
                    table_data = data

                csv_content = self._list_to_csv(table_data)

                async with aiofiles.open(csv_path, 'w', encoding='utf-8') as f:
                    await f.write(csv_content)

                job.file_path = str(csv_path)

            return True

        except Exception as e:
            job.error_message = f"CSV export failed: {e}"
            return False

    def _list_to_csv(self, data: List[Dict[str, Any]]) -> str:
        """Convert list of dictionaries to CSV string."""
        if not data:
            return ""

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    async def _export_xml(self, job: ExportJob) -> bool:
        """Export data as XML."""
        try:
            data = await self._gather_export_data(job)

            # Create XML structure
            root = ET.Element("export")
            root.set("created_at", datetime.now().isoformat())
            root.set("format", job.format.value)
            root.set("scope", job.scope.value)

            # Add metadata
            metadata_elem = ET.SubElement(root, "metadata")
            for key, value in job.metadata.items():
                meta_elem = ET.SubElement(metadata_elem, key)
                meta_elem.text = str(value)

            # Add data
            self._dict_to_xml(data, root)

            # Generate filename and save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.xml"
            file_path = self.export_dir / filename

            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"XML export failed: {e}"
            return False

    def _dict_to_xml(self, data: Any, parent: ET.Element, name: str = "data"):
        """Convert dictionary to XML elements recursively."""
        if isinstance(data, dict):
            for key, value in data.items():
                elem = ET.SubElement(parent, key)
                if isinstance(value, (dict, list)):
                    self._dict_to_xml(value, elem, key)
                else:
                    elem.text = str(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                elem = ET.SubElement(parent, "item")
                elem.set("index", str(i))
                self._dict_to_xml(item, elem, "item")
        else:
            parent.text = str(data)

    async def _export_yaml(self, job: ExportJob) -> bool:
        """Export data as YAML."""
        try:
            data = await self._gather_export_data(job)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.yaml"
            file_path = self.export_dir / filename

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                yaml_content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
                await f.write(yaml_content)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"YAML export failed: {e}"
            return False

    async def _export_pdf(self, job: ExportJob) -> bool:
        """Export data as PDF report."""
        try:
            data = await self._gather_export_data(job)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.pdf"
            file_path = self.export_dir / filename

            # Create PDF document
            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph(f"Data Export Report: {job.name}", title_style))
            story.append(Spacer(1, 12))

            # Metadata
            meta_data = [
                ["Export Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Format", job.format.value.upper()],
                ["Scope", job.scope.value.title()],
                ["Include Media", "Yes" if job.include_media else "No"]
            ]

            meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(meta_table)
            story.append(Spacer(1, 24))

            # Data summary
            story.append(Paragraph("Data Summary", styles['Heading2']))
            story.append(Spacer(1, 12))

            if isinstance(data, dict):
                for section_name, section_data in data.items():
                    story.append(Paragraph(f"{section_name.title()}", styles['Heading3']))

                    if isinstance(section_data, list):
                        story.append(Paragraph(f"Records: {len(section_data)}", styles['Normal']))

                        # Show sample data if available
                        if section_data and len(section_data) > 0:
                            sample = section_data[0]
                            if isinstance(sample, dict):
                                sample_data = [[k, str(v)[:50] + "..." if len(str(v)) > 50 else str(v)]
                                             for k, v in list(sample.items())[:5]]

                                sample_table = Table(sample_data, colWidths=[2*inch, 4*inch])
                                sample_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))

                                story.append(Paragraph("Sample Record:", styles['Normal']))
                                story.append(sample_table)

                    story.append(Spacer(1, 12))

            # Build PDF
            doc.build(story)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"PDF export failed: {e}"
            return False

    async def _export_html(self, job: ExportJob) -> bool:
        """Export data as HTML report."""
        try:
            data = await self._gather_export_data(job)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.html"
            file_path = self.export_dir / filename

            # HTML template
            html_template = Template("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Data Export Report - {{ job_name }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                    .metadata { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
                    .section { margin-bottom: 40px; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                    th { background-color: #f2f2f2; font-weight: bold; }
                    .record-count { color: #7f8c8d; font-style: italic; }
                    .sample-data { background-color: #ecf0f1; padding: 15px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Data Export Report: {{ job_name }}</h1>

                <div class="metadata">
                    <h2>Export Information</h2>
                    <table>
                        <tr><th>Export Date</th><td>{{ export_date }}</td></tr>
                        <tr><th>Format</th><td>{{ format }}</td></tr>
                        <tr><th>Scope</th><td>{{ scope }}</td></tr>
                        <tr><th>Include Media</th><td>{{ include_media }}</td></tr>
                    </table>
                </div>

                {% for section_name, section_data in data.items() %}
                <div class="section">
                    <h2>{{ section_name.title() }}</h2>
                    {% if section_data is iterable and section_data is not string %}
                        <p class="record-count">Records: {{ section_data|length }}</p>
                        {% if section_data %}
                            <div class="sample-data">
                                <strong>Sample Record:</strong>
                                <pre>{{ section_data[0] | tojson(indent=2) }}</pre>
                            </div>
                        {% endif %}
                    {% else %}
                        <p>{{ section_data }}</p>
                    {% endif %}
                </div>
                {% endfor %}

                <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
                    Generated on {{ export_date }} by Content Creation Platform
                </footer>
            </body>
            </html>
            """)

            html_content = html_template.render(
                job_name=job.name,
                export_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                format=job.format.value.upper(),
                scope=job.scope.value.title(),
                include_media="Yes" if job.include_media else "No",
                data=data
            )

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(html_content)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"HTML export failed: {e}"
            return False

    async def _export_markdown(self, job: ExportJob) -> bool:
        """Export data as Markdown."""
        try:
            data = await self._gather_export_data(job)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.md"
            file_path = self.export_dir / filename

            # Generate Markdown content
            md_content = f"# Data Export Report: {job.name}\n\n"
            md_content += f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
            md_content += f"**Format:** {job.format.value.upper()}  \n"
            md_content += f"**Scope:** {job.scope.value.title()}  \n"
            md_content += f"**Include Media:** {'Yes' if job.include_media else 'No'}  \n\n"

            md_content += "---\n\n"

            # Add data sections
            if isinstance(data, dict):
                for section_name, section_data in data.items():
                    md_content += f"## {section_name.title()}\n\n"

                    if isinstance(section_data, list):
                        md_content += f"**Records:** {len(section_data)}\n\n"

                        if section_data and isinstance(section_data[0], dict):
                            # Create table
                            headers = list(section_data[0].keys())
                            md_content += "| " + " | ".join(headers) + " |\n"
                            md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"

                            # Add sample rows (first 5)
                            for row in section_data[:5]:
                                values = [str(row.get(h, "")).replace("|", "\\|")[:50] for h in headers]
                                md_content += "| " + " | ".join(values) + " |\n"

                            if len(section_data) > 5:
                                md_content += f"\n*... and {len(section_data) - 5} more records*\n"
                    else:
                        md_content += f"```\n{json.dumps(section_data, indent=2, default=str)}\n```\n"

                    md_content += "\n"

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(md_content)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"Markdown export failed: {e}"
            return False

    async def _export_excel(self, job: ExportJob) -> bool:
        """Export data as Excel workbook."""
        try:
            data = await self._gather_export_data(job)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.xlsx"
            file_path = self.export_dir / filename

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Metadata sheet
                metadata_df = pd.DataFrame([
                    ["Export Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["Format", job.format.value.upper()],
                    ["Scope", job.scope.value.title()],
                    ["Include Media", "Yes" if job.include_media else "No"]
                ], columns=["Property", "Value"])

                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

                # Data sheets
                if isinstance(data, dict):
                    for sheet_name, sheet_data in data.items():
                        if isinstance(sheet_data, list) and sheet_data:
                            df = pd.DataFrame(sheet_data)
                            # Truncate sheet name to Excel limit
                            safe_sheet_name = sheet_name[:31]
                            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"Excel export failed: {e}"
            return False

    async def _export_zip(self, job: ExportJob) -> bool:
        """Export data as compressed ZIP archive."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"{job.name}_{timestamp}.zip"
            zip_path = self.export_dir / zip_filename

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Create multiple format exports
                temp_job = ExportJob(
                    id=f"temp_{uuid.uuid4().hex[:8]}",
                    name=job.name,
                    format=ExportFormat.JSON,
                    scope=job.scope,
                    filters=job.filters
                )

                # JSON export
                if await self._export_json(temp_job):
                    zipf.write(temp_job.file_path, f"{job.name}.json")
                    os.remove(temp_job.file_path)

                # CSV export
                temp_job.format = ExportFormat.CSV
                if await self._export_csv(temp_job):
                    if temp_job.file_path.endswith('.zip'):
                        # Extract CSV files from the inner ZIP
                        with zipfile.ZipFile(temp_job.file_path, 'r') as inner_zip:
                            for file_info in inner_zip.infolist():
                                zipf.writestr(file_info.filename, inner_zip.read(file_info))
                    else:
                        zipf.write(temp_job.file_path, f"{job.name}.csv")
                    os.remove(temp_job.file_path)

                # Add media files if requested
                if job.include_media:
                    await self._add_media_to_zip(zipf, job)

            job.file_path = str(zip_path)
            return True

        except Exception as e:
            job.error_message = f"ZIP export failed: {e}"
            return False

    async def _export_backup(self, job: ExportJob) -> bool:
        """Create full system backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{timestamp}.zip"
            backup_path = self.export_dir / backup_filename

            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add database files
                if os.path.exists(self.db_path):
                    zipf.write(self.db_path, "database/content_creation.db")

                # Add configuration files
                config_files = [".env", "config.json", "settings.yaml"]
                for config_file in config_files:
                    if os.path.exists(config_file):
                        zipf.write(config_file, f"config/{config_file}")

                # Add all data exports
                data = await self._gather_export_data(job)

                # JSON backup of all data
                json_content = json.dumps(data, indent=2, default=str, ensure_ascii=False)
                zipf.writestr("data/full_export.json", json_content)

                # Add media files if requested
                if job.include_media:
                    await self._add_media_to_zip(zipf, job)

                # Add backup metadata
                backup_metadata = {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "platform": "content_creation",
                    "scope": job.scope.value,
                    "include_media": job.include_media
                }

                zipf.writestr("backup_info.json", json.dumps(backup_metadata, indent=2))

            job.file_path = str(backup_path)
            return True

        except Exception as e:
            job.error_message = f"Backup export failed: {e}"
            return False

    async def _export_sql(self, job: ExportJob) -> bool:
        """Export data as SQL dump."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job.name}_{timestamp}.sql"
            file_path = self.export_dir / filename

            sql_content = f"-- SQL Export for {job.name}\n"
            sql_content += f"-- Generated on {datetime.now().isoformat()}\n\n"

            # Connect to database and export tables
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                for table in tables:
                    # Skip if filtering by scope
                    if job.scope != ExportScope.ALL:
                        if not self._table_matches_scope(table, job.scope):
                            continue

                    # Get table schema
                    cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}'")
                    create_statement = cursor.fetchone()[0]
                    sql_content += f"{create_statement};\n\n"

                    # Get table data
                    cursor.execute(f"SELECT * FROM {table}")
                    rows = cursor.fetchall()

                    if rows:
                        # Get column names
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [row[1] for row in cursor.fetchall()]

                        # Generate INSERT statements
                        for row in rows:
                            values = []
                            for value in row:
                                if value is None:
                                    values.append("NULL")
                                elif isinstance(value, str):
                                    values.append(f"'{value.replace('\'', '\'\'')}'")
                                else:
                                    values.append(str(value))

                            sql_content += f"INSERT INTO {table} ({', '.join(columns)}) "
                            sql_content += f"VALUES ({', '.join(values)});\n"

                        sql_content += "\n"

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(sql_content)

            job.file_path = str(file_path)
            return True

        except Exception as e:
            job.error_message = f"SQL export failed: {e}"
            return False

    def _table_matches_scope(self, table_name: str, scope: ExportScope) -> bool:
        """Check if table matches export scope."""
        scope_mappings = {
            ExportScope.CONTENT: ["content_history", "posts", "media"],
            ExportScope.ANALYTICS: ["analytics_data", "api_usage", "performance_metrics"],
            ExportScope.USERS: ["users", "sessions", "api_keys"],
            ExportScope.SETTINGS: ["settings", "preferences", "configurations"]
        }

        if scope in scope_mappings:
            return any(keyword in table_name.lower() for keyword in scope_mappings[scope])

        return True

    async def _gather_export_data(self, job: ExportJob) -> Dict[str, Any]:
        """Gather data based on export scope and filters."""
        data = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if job.scope == ExportScope.ALL or job.scope == ExportScope.CONTENT:
                    # Get content data
                    cursor.execute("SELECT * FROM content_history ORDER BY created_at DESC")
                    content_rows = cursor.fetchall()
                    data["content"] = [dict(row) for row in content_rows]

                if job.scope == ExportScope.ALL or job.scope == ExportScope.ANALYTICS:
                    # Get analytics data
                    try:
                        cursor.execute("SELECT * FROM analytics_data ORDER BY recorded_at DESC")
                        analytics_rows = cursor.fetchall()
                        data["analytics"] = [dict(row) for row in analytics_rows]
                    except sqlite3.OperationalError:
                        data["analytics"] = []

                    try:
                        cursor.execute("SELECT * FROM api_usage ORDER BY timestamp DESC")
                        api_usage_rows = cursor.fetchall()
                        data["api_usage"] = [dict(row) for row in api_usage_rows]
                    except sqlite3.OperationalError:
                        data["api_usage"] = []

                if job.scope == ExportScope.ALL or job.scope == ExportScope.USERS:
                    # Get user data (if exists)
                    try:
                        cursor.execute("SELECT id, username, email, role, created_at, last_login FROM users")
                        user_rows = cursor.fetchall()
                        data["users"] = [dict(row) for row in user_rows]
                    except sqlite3.OperationalError:
                        data["users"] = []

                if job.scope == ExportScope.ALL or job.scope == ExportScope.SETTINGS:
                    # Get settings and preferences
                    try:
                        cursor.execute("SELECT * FROM user_preferences")
                        pref_rows = cursor.fetchall()
                        data["preferences"] = [dict(row) for row in pref_rows]
                    except sqlite3.OperationalError:
                        data["preferences"] = []

            # Apply filters if specified
            if job.filters:
                data = self._apply_filters(data, job.filters)

            return data

        except Exception as e:
            logger.error(f"Failed to gather export data: {e}")
            return {}

    def _apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to export data."""
        filtered_data = {}

        for section_name, section_data in data.items():
            if isinstance(section_data, list):
                filtered_items = []

                for item in section_data:
                    if self._item_matches_filters(item, filters):
                        filtered_items.append(item)

                filtered_data[section_name] = filtered_items
            else:
                filtered_data[section_name] = section_data

        return filtered_data

    def _item_matches_filters(self, item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if item matches filter criteria."""
        for filter_key, filter_value in filters.items():
            if filter_key in item:
                if isinstance(filter_value, dict):
                    # Range filter
                    if "min" in filter_value and item[filter_key] < filter_value["min"]:
                        return False
                    if "max" in filter_value and item[filter_key] > filter_value["max"]:
                        return False
                elif isinstance(filter_value, list):
                    # List filter (item must be in list)
                    if item[filter_key] not in filter_value:
                        return False
                else:
                    # Exact match
                    if item[filter_key] != filter_value:
                        return False

        return True

    async def _add_media_to_zip(self, zipf: zipfile.ZipFile, job: ExportJob):
        """Add media files to ZIP archive."""
        try:
            # Look for media files in content output directory
            content_output_dir = Path("content_output")
            if content_output_dir.exists():
                for media_file in content_output_dir.glob("*"):
                    if media_file.is_file():
                        # Add to ZIP with media/ prefix
                        zipf.write(media_file, f"media/{media_file.name}")
        except Exception as e:
            logger.warning(f"Failed to add media files: {e}")

    def _save_export_job(self, job: ExportJob):
        """Save export job to database."""
        export_db_path = self.export_dir / "exports.db"

        with sqlite3.connect(export_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO export_jobs
                (id, name, format, scope, filters, include_media, compress,
                 password_protected, scheduled, schedule_pattern, status,
                 created_at, started_at, completed_at, file_path, file_size,
                 error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id,
                job.name,
                job.format.value,
                job.scope.value,
                json.dumps(job.filters),
                job.include_media,
                job.compress,
                job.password_protected,
                job.scheduled,
                job.schedule_pattern,
                job.status.value,
                job.created_at.isoformat() if job.created_at else None,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.file_path,
                job.file_size,
                job.error_message,
                json.dumps(job.metadata)
            ))
            conn.commit()

    def get_export_jobs(self, status: Optional[ExportStatus] = None,
                       limit: int = 50) -> List[ExportJob]:
        """Get export jobs with optional status filtering."""
        export_db_path = self.export_dir / "exports.db"

        with sqlite3.connect(export_db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM export_jobs WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            jobs = []
            for row in cursor.fetchall():
                jobs.append(self._row_to_export_job(row))

            return jobs

    def _row_to_export_job(self, row) -> ExportJob:
        """Convert database row to ExportJob object."""
        return ExportJob(
            id=row[0],
            name=row[1],
            format=ExportFormat(row[2]),
            scope=ExportScope(row[3]),
            filters=json.loads(row[4]) if row[4] else {},
            include_media=bool(row[5]),
            compress=bool(row[6]),
            password_protected=bool(row[7]),
            scheduled=bool(row[8]),
            schedule_pattern=row[9],
            status=ExportStatus(row[10]),
            created_at=datetime.fromisoformat(row[11]) if row[11] else None,
            started_at=datetime.fromisoformat(row[12]) if row[12] else None,
            completed_at=datetime.fromisoformat(row[13]) if row[13] else None,
            file_path=row[14],
            file_size=row[15],
            error_message=row[16],
            metadata=json.loads(row[17]) if row[17] else {}
        )

    def get_export_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get export summary statistics."""
        export_db_path = self.export_dir / "exports.db"

        with sqlite3.connect(export_db_path) as conn:
            cursor = conn.cursor()

            # Total exports
            cursor.execute("""
                SELECT COUNT(*) FROM export_jobs
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days))
            total_exports = cursor.fetchone()[0]

            # Exports by status
            cursor.execute("""
                SELECT status, COUNT(*) FROM export_jobs
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY status
            """.format(days))
            status_counts = dict(cursor.fetchall())

            # Exports by format
            cursor.execute("""
                SELECT format, COUNT(*) FROM export_jobs
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY format
            """.format(days))
            format_counts = dict(cursor.fetchall())

            # Total file size
            cursor.execute("""
                SELECT SUM(file_size) FROM export_jobs
                WHERE created_at >= datetime('now', '-{} days') AND file_size IS NOT NULL
            """.format(days))
            total_size = cursor.fetchone()[0] or 0

            return {
                "total_exports": total_exports,
                "status_counts": status_counts,
                "format_counts": format_counts,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "days": days
            }

# Factory function
def create_export_manager(export_dir: str = "exports",
                        db_path: str = "content_creation.db") -> DataExportManager:
    """Create and configure data export manager."""
    return DataExportManager(export_dir, db_path)

# Example usage and testing
if __name__ == "__main__":
    async def test_export_system():
        """Test the data export system."""
        print("📤 Testing Data Export System...")

        # Create export manager
        export_manager = create_export_manager("test_exports")

        # Create sample export jobs
        json_job = ExportJob(
            id=str(uuid.uuid4()),
            name="content_backup",
            format=ExportFormat.JSON,
            scope=ExportScope.CONTENT,
            filters={}
        )

        csv_job = ExportJob(
            id=str(uuid.uuid4()),
            name="analytics_report",
            format=ExportFormat.CSV,
            scope=ExportScope.ANALYTICS,
            filters={}
        )

        pdf_job = ExportJob(
            id=str(uuid.uuid4()),
            name="summary_report",
            format=ExportFormat.PDF,
            scope=ExportScope.ALL,
            filters={}
        )

        # Test exports
        test_jobs = [json_job, csv_job, pdf_job]

        for job in test_jobs:
            print(f"Testing {job.format.value} export...")
            success = await export_manager.create_export_job(job)
            print(f"✅ {job.format.value} export: {'Success' if success else 'Failed'}")

            if success and job.file_path:
                print(f"   File: {job.file_path}")
                print(f"   Size: {job.file_size} bytes")

        # Get export summary
        summary = export_manager.get_export_summary(days=1)
        print(f"✅ Export summary: {summary}")

        print("🎉 Data Export System test completed!")

    # Run test
    asyncio.run(test_export_system())
