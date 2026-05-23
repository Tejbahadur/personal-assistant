"""
Database Manager for application tracking.
Handles SQLite persistence for job applications.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class DatabaseManager:
    def __init__(self, db_path: str = "data/assistant.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    role_title TEXT NOT NULL,
                    jd_text TEXT,
                    relevance_score INTEGER,
                    matched_skills TEXT,
                    missing_skills TEXT,
                    key_challenges TEXT,
                    folder_path TEXT NOT NULL,
                    status TEXT DEFAULT 'Drafted',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP,
                    submitted_at TIMESTAMP
                )
            """)
            conn.commit()

    def add_application(
        self,
        company: str,
        role: str,
        jd_text: str,
        relevance_score: int,
        matched_skills: List[str],
        missing_skills: List[str],
        key_challenges: List[str],
        folder_path: str
    ) -> int:
        """Add a new application to the database."""
        import json
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO applications 
                (company_name, role_title, jd_text, relevance_score, 
                 matched_skills, missing_skills, key_challenges, folder_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                company,
                role,
                jd_text,
                relevance_score,
                json.dumps(matched_skills),
                json.dumps(missing_skills),
                json.dumps(key_challenges),
                folder_path,
                "Drafted"
            ))
            conn.commit()
            return cursor.lastrowid

    def update_status(self, app_id: int, status: str):
        """Update application status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat() if status == "Approved & Ready" else None
            cursor.execute(
                "UPDATE applications SET status = ?, approved_at = ? WHERE id = ?",
                (status, timestamp, app_id)
            )
            conn.commit()

    def get_application(self, app_id: int) -> Optional[Dict]:
        """Fetch a single application."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM applications WHERE id = ?", (app_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_applications(self, status: Optional[str] = None) -> List[Dict]:
        """List all applications, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM applications WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM applications ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
