"""
Storage Agent
Responsible ONLY for file persistence and database tracking.
Creates folder structure, saves documents, updates SQLite.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from agents.db_manager import DatabaseManager


class StorageAgent:
    """Handles file saving and database persistence."""

    def __init__(self, base_path: str = "data/applications"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager()

    def _sanitize_name(self, name: str) -> str:
        """Convert a string to a safe folder name."""
        return "".join([c if c.isalnum() else "_" for c in name]).strip("_")

    def save_application(
        self,
        company: str,
        role: str,
        cover_letter: str,
        tailored_resume: str,
        jd_text: str,
        relevance_score: int,
        matched_skills: List[str],
        missing_skills: List[str],
        key_challenges: List[str]
    ) -> Dict[str, Any]:
        """
        Save application artifacts and database entry.
        Returns metadata about the saved application.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe_company = self._sanitize_name(company)
        safe_role = self._sanitize_name(role)

        folder_name = f"{date_str}_{safe_company}_{safe_role}"
        folder_path = self.base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Save files
        cl_file = folder_path / "Cover_Letter.txt"
        res_file = folder_path / "Tailored_Resume.md"
        jd_file = folder_path / "Job_Description.txt"
        metadata_file = folder_path / "metadata.txt"

        cl_file.write_text(cover_letter, encoding="utf-8")
        res_file.write_text(tailored_resume, encoding="utf-8")
        jd_file.write_text(jd_text, encoding="utf-8")

        # Write metadata
        metadata_content = f"""Application Metadata
===================
Company: {company}
Role: {role}
Date Generated: {datetime.now().isoformat()}
Relevance Score: {relevance_score}%
Matched Skills: {', '.join(matched_skills)}
Missing Skills: {', '.join(missing_skills)}
Key Challenges: {', '.join(key_challenges)}
Folder: {folder_name}
"""
        metadata_file.write_text(metadata_content, encoding="utf-8")

        # Save to SQLite
        app_id = self.db.add_application(
            company=company,
            role=role,
            jd_text=jd_text,
            relevance_score=relevance_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            key_challenges=key_challenges,
            folder_path=str(folder_path)
        )

        return {
            "id": app_id,
            "folder_path": str(folder_path),
            "cover_letter_file": str(cl_file),
            "resume_file": str(res_file),
            "status": "Drafted"
        }

    def mark_approved(self, app_id: int):
        """Mark application as approved."""
        self.db.update_status(app_id, "Approved & Ready")

    def mark_submitted(self, app_id: int):
        """Mark application as submitted."""
        self.db.update_status(app_id, "Submitted")

    def get_application(self, app_id: int) -> Dict[str, Any]:
        """Fetch application details."""
        return self.db.get_application(app_id)

    def list_applications(self, status: str = None) -> List[Dict[str, Any]]:
        """List applications by status."""
        return self.db.list_applications(status)
