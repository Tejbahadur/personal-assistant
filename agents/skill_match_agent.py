"""
Skill Match Agent
Responsible ONLY for comparing JD analysis against user's resume.
Outputs: matched_skills, missing_skills, relevance_score
Uses: Lightweight LLM prompt for structured matching.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from agents.llm_client import LMStudioClient


class SkillMatchAgent:
    """Matches JD requirements against user's resume."""

    def __init__(self, resume_path: str = "data/resumes/tejbahadur-updated-resume.pdf"):
        self.llm = LMStudioClient()
        self.resume_path = Path(resume_path)

    def load_resume(self) -> str:
        """Load master resume from disk."""
        if not self.resume_path.exists():
            return "No resume found. Please add your resume to data/resumes/."

        try:
            if self.resume_path.suffix.lower() == ".pdf":
                import PyPDF2
                text = ""
                with open(self.resume_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
                return text if text.strip() else "Resume PDF found but no text extracted."
            else:
                return self.resume_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading resume: {str(e)}"

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Robustly extract JSON from LLM response."""
        if not response or not isinstance(response, str):
            return {"error": "Empty response", "raw": str(response)}

        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}")

        if json_start == -1 or json_end == -1:
            return {"error": "No JSON found", "raw": response}

        try:
            return json.loads(cleaned[json_start:json_end + 1])
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {str(e)}", "raw": response}

    def match(self, jd_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare JD analysis against resume.
        Returns: matched_skills, missing_skills, relevance_score (0-100)
        """
        if "error" in jd_analysis:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": f"Skipped due to JD analysis error: {jd_analysis.get('error')}"
            }

        resume_text = self.load_resume()
        resume_snippet = resume_text[:1200]  # Safe limit

        # Prepare compact JD data (only essential fields)
        jd_clean = {
            "tech_stack": jd_analysis.get("tech_stack", []),
            "experience_required": jd_analysis.get("experience_required", "N/A"),
            "key_challenges": jd_analysis.get("key_challenges", []),
            "soft_skills": jd_analysis.get("soft_skills", [])
        }

        prompt = f"""Compare the resume with job requirements and identify matched and missing skills.

Return ONLY valid JSON in this exact format:
{{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "relevance_score": 85
}}

The relevance_score must be an integer between 0 and 100.

Resume:
{resume_snippet}

Job Requirements:
{json.dumps(jd_clean, ensure_ascii=False)}
"""

        messages = [
            {
                "role": "system",
                "content": "You are a JSON-only API. Output ONLY valid JSON. No prose, no markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload_size = sum(len(str(m.get("content", ""))) for m in messages)
        print(f"[SkillMatchAgent] Payload {payload_size} chars")

        if payload_size > 4000:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": f"Payload too large ({payload_size} chars). Truncate JD or resume."
            }

        try:
            response = self.llm.chat(messages)
            parsed = self._extract_json(response)

            if "error" in parsed:
                return {
                    "matched_skills": [],
                    "missing_skills": [],
                    "relevance_score": 0,
                    "error": parsed["error"]
                }

            return {
                "matched_skills": parsed.get("matched_skills", []),
                "missing_skills": parsed.get("missing_skills", []),
                "relevance_score": parsed.get("relevance_score", 0)
            }

        except Exception as e:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": f"Skill matching failed: {str(e)}"
            }
