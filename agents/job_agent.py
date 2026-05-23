"""
Job Application Agent
Handles JD analysis, skill matching, and resume tailoring for Tejbahadur Singh.
"""

import json
import re
from pathlib import Path
from typing import Any

from agents.llm_client import LMStudioClient


# -----------------------------------------------------------------------
# Safe token budget constants
# Phi-4-mini-reasoning has ~4K context window on local hardware.
# We leave ~1K for system prompt + JSON format instructions + response.
# So resume + JD combined must stay under ~2500 chars total.
# -----------------------------------------------------------------------
JD_CHAR_LIMIT     = 1200   # characters sent from JD text
RESUME_CHAR_LIMIT = 1000   # characters sent from resume text


class JobApplicationAgent:

    def __init__(self):
        self.llm = LMStudioClient()
        self.master_resume_path = Path("data/resumes/tejbahadur-updated-resume.pdf")

    # ------------------------------------------------------------------
    # Resume Loader
    # ------------------------------------------------------------------

    def load_master_resume(self) -> str:
        if not self.master_resume_path.exists():
            return "No resume found. Please add your resume to data/resumes/."

        try:
            if self.master_resume_path.suffix.lower() == ".pdf":
                import PyPDF2
                text = ""
                with open(self.master_resume_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
                if not text.strip():
                    return "Resume PDF was read but no text could be extracted."
                return text

            return self.master_resume_path.read_text(encoding="utf-8")

        except Exception as e:
            return f"Error reading resume file: {str(e)}"

    # ------------------------------------------------------------------
    # JSON Extractor
    # ------------------------------------------------------------------

    def _extract_json(self, response: Any) -> dict:
        # Guard: must be a non-empty string
        if not response or not isinstance(response, str) or not response.strip():
            return {
                "error": "LLM returned an empty or non-string response.",
                "raw": str(response)
            }

        # Step 1: Remove <think>...</think> reasoning blocks
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Step 2: Strip markdown code fences
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Step 3: Locate outermost JSON object
        json_start = cleaned.find("{")
        json_end   = cleaned.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return {
                "error": "No JSON object found in LLM response.",
                "raw": response
            }

        json_text = cleaned[json_start : json_end + 1]

        # Step 4: Parse
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw": response
            }

    # ------------------------------------------------------------------
    # Payload Size Guard
    # ------------------------------------------------------------------

    def _safe_payload_size(self, messages: list) -> int:
        """Returns total character count of all message content combined."""
        return sum(len(str(m.get("content", ""))) for m in messages)

    # ------------------------------------------------------------------
    # JD Analysis
    # ------------------------------------------------------------------

    def analyze_jd(self, jd_text: str) -> dict:
        # Truncate JD to safe limit
        jd_snippet = jd_text[:JD_CHAR_LIMIT]

        prompt = (
            "Analyze the Job Description below.\n"
            "Return ONLY valid JSON. No explanation. No markdown. No extra text.\n\n"
            "Format:\n"
            '{\n'
            '  "tech_stack": ["tech1", "tech2"],\n'
            '  "experience_required": "years/level",\n'
            '  "key_challenges": ["challenge1", "challenge2"],\n'
            '  "soft_skills": ["skill1", "skill2"]\n'
            '}\n\n'
            f"Job Description:\n{jd_snippet}"
        )

        messages = [
            {"role": "system", "content": "You are a JSON-only API. Output ONLY valid JSON. No prose."},
            {"role": "user",   "content": prompt}
        ]

        # Guard: log payload size before sending
        payload_size = self._safe_payload_size(messages)
        print(f"[analyze_jd] Payload size: {payload_size} chars")

        try:
            response = self.llm.chat(messages)
        except Exception as e:
            return {
                "tech_stack": [],
                "experience_required": "N/A",
                "key_challenges": [],
                "soft_skills": [],
                "error": f"LLM call failed in analyze_jd: {str(e)}",
                "raw": ""
            }

        parsed = self._extract_json(response)

        if "error" in parsed:
            return {
                "tech_stack": [],
                "experience_required": "N/A",
                "key_challenges": [],
                "soft_skills": [],
                "error": parsed["error"],
                "raw": parsed.get("raw", response)
            }

        return parsed

    # ------------------------------------------------------------------
    # Skill Matching
    # ------------------------------------------------------------------

    def match_skills(self, jd_analysis: dict) -> dict:

        # Guard: if JD analysis itself failed, skip the LLM call entirely
        if "error" in jd_analysis and not jd_analysis.get("tech_stack"):
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": f"Skipped — JD analysis failed: {jd_analysis.get('error', 'unknown')}",
                "raw": ""
            }

        resume_text = self.load_master_resume()

        # Truncate both inputs to safe limits
        resume_snippet = resume_text[:RESUME_CHAR_LIMIT]

        # Only send essential JD fields — drop "raw" and "error" keys
        jd_clean = {
            "tech_stack":           jd_analysis.get("tech_stack", []),
            "experience_required":  jd_analysis.get("experience_required", "N/A"),
            "key_challenges":       jd_analysis.get("key_challenges", []),
            "soft_skills":          jd_analysis.get("soft_skills", [])
        }
        jd_json = json.dumps(jd_clean, ensure_ascii=False)

        prompt = (
            "Compare the resume with the job requirements below.\n"
            "Return ONLY valid JSON. No explanation. No markdown. No extra text.\n\n"
            "Format:\n"
            '{\n'
            '  "matched_skills": ["skill1", "skill2"],\n'
            '  "missing_skills": ["skill3", "skill4"],\n'
            '  "relevance_score": 85\n'
            '}\n\n'
            "The relevance_score must be an integer between 0 and 100.\n\n"
            f"Resume:\n{resume_snippet}\n\n"
            f"Job Requirements:\n{jd_json}"
        )

        messages = [
            {"role": "system", "content": "You are a JSON-only API. Output ONLY valid JSON. No prose."},
            {"role": "user",   "content": prompt}
        ]

        # log and check payload size before sending
        payload_size = self._safe_payload_size(messages)
        print(f"[match_skills] Payload size: {payload_size} chars")

        if payload_size > 4000:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": f"Payload too large ({payload_size} chars). Reduce JD or resume size.",
                "raw": ""
            }

        try:
            response = self.llm.chat(messages)
        except Exception as e:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": f"LLM call failed in match_skills: {str(e)}",
                "raw": ""
            }

        parsed = self._extract_json(response)

        if "error" in parsed:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "relevance_score": 0,
                "error": parsed["error"],
                "raw": parsed.get("raw", response)
            }

        return {
            "matched_skills":  parsed.get("matched_skills", []),
            "missing_skills":  parsed.get("missing_skills", []),
            "relevance_score": parsed.get("relevance_score", 0),
            "error": None,
            "raw": response
        }
