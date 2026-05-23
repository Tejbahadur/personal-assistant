"""
JD Analyst Agent
Responsible ONLY for analyzing Job Descriptions.
Extracts: tech_stack, experience_required, key_challenges, soft_skills
Uses: Lightweight LLM prompt for fast structured output.
"""

import json
import re
from typing import Dict, Any
from agents.llm_client import LMStudioClient


class JDAnalystAgent:
    """Analyzes JD and extracts structured information."""

    def __init__(self):
        self.llm = LMStudioClient()

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Robustly extract JSON from LLM response."""
        if not response or not isinstance(response, str):
            return {
                "error": "Empty or invalid LLM response",
                "raw": str(response)
            }

        # Remove reasoning tags
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Remove markdown fences
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Find JSON block
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return {
                "error": "No JSON object found in LLM response",
                "raw": response
            }

        json_text = cleaned[json_start:json_end + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw": response
            }

    def analyze(self, jd_text: str) -> Dict[str, Any]:
        """
        Analyze Job Description and extract structured data.
        Returns compact JSON with: tech_stack, experience_required, key_challenges, soft_skills
        """
        # Truncate to safe limit for Gemma 2B
        jd_snippet = jd_text[:1500]

        prompt = f"""Analyze the following Job Description and extract structured information.

Return ONLY valid JSON in this exact format:
{{
  "tech_stack": ["tech1", "tech2", "tech3"],
  "experience_required": "years/level description",
  "key_challenges": ["challenge1", "challenge2"],
  "soft_skills": ["skill1", "skill2"]
}}

Job Description:
{jd_snippet}
"""

        messages = [
            {
                "role": "system",
                "content": "You are a JSON-only API. Output ONLY valid JSON. No prose, no markdown, no explanation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print(f"[JDAnalystAgent] Payload size: {sum(len(str(m.get('content', ''))) for m in messages)} chars")

        try:
            response = self.llm.chat(messages)
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

            # Ensure all expected keys exist
            return {
                "tech_stack": parsed.get("tech_stack", []),
                "experience_required": parsed.get("experience_required", "N/A"),
                "key_challenges": parsed.get("key_challenges", []),
                "soft_skills": parsed.get("soft_skills", [])
            }

        except Exception as e:
            return {
                "tech_stack": [],
                "experience_required": "N/A",
                "key_challenges": [],
                "soft_skills": [],
                "error": f"JD Analysis failed: {str(e)}"
            }
