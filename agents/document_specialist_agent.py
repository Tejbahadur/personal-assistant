"""
Document Specialist Agent
Responsible ONLY for generating tailored documents.
Outputs: Cover Letter + Tailored Markdown Resume
Uses: Writing-optimized prompt with modern, punchy style.
"""

import re
from pathlib import Path
from typing import Dict, Any, Tuple
from agents.llm_client import LMStudioClient


class DocumentSpecialistAgent:
    """Generates tailored cover letters and resumes."""

    def __init__(self, resume_path: str = "data/resumes/tejbahadur-updated-resume.pdf"):
        self.llm = LMStudioClient()
        self.resume_path = Path(resume_path)

    def load_resume(self) -> str:
        """Load master resume."""
        if not self.resume_path.exists():
            return "No resume found."

        try:
            if self.resume_path.suffix.lower() == ".pdf":
                import PyPDF2
                text = ""
                with open(self.resume_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += (page.extract_text() or "") + "\n"
                return text if text.strip() else "Resume PDF empty."
            else:
                return self.resume_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading resume: {str(e)}"

    def generate_cover_letter(
        self,
        company: str,
        role: str,
        jd_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate a modern, punchy, problem-solver cover letter.
        Focuses on 1-2 key challenges from the JD.
        """
        resume_text = self.load_resume()
        resume_snippet = resume_text[:1500]  # Safe limit

        challenges = jd_analysis.get("key_challenges", [])
        focus_challenges = challenges[:2] if challenges else ["core responsibilities"]
        challenges_str = " and ".join(f'"{c}"' for c in focus_challenges)

        prompt = f"""Write a modern, punchy, problem-solver Cover Letter for Tejbahadur Singh.
Company: {company}
Role: {role}

STRICT RULES:
1. Start with a hook that addresses the company's technical challenges directly. NO generic openings.
2. Focus on these 1-2 specific challenges: {challenges_str}
3. Reference Tejbahadur's past experience that DIRECTLY SOLVES these challenges.
4. Include specific metrics or outcomes from the resume (e.g., "built systems handling X requests/sec", "reduced deployment time by Y%").
5. Keep it under 250 words. Be confident, pragmatic, action-oriented.
6. DO NOT just list technologies. Show impact and problem-solving capability.
7. Modern tone: professional but direct, no fluff.

Tejbahadur's Resume:
{resume_snippet}

Job Analysis Context:
Tech {', '.join(jd_analysis.get('tech_stack', []))}
Experience Level: {jd_analysis.get('experience_required', 'N/A')}
Key Challenges: {', '.join(focus_challenges)}
Soft Skills Needed: {', '.join(jd_analysis.get('soft_skills', []))}

IMPORTANT: Output ONLY the cover letter text. No markdown blocks, no explanations, no reasoning."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert professional writer. Generate compelling, authentic cover letters. Output ONLY the letter text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print(f"[DocumentSpecialistAgent.cover_letter] Payload size: {sum(len(str(m.get('content', ''))) for m in messages)} chars")

        try:
            response = self.llm.chat(messages)
            # Clean up any reasoning tags
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return cleaned if cleaned else "Failed to generate cover letter."
        except Exception as e:
            return f"Cover Letter Generation Error: {str(e)}"

    def generate_tailored_resume(
        self,
        jd_analysis: Dict[str, Any],
        skill_match: Dict[str, Any]
    ) -> str:
        """
        Generate a tailored Markdown resume.
        Reorders sections to emphasize relevant experience.
        Highlights how to bridge gaps in missing skills.
        """
        resume_text = self.load_resume()
        resume_snippet = resume_text[:2000]

        missing_skills = skill_match.get("missing_skills", [])
        missing_str = ", ".join(missing_skills) if missing_skills else "none"

        prompt = f"""Rewrite the resume in Markdown format, tailored to this specific job.

STRICT RULES:
1. Output ONLY valid Markdown.
2. Reorder sections so the most relevant experience appears first.
3. Reorder bullet points within sections to match job priorities.
4. For these missing skills ({missing_str}): Find and highlight transferable or adjacent skills from the resume. Do NOT invent or lie.
5. Enhance the professional summary to position Tejbahadur perfectly for THIS role.
6. Use consistent Markdown formatting: # for headings, ## for subsections, - for bullets.
7. Keep factual accuracy. Do not add false claims.

Job Requirements:
Tech Stack: {', '.join(jd_analysis.get('tech_stack', []))}
Experience: {jd_analysis.get('experience_required', 'N/A')}
Challenges: {', '.join(jd_analysis.get('key_challenges', []))}
Missing Skills: {missing_str}

Original Resume:
{resume_snippet}

IMPORTANT: Output ONLY the Markdown resume. No explanations."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert resume writer. Generate tailored, compelling Markdown resumes. Output ONLY the Markdown text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print(f"[DocumentSpecialistAgent.resume] Payload size: {sum(len(str(m.get('content', ''))) for m in messages)} chars")

        try:
            response = self.llm.chat(messages)
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return cleaned if cleaned else "Failed to generate tailored resume."
        except Exception as e:
            return f"Resume Tailoring Error: {str(e)}"

    def generate_documents(
        self,
        company: str,
        role: str,
        jd_analysis: Dict[str, Any],
        skill_match: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Generate both cover letter and tailored resume."""
        cover_letter = self.generate_cover_letter(company, role, jd_analysis)
        tailored_resume = self.generate_tailored_resume(jd_analysis, skill_match)
        return cover_letter, tailored_resume
