"""
Orchestrator Agent
Coordinates the multi-agent workflow.
Routes requests and manages human-in-the-loop approvals.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Literal
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agents.llm_client import LMStudioClient
from agents.state import AssistantState, RouteName
from agents.jd_analyst_agent import JDAnalystAgent
from agents.skill_match_agent import SkillMatchAgent
from agents.document_specialist_agent import DocumentSpecialistAgent
from agents.storage_agent import StorageAgent

# Initialize agents
llm_client = LMStudioClient()
jd_analyst = JDAnalystAgent()
skill_matcher = SkillMatchAgent()
doc_specialist = DocumentSpecialistAgent()
storage = StorageAgent()

# Job keywords for routing
JOB_KEYWORDS = [
    "job", "resume", "cv", "cover letter", "application", "apply",
    "interview", "jd", "job description", "salary", "role"
]

DEVOPS_KEYWORDS = [
    "terraform", "kubernetes", "k8s", "azure", "aws", "github actions",
    "devops", "sre", "platform", "iac", "pipeline", "prometheus", "grafana"
]

PERSONAL_KEYWORDS = [
    "schedule", "task", "meeting", "email", "draft", "reminder",
    "plan my day", "todo", "calendar", "document"
]


# ============================================================================
# Helper Functions
# ============================================================================

def _message_to_text(content: Any) -> str:
    """Convert message content to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _to_openai_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain messages to OpenAI format."""
    converted = []
    for msg in messages:
        if msg.type == "human":
            role = "user"
        elif msg.type == "ai":
            role = "assistant"
        else:
            role = "system"
        converted.append({"role": role, "content": _message_to_text(msg.content)})
    return converted


def _get_last_user_message(messages: List[BaseMessage]) -> str:
    """Get the most recent user message."""
    for msg in reversed(messages):
        if msg.type == "human":
            return _message_to_text(msg.content)
    return ""


def _detect_route(text: str) -> RouteName:
    """Lightweight keyword-based router."""
    lower_text = text.lower()
    if any(keyword in lower_text for keyword in JOB_KEYWORDS):
        return "job_application"
    if any(keyword in lower_text for keyword in DEVOPS_KEYWORDS):
        return "devops"
    if any(keyword in lower_text for keyword in PERSONAL_KEYWORDS):
        return "personal"
    return "general"


def _system_prompt(route: RouteName) -> str:
    """Get system prompt for a route."""
    if route == "job_application":
        return (
            "You are a job-application assistant for Tejbahadur Singh. "
            "Help with job search, resume tailoring, cover letters, application checklists, "
            "and interview preparation. Keep responses practical, concise, and useful."
        )
    if route == "devops":
        return (
            "You are a DevOps / Platform Engineering assistant for Tejbahadur Singh. "
            "Help with Terraform, Azure, GitHub Actions, Kubernetes, observability, and SRE tasks. "
            "Give practical implementation-oriented answers."
        )
    if route == "personal":
        return (
            "You are a personal productivity assistant for Tejbahadur Singh. "
            "Help draft emails, manage tasks, plan work, summarize documents, and prepare meetings. "
            "Be concise and action-oriented."
        )
    return (
        "You are a helpful general assistant for Tejbahadur Singh. "
        "Answer clearly, briefly, and accurately."
    )


def _generate_response(state: AssistantState, route: RouteName) -> AssistantState:
    """Generate a generic response for non-job routes."""
    messages = state.get("messages", [])
    system_prompt = _system_prompt(route)
    payload = [
        {"role": "system", "content": system_prompt},
        *_to_openai_messages(messages),
    ]
    response_text = llm_client.chat(payload)
    return {
        "route": route,
        "response_text": response_text,
        "messages": [AIMessage(content=response_text)],
        "metadata": {"routed_to": route}
    }


# ============================================================================
# Node: Classify
# ============================================================================

def classify_node(state: AssistantState) -> AssistantState:
    """Classify the user request determine which agent to route to."""
    messages = state.get("messages", [])
    last_user_text = _get_last_user_message(messages)
    route = _detect_route(last_user_text)
    return {
        "route": route,
        "metadata": {
            "last_user_preview": last_user_text[:120],
            "route": route
        }
    }


# ============================================================================
# Node: Job Application Orchestrator
# ============================================================================

def job_application_node(state: AssistantState) -> AssistantState:
    """
    Orchestrate the job application workflow.
    
    States:
    1. User pastes JD → Analyze JD + Match Skills → Show results + request UI action
    2. UI Button "Generate" → Document Specialist writes cover letter + resume → Save to disk
    3. UI Button "Approve" → Mark in SQLite as approved
    """
    messages = state.get("messages", [])
    last_text = _get_last_user_message(messages)

    # ==========================
    # STATE 1: User pastes JD
    # ==========================
    if "http" in last_text or len(last_text) > 300:
        print("[job_application_node] STATE 1: Analyzing JD...")

        # Run JD Analyst Agent
        jd_analysis = jd_analyst.analyze(last_text)
        print(f"JD Analysis: {jd_analysis}")

        # Run Skill Match Agent
        skill_match = skill_matcher.match(jd_analysis)
        print(f"Skill Match: {skill_match}")

        # Save temp context for next steps
        temp_data = {
            "jd_text": last_text,
            "jd_analysis": jd_analysis,
            "skill_match": skill_match
        }
        Path("data/temp_jd.json").write_text(json.dumps(temp_data), encoding="utf-8")

        score = skill_match.get("relevance_score", "N/A")
        matched = skill_match.get("matched_skills", [])
        missing = skill_match.get("missing_skills", [])
        challenges = jd_analysis.get("key_challenges", [])

        response_text = f"""### 📊 Job Analysis Complete

**Relevance Score:** {score}%

**Matched Skills:**
{chr(10).join(f"- {s}" for s in matched) if matched else "- None identified"}

**⚠️ Missing Skills:**
{chr(10).join(f"- {s}" for s in missing) if missing else "- None identified"}

**Key Challenges Identified:**
{chr(10).join(f"- {c}" for c in challenges) if challenges else "- None identified"}

---

I have analyzed the job posting. Ready to generate a tailored Cover Letter and Markdown Resume addressing these specific challenges?
"""

        return {
            "route": "job_application",
            "response_text": response_text,
            "messages": [AIMessage(content=response_text)],
            "metadata": {
                "routed_to": "job_application",
                "ui_action_required": "generate_docs",
                "score": score
            }
        }

    # ==========================
    # STATE 2: UI Button "Generate" clicked
    # ==========================
    if last_text == "UI_ACTION: GENERATE_DOCS":
        print("[job_application_node] STATE 2: Generating documents...")

        temp_file = Path("data/temp_jd.json")
        if not temp_file.exists():
            return {
                "route": "job_application",
                "response_text": "Session lost. Please paste the JD again.",
                "messages": [AIMessage(content="Session lost.")],
                "metadata": {"routed_to": "job_application"}
            }

        data = json.loads(temp_file.read_text(encoding="utf-8"))
        jd_text = data["jd_text"]
        jd_analysis = data["jd_analysis"]
        skill_match = data["skill_match"]

        # For now, extract company/role from JD text using LLM (or manual extraction)
        # Simple fallback: use LLM to extract company and role
        extraction_prompt = f"""Extract the company name and job role from this job description. Return as JSON only.
Format: {{"company": "Company Name", "role": "Job Title"}}

JD:
{jd_text[:500]}
"""
        try:
            extraction_response = llm_client.chat([
                {"role": "system", "content": "You are a JSON API. Output ONLY JSON."},
                {"role": "user", "content": extraction_prompt}
            ])
            extraction_data = json.loads(extraction_response)
            company = extraction_data.get("company", "Unknown_Company")
            role = extraction_data.get("role", "Unknown_Role")
        except:
            company = "Tech_Company"
            role = "Platform_Engineer"

        print(f"Extracted: {company} / {role}")

        # Run Specialist Agent
        cover_letter, tailored_resume = doc_specialist.generate_documents(
            company, role, jd_analysis, skill_match
        )

        print(f"Generated Cover Letter ({len(cover_letter)} chars)")
        print(f"Generated Resume ({len(tailored_resume)} chars)")

        # Run Storage Agent to save files and database entry
        save_result = storage.save_application(
            company=company,
            role=role,
            cover_letter=cover_letter,
            tailored_resume=tailored_resume,
            jd_text=jd_text,
            relevance_score=skill_match.get("relevance_score", 0),
            matched_skills=skill_match.get("matched_skills", []),
            missing_skills=skill_match.get("missing_skills", []),
            key_challenges=jd_analysis.get("key_challenges", [])
        )

        app_id = save_result["app_id"]
        folder_path = save_result["folder_path"]

        response_text = f"""### ✅ Documents Generated & Saved

I have successfully generated your tailored Cover Letter and Markdown Resume.

📂 **Folder:** `{folder_path}`
💾 **Database ID:** `{app_id}`
📄 **Files Created:**
- Cover_Letter.txt
- Tailored_Resume.md
- Job_Description.txt
- metadata.txt

Please open the folder and review the documents. Are they ready for submission?
"""

        return {
            "route": "job_application",
            "response_text": response_text,
            "messages": [AIMessage(content=response_text)],
            "metadata": {
                "routed_to": "job_application",
                "ui_action_required": "approve_submission",
                "app_id": app_id,
                "folder_path": folder_path
            }
        }

    # ==========================
    # STATE 3: UI Button "Approve" clicked
    # ==========================
    if last_text.startswith("UI_ACTION: APPROVE_SUBMISSION_"):
        print("[job_application_node] STATE 3: Approving submission...")

        app_id_str = last_text.replace("UI_ACTION: APPROVE_SUBMISSION_", "")
        try:
            app_id = int(app_id_str)
        except ValueError:
            return {
                "route": "job_application",
                "response_text": "Invalid application ID.",
                "messages": [AIMessage(content="Invalid ID.")],
                "metadata": {"routed_to": "job_application"}
            }

        storage.mark_approved(app_id)
        app_data = storage.get_application(app_id)

        response_text = f"""### 🎉 Application Approved!

Application **#{app_id}** for **{app_data['company_name']}** / **{app_data['role_title']}** is now marked as **"Approved & Ready"** in your database.

📂 Location: `{app_data['folder_path']}`
✅ Status: Approved & Ready
⏰ Approved At: {app_data['approved_at']}

Next steps:
1. You can now submit this application manually to the company.
2. Once submitted, I can help you set follow-up reminders (coming soon).

Would you like to process another job posting?
"""

        return {
            "route": "job_application",
            "response_text": response_text,
            "messages": [AIMessage(content=response_text)],
            "metadata": {"routed_to": "job_application", "app_id": app_id}
        }

    # ==========================
    # Fallback: General conversation
    # ==========================
    response_text = llm_client.chat([
        {"role": "system", "content": _system_prompt("job_application")},
        {"role": "user", "content": last_text}
    ])

    return {
        "route": "job_application",
        "response_text": response_text,
        "messages": [AIMessage(content=response_text)],
        "metadata": {"routed_to": "job_application"}
    }


# ============================================================================
# Nodes for other routes
# ============================================================================

def devops_node(state: AssistantState) -> AssistantState:
    return _generate_response(state, "devops")


def personal_node(state: AssistantState) -> AssistantState:
    return _generate_response(state, "personal")


def general_node(state: AssistantState) -> AssistantState:
    return _generate_response(state, "general")


# ============================================================================
# Route selector
# ============================================================================

def route_selector(state: AssistantState) -> str:
    """Decide which node to execute based on route."""
    return state.get("route", "general")


# ============================================================================
# Graph builder
# ============================================================================

def create_assistant_graph():
    """Build the LangGraph orchestrator."""
    workflow = StateGraph(AssistantState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("job_application_agent", job_application_node)
    workflow.add_node("devops_agent", devops_node)
    workflow.add_node("personal_agent", personal_node)
    workflow.add_node("general_agent", general_node)

    # Set entry point
    workflow.set_entry_point("classify")

    # Add conditional edges from classify
    workflow.add_conditional_edges(
        "classify",
        route_selector,
        {
            "job_application": "job_application_agent",
            "devops": "devops_agent",
            "personal": "personal_agent",
            "general": "general_agent",
        },
    )

    # All agents lead to END
    workflow.add_edge("job_application_agent", END)
    workflow.add_edge("devops_agent", END)
    workflow.add_edge("personal_agent", END)
    workflow.add_edge("general_agent", END)

    return workflow.compile()
