from typing import Any, Dict, List, Literal, TypedDict
from langchain_core.messages import BaseMessage

RouteName = Literal["job_application", "devops", "personal", "general"]

class AssistantState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: RouteName
    response_text: str
    metadata: Dict[str, Any]
    