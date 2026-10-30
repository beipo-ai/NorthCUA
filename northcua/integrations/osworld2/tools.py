"""Bind the task owner's user-simulation callback without packaging its backend."""

from collections.abc import Callable

from nexau.archs.tool import Tool
from pydantic import BaseModel, ConfigDict, Field

from northcua.agentkernel.tools.projection import contract_to_nexau_tool
from northcua_runtime_shared.tool_contracts import ToolContract, ToolPlacement


class AskUserParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)


ASK_USER_CONTRACT = ToolContract(
    tool_id="ask_user",
    namespace="osworld2",
    agent_kind="desktop",
    description="Ask the user one precise question when essential task information is missing.",
    params_model=AskUserParams,
    placement=ToolPlacement.LOCAL_HANDLER,
    side_effect=True,
)


def ask_user_tool(answer: Callable[[str], str]) -> Tool:
    """Use the original task's responder; never invent a substitute answer."""

    def ask_user(question: str) -> str:
        if not question.strip():
            raise ValueError("ask_user requires a nonempty question")
        return answer(question)

    return contract_to_nexau_tool(
        ASK_USER_CONTRACT,
        implementation=ask_user,
        execution="sync",
        disable_parallel=True,
    )
