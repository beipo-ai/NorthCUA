"""A projected local tool used only by runtime extension regression tests."""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from northcua.agentkernel.tools.projection import contract_to_nexau_tool
from northcua_runtime_shared.tool_contracts import ToolContract, ToolPlacement


class LookupContextParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)


def lookup_context_tool(lookup: Callable[[str], str]):
    contract = ToolContract(
        tool_id="lookup_context",
        namespace="fixture",
        agent_kind="desktop",
        description="Read a context value from an application-owned local extension.",
        params_model=LookupContextParams,
        placement=ToolPlacement.LOCAL_HANDLER,
        side_effect=False,
    )

    def lookup_context(question: str) -> str:
        if not question.strip():
            raise ValueError("lookup_context requires a nonempty question")
        return lookup(question)

    return contract_to_nexau_tool(contract, implementation=lookup_context, execution="sync", disable_parallel=True)
