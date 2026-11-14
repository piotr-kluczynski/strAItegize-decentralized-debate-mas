from typing import Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.tools import tool, InjectedToolCallId
from langchain.tools import InjectedState
from langgraph.types import Command
from langchain.messages import ToolMessage
from typing import Annotated

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order, describe_order

class AgentPost(BaseModel):
    author: str = Field(
        description="Author agent of the posted message."
    )
    content: str = Field(
        description="Text message posted in the message."
    )

@tool
def post_message(
    content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]    
) -> Command:
    """Post message on the shared conversation with other agents."""
    
    shared_conversation = state["shared_conversation"]
    
    agent_post = AgentPost(
        author=state["current_agent_id"],
        content=content
    )

    shared_conversation.append(agent_post)

    return Command(
        update={
            "shared_conversation": shared_conversation,
            "regional_messages": [
                ToolMessage(
                    content=f"Your message: {content} was posted on the shared conversation.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

# Federal Agent Tools
@tool 
def set_recruitment(
    unit_type: Literal["light_infantry", "heavy_infantry", "cavalry", "none"],
    region: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Registering the recruitment choice to propose for the regional agents."""
    recruitment_proposal = state.get("recruitment_proposal") or {}
    
    # Verifying the ownership over the region
    if region not in state["regions"]:
        return f"Error: The country does not control the {region} region. List of controlled regions: {state["regions"]}."

    # Serving removal recrutment orders
    if unit_type == "none":
        recruitment_proposal[region] = None
        
        return Command(
            update={
                "recruitment_proposal": recruitment_proposal,
                "federal_agent_messages": [
                    ToolMessage(
                        content=f"Properly removed the recruitment choice for {region} region from the recruitment proposal.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Verifying that current recruitments does not exceed the supply surplus
    # Need to remove the hard-coded part
    spent_supplies = 0
    for recruitment in recruitment_proposal.values():
        if not recruitment:
            continue
        
        if recruitment.unit_type == "light_infantry":
            spent_supplies += 1
        elif recruitment.unit_type == "heavy_infantry":
            spent_supplies += 2
        elif recruitment.unit_type == "cavalry":
            spent_supplies += 3
            
    new_cost = {"light_infantry": 1, "heavy_infantry": 2, "cavalry": 3}.get(unit_type, 0)    
    if state["supply_surplus"] + new_cost < spent_supplies:
        return f"Error: Cannot register new recruitment order, as it would exceed the total supply surplus: {state["supply_surplus"]}. Remaining supplies to spent: {state['supply_surplus'] - spent_supplies}."

    # Registering the recruitment
    recruitment_choice = RecruitmentChoice(
        unit_type=unit_type,
        region=region
    )

    recruitment_proposal[region] = recruitment_choice
    
    return Command(
        update={
            "recruitment_proposal": recruitment_proposal,
            "federal_agent_messages": [
                ToolMessage(
                    content=f"The recruitment choice to recruit {unit_type} in region {region}, was properly assigned to the recruitment proposal.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


# Regional Agent Tools
@tool
def set_vote(
    decision: bool,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Signal whether you agree with the recruitment choice proposed by the Federal Agent or not."""

    return Command(
        update={
            "vote": decision,
            "regional_messages": [
                ToolMessage(
                    content=f"Your vote: {decision}, has been submitted properly.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool
def assign_order(
    unit_id: str,
    order_type: Literal["move", "hold", "attack", "support"],
    target_id: Optional[str],
    move_q: Optional[int],
    move_r: Optional[int],
    move_s: Optional[int],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Assign chosen order to the specified unit."""
    # Verifying the unit ownership
    if unit_id not in state["controlled_units"]:
        return f"Error: You can only assign orders to the units controlled by you."
    
    # Verifying the movement order
    if order_type == "move":
        # Checking if movement coordinates are properly assigned
        if move_q is None or move_r is None or move_s is None:
            return f"Error: Movement orders require q, r and s coordinates to be provided."
        
        # Checking the q + r + s = 0
        if move_q + move_r + move_s != 0:
            return f"Error: Movement coordinates need to sum up to 0."
    
    # Verifying the attack/support orders
    elif order_type == "attack" or order_type == "support":
        # Checking if target id is properly assigned
        if not target_id:
            return f"Error: Target id needs to be provided in case of attack/support orders"

    order = Order(
        order_type=order_type,
        unit_id=unit_id,
        target_id=target_id,
        move_q=move_q,
        move_r=move_r,
        move_s=move_s
    )
    
    controlled_units = state["controlled_units"]
    controlled_units[unit_id] = order

    return Command(
        update={
            "controlled_units": controlled_units,
            "regional_messages": [
                ToolMessage(
                    content=f"The order {order_type} for the unit {unit_id} was properly assigned.",
                    tool_call_id=tool_call_id
                )
            ]
        },
    )

@tool
def get_order_list(state: Annotated[dict, InjectedState]) -> str:
    """Read the currently assigned orders of your units.

    Returns:
        The list of orders assigned to your units.
    """
    controlled_units = state.get("controlled_units") or {}

    # Formatting the tool output to be more readable for the LLM
    if len(controlled_units) > 0:
        observation = f"Currently assigned orders:\r\n"
        for order in controlled_units.values():
            observation += f"- {describe_order(order)};"
        observation += "."
    else:
        observation = "There are no orders assigned to your units."

    return observation

@tool
def get_recent_events(state: Annotated[dict, InjectedState]) -> str:
    """Read the list of all events that took place this round.

    Returns:
        The list of events from this round.    
    """
    events = state["recent_events"]

    # Formatting the tool output to be more readable for the LLM
    if len(events) > 0:
        observation = "Events that had occurred on this round:\r\n"
        for event in events:
            observation += f"- {event},"
        observation += "."
    else:
        observation = "There've been no events since the last round."

    return observation