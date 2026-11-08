from langchain_core.tools import tool, InjectedToolCallId
from langchain.tools import InjectedState
from langgraph.types import Command
from langchain.messages import ToolMessage
from typing import Annotated

from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import describe_order

# General-Agent Tools
@tool
def assign_order(
    unit_id: str,
    unit_order: Order,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Assign chosen order to the specified unit."""

    if unit_id not in state["controlled_units"].keys():
        return f"Error: Unit {unit_id} does not exist."    

    if state["controlled_units"][unit_id] is not None:
        return f"Error: Unit {unit_id} has been already assigned an order."

    return Command(
        update={
            "controlled_units": {
                unit_id: unit_order
            },
            "general_messages": [
                ToolMessage(
                    content=f"Order to {unit_order.order_type} was properly assigned to unit {unit_id}.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool
def remove_order(
    unit_id: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Remove already assigned order from specified unit."""
    if unit_id not in state["controlled_units"].keys():
        return f"Error: Unit {unit_id} does not exist."    
    
    if state["controlled_units"][unit_id] is None:
        return f"Error: Unit {unit_id} has not been assigned any orders yet."
        
    return Command(
        update={
            "controlled_units": {
                unit_id: None
            },
            "general_messages": [
                ToolMessage(
                    content=f"The order was properly removed from unit {unit_id}.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

# Leader-Agent Tools
@tool
def recruit_unit(
    unit_type: str,
    region: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Recruit a unit of given type in the specified region."""
    choice = RecruitmentChoice(unit_type=unit_type, region=region)
    current_choices = state.get("recruitment_choice", [])

    return Command(
        update={
            "recruitment_choice": current_choices + [choice],
            "leader_messages": [
                ToolMessage(
                    content=f"The recruitment order for {unit_type} in {region} region, was properly registered.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool
def delegate_general(
    general_id: str,
    command: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Delegate tactical movement and attack orders to a General Agent."""
    
    # Verifying that the general exists
    if general_id not in state["generals_messages"]:
        return f"Error: General '{general_id}' does not exist. Available generals: {list(state['generals_units'].keys())}" 

    # Initializing dict of controlled units for the general
    controlled_units = {unit: None for unit in state["generals_units"][general_id]}

    # Updating state
    return Command(
        update={
            "active_general": general_id,
            "general_id": general_id,
            "received_orders": command,
            "controlled_units": controlled_units,
            "general_messages": state["generals_messages"].get(general_id),
            "leader_messages": [
                ToolMessage(
                    content=f"The task was dispatched to General {general_id}.",
                    name="delegate_general",
                    tool_call_id=tool_call_id,
                    id=f"tool_{tool_call_id}"
                )
            ]
        }
    )

@tool
def get_order_list(state: Annotated[dict, InjectedState]) -> str:
    """Read the current list of orders given to the units.

    Returns:
        The list of orders assigned this round.
    """
    order_list = state["order_list"]

    # Formatting the tool output to be more readable for the LLM
    if len(order_list) > 0:
        observation = f"Currently assigned orders:\r\n"
        for order in order_list:
            observation += f"- {describe_order(order)};"
        observation += "."
    else:
        observation = "There are no orders yet, assigned to your units."

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