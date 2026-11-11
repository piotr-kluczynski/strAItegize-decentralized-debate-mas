from typing import Literal
from functools import partial
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage

from federal_agent import get_federal_agent
from regional_agent import get_regional_agent
from prompts import conversation_prompt, federal_agent_voting_prompt, regional_agent_voting_prompt, voting_result_prompt
from states import DecentralizedDebateState, RegionalState
from tools import post_message, assign_order, set_recruitment, set_vote, get_order_list, get_recent_events

from strAItegize_common.client.tools import get_generic_tools
from strAItegize_common.communication.message_types import MessageType
from strAItegize_common.client.communication.communication_context import CommunicationContext

def phase_node(state: DecentralizedDebateState) -> DecentralizedDebateState:
    # Adding the round message to the agents
    input_prompt = state.get("input_prompt")
    federal_agent_messages = state["federal_agent_messages"]
    regional_agents_messages = state["regional_agents_messages"]
    
    if input_prompt:
        federal_agent_messages = federal_agent_messages + [input_prompt]
        
        for agent_id in regional_agents_messages.keys():
            regional_agents_messages[agent_id] = regional_agents_messages[agent_id] + [input_prompt]
            
    # Blocking selected tools
    if state["upkeep_phase"]:
        blocked_tools = ["assign_order"]
    else:
        blocked_tools = ["set_recruitment", "set_vote"]
    
    return {
        "current_turn": 0,
        "current_voting_turn": 0,
        "federal_agent_messages": federal_agent_messages,
        "active_agents": list(state["regions"]),
        "finished_agents": [],
        "regional_agents_messages": regional_agents_messages,
        "blocked_tools": blocked_tools,
        "units_orders": {} if not state["upkeep_phase"] else state.get("units_orders", {}),
        "region_recruitment": {} if state["upkeep_phase"] else state.get("region_recruitment", {})
    }
    
def agent_manage_node(state: DecentralizedDebateState, communication: CommunicationContext) -> DecentralizedDebateState:
    regions = list(state["regions"])
    current_units = set(state.get("units", []))
    regions_units = {region: list(units) for region, units in state.get("regions_units", {}).items()}
    regional_agent_messages = dict(state.get("regional_agents_messages", {}))
    
    # Create new regional agents
    for region in regions:
        if region not in regions_units:
            regions_units[region] = []
        
        if region not in regional_agent_messages:
            regional_agent_messages[region] = [state["init_msg"]]
    
    # Collect orphaned units
    orphaned_units = []
    for region in list(regions_units.keys()):
        if region not in regions:
            orphaned_units.extend(regions_units.pop(region))
            regional_agent_messages.pop(region, None)
    
    # Clean up dead units
    for region in regions:
        regions_units[region] = [unit for unit in regions_units[region] if unit in current_units]
    
    # Assign orphaned and new units 
    all_assigned = {unit for unit_list in regions_units.values() for unit in unit_list}
    unassigned = [unit for unit in current_units if unit not in all_assigned] + [unit for unit in orphaned_units if unit in current_units]
    
    for unit_id in unassigned:
        best_region = _get_closest_region(unit_id, regions, communication)
        regions_units[best_region].append(unit_id)
        
    return {
        "regions_units": regions_units,
        "regional_agents_messages": regional_agent_messages
    }

def dispatcher_node(state: DecentralizedDebateState) -> dict:
    upkeep = state.get("upkeep_phase", False)
    current_voting_turn = state.get("current_voting_turn", 0)
    voting_turn_limit = state.get("voting_turn_limit", 3)
    
    # 1. UPKEEP: Check if Federal Agent needs to propose 
    if upkeep and not state.get("recruitment_proposal"):
        # Check if we exceeded total voting revisions
        if current_voting_turn >= voting_turn_limit:
            return {
                "region_recruitment": {},
                "next_action": "end"
            }
        
        # Prepare Federal Agent turn with shared conversation and voting prompt
        federal_messages = list(state.get("federal_agent_messages", [])) + [
            SystemMessage(content=conversation_prompt(state.get("shared_conversation", []))),
            SystemMessage(content=federal_agent_voting_prompt(
                current_turn=current_voting_turn + 1,
                turn_limit=voting_turn_limit
            ))
        ]
        
        return {
            "current_voting_turn": current_voting_turn + 1,
            "federal_agent_messages": federal_messages,
            "next_action": "federal_agent"
        }

    # 2. Check Queue of Regional Agents
    active = list(state.get("active_agents", []))
    finished = list(state.get("finished_agents", []))
    current_turn = state.get("current_turn", 0)
    turn_limit = state.get("turn_limit", 3)
    last_cycle_activity = state.get("last_cycle_activity", -1)
    
    # If all regional agents finished their turns in this cycle:
    if not active:
        
        # A. In Upkeep: Finish voting turn if debate ended
        if upkeep and state.get("recruitment_proposal"):
            votes = state.get("agents_votes", {})
            total_regions = len(state.get("regions", []))
            all_voted = len(votes) >= total_regions
            
            activity_now = (
                len(state.get("shared_conversation", [])) +
                len(state.get("units_orders", {})) +
                len(state.get("agents_votes", {}))
            )
            is_quiescent = (current_turn > 0 and activity_now == last_cycle_activity)
            turn_limit_reached = (current_turn >= turn_limit)
            
            # If all votes have been cast or debate timed out, decide about the outcome
            if all_voted or is_quiescent or turn_limit_reached:
                yes_votes = sum(1 for v in votes.values() if v is True)
                  
                # Passed
                if yes_votes > (total_regions / 2):
                    return {
                        "region_recruitment": state["recruitment_proposal"],
                        "next_action": "end"
                    }
                    
                # Rejected
                # Checking voting turn limit
                next_voting_turn = current_voting_turn + 1
                if next_voting_turn >= voting_turn_limit:
                    return {
                        "region_recruitment": {},
                        "next_action": "end"
                    }
                    
                result_msg = SystemMessage(content=voting_result_prompt(
                    outcome=False,
                    current_turn=current_voting_turn,
                    turn_limit=voting_turn_limit
                ))
                federal_messages = state.get("federal_agent_messages", []) + [
                    result_msg,
                    SystemMessage(content=conversation_prompt(state.get("shared_conversation", []))),
                    SystemMessage(content=federal_agent_voting_prompt(
                        current_turn=next_voting_turn,
                        turn_limit=voting_turn_limit
                    ))
                ]
                
                return {
                    "recruitment_proposal": None,
                    "agents_votes": {},
                    "current_turn": 0,
                    "active_agents": [],
                    "finished_agents": [],
                    "last_cycle_activity": -1,
                    "current_voting_turn": next_voting_turn,
                    "federal_agent_messages": federal_messages,
                    "next_action": "federal_agent"
                }
                    
        # B. Check Quiescence & Turn Limit
        activity_now = (
            len(state.get("shared_conversation", [])) +
            len(state.get("units_orders", {})) +
            len(state.get("agents_votes", {}))
        )
        if current_turn >= turn_limit or (current_turn > 0 and activity_now == last_cycle_activity):
            return {"next_action": "end"}
        
        # D. Refill queue for next turn
        active = list(state.get("regions", []))
        finished = []
        current_turn += 1
        last_cycle_activity = activity_now

    # 3. Dispatch the Next Regional Agent
    agent_id = active.pop(0)
    history = list(state.get("regional_agents_messages", {}).get(agent_id, []))
    
    # Shared conversation board
    prompt_list = [SystemMessage(content=conversation_prompt(state.get("shared_conversation", [])))]
    
    # In Upkeep: add voting prompt containing current proposal and vote tally
    if upkeep and state.get("recruitment_proposal"):
        prompt_list.append(SystemMessage(content=regional_agent_voting_prompt(
            recruitment_proposal=state["recruitment_proposal"],
            votes=state.get("agents_votes", {}),
            current_turn=current_voting_turn,
            turn_limit=voting_turn_limit
        )))
        
    # Set up staging channels for RegionalState
    units = state.get("regions_units", {}).get(agent_id, [])
    controlled_units = {unit: state.get("units_orders", {}).get(unit) for unit in units}
    
    return {
        "active_agents": active,
        "finished_agents": finished,
        "last_cycle_activity": last_cycle_activity,
        "current_turn": current_turn,
        "current_agent_id": agent_id,
        "controlled_units": controlled_units,
        "vote": state.get("agents_votes", {}).get(agent_id, None),
        "regional_messages": history + prompt_list,
        "next_action": "regional_agent"
    }

def agent_iteration_router(state: DecentralizedDebateState) -> Literal["federal_agent", "regional_agent", "end"]:
    return state["next_action"]

def regional_agent_tool_router(state: RegionalState) -> Literal["tools", "end"]:
    last_msg = state["regional_messages"][-1]
    if last_msg.tool_calls:
        return "tools"    
    return "end" 

def federal_agent_tool_router(state: DecentralizedDebateState) -> Literal["tools", "end"]:
    last_msg = state["federal_agent_messages"][-1]
    if last_msg.tool_calls:
        return "tools"    
    return "end"

def agent_output_node(state: DecentralizedDebateState) -> DecentralizedDebateState:
    finished = state["finished_agents"]
    
    agent_id = state["current_agent_id"]
    units_orders = state["units_orders"]
    agents_votes = state["agents_votes"]
    regional_agents_messages = state["regional_agents_messages"]
    
    # Update agent's message history
    regional_agents_messages[agent_id] = state["regional_messages"]
    
    # Update agent's orders
    for unit, order in state["controlled_units"].items():
        units_orders[unit] = order
        
    # Update agent vote
    agents_votes[agent_id] = state["vote"]
    
    # Changing agent's status
    finished.append(agent_id)
    
    # Cleaning shared channel and updating the state
    return {
        "current_agent_id": None,
        "vote": None,
        "controlled_units": None,
        "regional_messages": None,
        "finished_agents": finished,
        "regional_agents_messages": regional_agents_messages,
        "units_orders": units_orders,
        "agents_votes": agents_votes
    }

def get_decentralized_debate_mas(
    communication: CommunicationContext,
    federal_agent_model = init_chat_model("openai:gpt-4o"),
    regional_agent_model = init_chat_model("openai:gpt-4o")
):
    # Assigning tools
    federal_agent_tools = [set_recruitment, get_order_list, get_recent_events] + get_generic_tools(communication=communication)
    regional_agent_tools = [post_message, assign_order, set_vote, get_order_list, get_recent_events] + get_generic_tools(communication=communication)
    
    # Building regional agent subgraph
    regional_agent_builder = StateGraph(RegionalState)
    
    regional_agent_builder.add_node("regional_agent", get_regional_agent(tools=regional_agent_tools, model=regional_agent_model))
    regional_agent_builder.add_node("tools", ToolNode(regional_agent_tools, messages_key="regional_messages"))
    
    regional_agent_builder.add_edge(START, "regional_agent")
    regional_agent_builder.add_conditional_edges("regional_agent", regional_agent_tool_router, {
        "tools": "tools",
        "end": END
    })
    regional_agent_builder.add_edge("tools", "regional_agent")
    
    compiled_regional_agent = regional_agent_builder.compile()  
    
    # Building Decentralized Debate Multi-Agent System graph
    mas_builder = StateGraph(DecentralizedDebateState)
    
    mas_builder.add_node("phase_node", phase_node)
    mas_builder.add_node("agent_manage_node", partial(agent_manage_node, communication=communication))
    mas_builder.add_node("dispatcher_node", dispatcher_node)
    mas_builder.add_node("federal_agent", get_federal_agent(tools=federal_agent_tools, model=federal_agent_model))
    mas_builder.add_node("federal_tools", ToolNode(tools=federal_agent_tools, messages_key="federal_agent_messages"))
    mas_builder.add_node("regional_agent", compiled_regional_agent)
    mas_builder.add_node("agent_output_node", agent_output_node)
    
    mas_builder.add_edge(START, "agent_manage_node")
    mas_builder.add_edge("agent_manage_node", "phase_node")
    mas_builder.add_edge("phase_node", "dispatcher_node")
    mas_builder.add_conditional_edges("dispatcher_node", agent_iteration_router, {
        "federal_agent": "federal_agent",
        "regional_agent": "regional_agent",
        "end": END
    })
    mas_builder.add_conditional_edges("federal_agent", federal_agent_tool_router, {
        "tools": "federal_tools",
        "end": "dispatcher_node"
    })
    mas_builder.add_edge("federal_tools", "federal_agent")
    mas_builder.add_edge("regional_agent", "agent_output_node")
    mas_builder.add_edge("agent_output_node", "dispatcher_node")
    
    checkpointer = MemorySaver()
    return mas_builder.compile(checkpointer=checkpointer)

def _get_closest_region(unit_id: str, regions: list[str], communication) -> str:
    # Get unit position
    unit_data = communication.request_data(
        request_type=MessageType.GetUnitDetails,
        request_args={"unit_id": unit_id}
    )
    
    if not unit_data or not hasattr(unit_data, "coord"):
        return regions[0]
    
    unit_coord_split = unit_data.coord.split(",")
    unit_q = int(unit_coord_split[0])
    unit_r = int(unit_coord_split[1])
    unit_s = int(unit_coord_split[2])
    
    closest_region = regions[0]
    min_dist = float("inf")
    
    # Find closest distance
    for region_id in regions:
        reg_data = communication.request_data(
            request_type=MessageType.GetRegionDetails,
            request_args={"region_id": region_id}
        )
        
        if reg_data and hasattr(reg_data, "command_center_coord"):
            region_coord_split = reg_data.command_center_coord.split(",")
            reg_q = int(region_coord_split[0])
            reg_r = int(region_coord_split[1])
            reg_s = int(region_coord_split[2])
            
            dist = (abs(unit_q-reg_q) + abs(unit_r-reg_r) + abs(unit_s-reg_s)) // 2
            
            if dist < min_dist:
                min_dist = dist
                closest_region = region_id
                
    return closest_region