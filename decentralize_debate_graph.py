from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage

from federal_agent import get_federal_agent
from regional_agent import get_regional_agent
from states import DecentralizedDebateState, RegionalState
from tools import post_message, assign_order, set_recruitment, set_vote, get_order_list, get_recent_events

from strAItegize_common.client.tools import get_generic_tools
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
        "active_agents": [list(state["regions"])],
        "finished_agents": [],
        "regional_agents_messages": regional_agents_messages,
        "blocked_tools": blocked_tools,
        "unit_orders": {} if not state["upkeep_phase"] else state.get("unit_orders", {}),
        "region_recruitment": {} if state["upkeep_phase"] else state.get("region_recruitment", {})
    }
    
# TO DO
def agent_manage_node(state: DecentralizedDebateState) -> DecentralizedDebateState:
    agents_messages = state["agents_messages"]
    agent_types = state["agent_types"]
    
    # Removing missing agents
    for agent in list(agents_messages.keys()):
        # Reassigning leftover units
        
        
        if agent not in list(state["units"]) + list(state["regions"]):
            del agents_messages[agent]
            del agent_types[agent]
            
    # Assigning new units

    # Adding new agents
    for agent in list(state["regions"]):
        if agent not in agents_messages.keys():
            agent_types[agent] = "region"
            agents_messages[agent] = [state["init_msg"]]

    return {
        "agents_messages": agents_messages,
        "agent_types": agent_types
    }
    
def phase_router(state: DecentralizedDebateState) -> Literal["planning_node", "voting_node"]:
    if state["upkeep_phase"]:
        return "voting_node"
    
    return "planning_node"

def voting_node(state: DecentralizedDebateState) -> DecentralizedDebateState:
    current_voting_turn = state["current_voting_turn"]
    recruitment_proposal = state["recruitment_proposal"]
    
    # If no proposal was made yet, the federal agent is called
    if not recruitment_proposal:
        current_voting_turn += 1
        
        return {
            "current_voting_turn": current_voting_turn,
            "next_action": "federal_agent"
        }
        
    # Start voting process
    current_turn = state["current_turn"]
    turn_limit = state["turn_limit"]
    agent_activity = state["agent_activity"]
    
    active = state["active_agents"]
    finished = state["finished_agents"]   
    
    # If there are no more active agents check turn condition
    if len(active) == 0:
        # Ending if the agents took too many turns
        if current_turn >= turn_limit:
            return {
                "active_agent": [],
                "finished_agents": finished,
                "current_turn": current_turn,
                "next_action": "end"
            }
    
        else:
            # Ending if there was no activity
            if not agent_activity:
                return {
                    "active_agent": [],
                    "finished_agents": finished,
                    "current_turn": current_turn,
                    "next_action": "end"
                }
    
            # Starting new turn
            current_turn += 1
            active = finished.reverse()
            agent_activity = False
            finished = []
    
    # Activate next agent
    current_agent = active.pop(0)
    
    return {
        "active_agent": current_agent,
        "active_agents": active,
        "finished_agents": finished,
        "current_turn": current_turn,
        "agent_activity": agent_activity,
        "next_action": "regional_agent",
        "regional_id": current_agent,
        "controlled_units": state["regions_units"][current_agent],
        "regional_messages": state["regional_agents_messages"][current_agent]
    }

def planning_node(state: DecentralizedDebateState) -> DecentralizedDebateState:
    current_turn = state["current_turn"]
    turn_limit = state["turn_limit"]
    agent_activity = state["agent_activity"]
    
    active = state["active_agents"]
    finished = state["finished_agents"]
    
    # If there are no more active agents check turn condition
    if len(active) == 0:
        # Ending if the agents took too many turns
        if current_turn >= turn_limit:
            return {
                "active_agent": [],
                "finished_agents": finished,
                "current_turn": current_turn,
                "next_action": "end"
            }
        
        else:
            # Ending if there was no activity
            if not agent_activity:
                return {
                    "active_agent": [],
                    "finished_agents": finished,
                    "current_turn": current_turn,
                    "next_action": "end"
                }
            
            # Starting new turn
            current_turn += 1
            active = finished.reverse()
            agent_activity = False
            finished = []
            
    # Activate next agent
    current_agent = active.pop(0)
    
    return {
        "active_agent": current_agent,
        "active_agents": active,
        "finished_agents": finished,
        "current_turn": current_turn,
        "agent_activity": agent_activity,
        "next_action": "regional_agent",
        "regional_id": current_agent,
        "controlled_units": state["regions_units"][current_agent],
        "regional_messages": state["regional_agents_messages"][current_agent]
    }
        
    

def agent_iteration_router(state: DecentralizedDebateState) -> Literal["federal_agent", "regional_agent", "agent_output_node"]:
    return state["next_action"]

def regional_agent_tool_router(state: RegionalState) -> Literal["tools", "end"]:
    last_msg = state["agent_messages"][-1]
    if last_msg.tool_calls:
        tools_names = [tool_call.name for tool_call in last_msg.tool_calls]
        
        # Checking for turn-ending tools
        if "post_message" not in tools_names:
            return "tools"
    
    return "end"

def agent_output_node(state: DecentralizedDebateState) -> DecentralizedDebateState:
    finished = state["finished_agents"]
    
    agent_id = state["regional_id"]
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
        "regional_id": None,
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
    federal_agent_tools = [get_order_list, get_recent_events] + get_generic_tools(communication=communication)
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
    
    # ...
    
    checkpointer = MemorySaver()
    return mas_builder.compile(checkpointer=checkpointer)