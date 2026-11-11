from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from tools import AgentPost

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

class DecentralizedDebateState(TypedDict):
    # Simulation data
    player_id: str
    units: list[str]
    regions: list[str]
    upkeep_phase: bool
    supply_surplus: int
    recent_events: list[str]
    init_msg: str
    input_prompt: Optional[BaseMessage]
    
    # Unit & region assignment
    regions_units: dict[str, list[str]]
    units_orders: dict[str, Order]
    region_recruitment: dict[str, RecruitmentChoice]
    
    # Message logs, debate board & blocked tools
    federal_agent_messages: Annotated[Sequence[BaseMessage], add_messages]
    regional_agents_messages: dict[str, list[BaseMessage]]
    shared_conversation: list[AgentPost]
    blocked_tools: list[str]
    
    # Voting & proposals
    recruitment_proposal: Optional[dict[str, RecruitmentChoice]]
    agents_votes: dict[str, bool]
    
    # Turn queue & quiescence
    active_agents: list[str]
    finished_agents: list[str]
    current_turn: int
    current_voting_turn: int
    turn_limit: int
    voting_turn_limit: int
    last_cycle_activity: int
    next_action: Literal["federal_agent", "regional_agent", "end"]
    
    # Subgraph channel for active regional agent
    current_agent_id: Optional[str]
    vote: Optional[bool]
    controlled_units: Optional[dict[str, Order]]
    regional_messages: Optional[Annotated[Sequence[BaseMessage], add_messages]]

class RegionalState(TypedDict):  
    current_agent_id: str
    vote: bool
    controlled_units: dict[str, Order]
    regional_messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Channel for additional information
    player_id: str
    blocked_tools: list[str]
    regions_units: dict[str, list[str]]
    shared_conversation: list[AgentPost]