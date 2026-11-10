from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from tools import AgentPost

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order

class DecentralizedDebateState(TypedDict):
    units: List[str]
    regions: List[str]
    recent_events: List[str]
    
    input_prompt: Optional[BaseMessage]
    upkeep_phase: bool = False
    supply_surplus: int = 0
    
    player_id: str
    init_msg: str
    blocked_tools: list[str]
    
    regions_units: dict[str, list[str]]
    units_orders: dict[str, Order]
    region_recruitment: dict[str, RecruitmentChoice]
    
    federal_agent_messages: Annotated[Sequence[BaseMessage], add_messages]
    regional_agents_messages: dict[str, list[BaseMessage]]
    shared_conversation: list[AgentPost]

    # Navigation variables
    next_action: Literal["federal_agent", "regional_agent", "agent_output_node"]
    agent_activity: bool
    current_voting_turn: int
    voting_turn_limit: int
    current_turn: int
    turn_limit: int
    active_agent: str

    active_agents: list[str]
    finished_agents: list[str]
    
    # Voting behaviour
    agents_votes: dict[str, bool]
    recruitment_proposal: dict[str, RecruitmentChoice]

    # Shared channels with RegionalState
    regional_id: Optional[str]
    vote: Optional[bool]
    controlled_units: Optional[dict[str, Order]]
    regional_messages: Optional[Annotated[Sequence[BaseMessage], add_messages]]

class RegionalState(TypedDict):
    player_id: str
    agent_activity: bool
    regions_units: dict[str, list[str]]
    shared_conversation: list[AgentPost]
    
    regional_id: str
    vote: bool
    controlled_units: dict[str, Order]
    regional_messages: Annotated[Sequence[BaseMessage], add_messages]