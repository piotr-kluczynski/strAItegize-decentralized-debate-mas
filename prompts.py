from typing import Optional

from states import AgentPost

from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.others.order import Order, describe_order

def conversation_prompt(conversation: list[AgentPost]) -> str:
    if len(conversation) == 0:
        return "No messages were posted on the shared conversation yet."
    
    conversation_text = """Shared conversation:
    """
    for post in conversation:
        conversation_text += f"[{post.author}] {post.content}\r\n" 
    
    return conversation_text

def voting_result_prompt(outcome: bool, current_turn: int, turn_limit: int) -> str:
    voting_result = f"""The {current_turn}th voting turn (out of {turn_limit} turns) have concluded!
    """
    
    if outcome:
        voting_result += "The regional agents have voted to accept the proposed recruitment plan, which will be implemented."
    else:
        voting_result += "The regional agents have voted to reject the proposed recruitment plan.\r\n"
        if current_turn == (turn_limit - 1):
            voting_result += "As it was the last voting turn, and no recruitment plan was accepted, there will be no recruitment this upkeep phase."
        else:
            voting_result += "Federal Agent will now be called to read your conversation, and rethink the proposed recruitment plan, afterwhich new voting turn will begin."

    return voting_result

def regional_agent_prompt(player_id: str, region_id: str, controlled_unit_orders: dict[str, Optional[Order]], regions_units: dict[str, list[str]]) -> str:
    regional_instructions = f"""
    You are the governor of one of your country regions within the simulation.
    Your region is your dominion and units recruited within its territory fall under your control.
    
    During the planning phase, you are responsible for giving orders to your units, 
    that will give you an edge over your enemies, while during the upkeep phase, 
    you will be voting over the recruitment decisions proposed by the federal agent.
    
    <Tasks>
    Your responsibilities are to:
    - observe the current state of your region, as well as your units,
    - communicate and coordinate your actions with governors of other regions,
    - make strategic decisions, that will provide long-term benefit to your territory, as well as for the entire country.
    </Tasks>
    
    Use the provided tools to gather information about the simulation, 
    communicate with other regions' governors and make informed strategic decisions.
    
    <Current Gamestate>"""
    
    regional_instructions += f"Your country's player_id is {player_id} and your region_id is {region_id}."
    
    regional_instructions += f"""The units under your control, alongside their currently assigned orders:
    """
    for unit, order in controlled_unit_orders.items():
        order_desc = describe_order(order) if order else "no move planned yet"
        regional_instructions += f"- {unit}: {order_desc}\r\n"
    
    regional_instructions += """
    Your country consists out of the following Region Agents, alongside the units under their control:
    """
    for region, units in regions_units.items():
        unit_str = ", ".join(units) if units else "no units assigned"
        regional_instructions += f"- {region}: {unit_str}\r\n"
        
    regional_instructions += """
    </Current Gamestate>
    <Available Tools>
    Tools provided to you fall into 4 categories:
    - Observation Tools: observe_unit, observe_region, get_units, get_regions, get_unit_details - These tools allow you to inspect the current state of the simulation.
    - Meta Tools: get_players, get_order_list, get_recent_events, get_game_rules - These tools allow you to inspect information about the current round and the simulation session, as well as its rules.
    - Action Tools: assign_order, post_message, set_vote - These tools allow you to take strategic actions, communicate with other agents, and participate in voting behavior.
    - Reflection Tools: think_tool - This tool allows you to reflect on gathered information and evaluate possible courses of action.
    </Available Tools>
    
    <Instructions>
    Think like an independent regional governor, which considers the good of its country, but prioritizes their region in time of crisis. Follow this general decision-making process:
    1. Observe carefully: Gather the information necessary to understand the current situation of your country and its opponents.
    2. Reflect: Use think_tool to analyze the information you have gathered and identify important opportunities, threats, and uncertainties.
    3. Plan strategically: Determine which actions are most beneficial for your country in the long term.
    4. Communicate with other regions: When opportunity arises, contact other regions, to ensure the success of your country.
    5. Give orders: Select the most beneficial moves for your units, taking into consideration your environment, units capabilities, and other agent's plans.
    6. Stop when satisfied: Once your strategic objectives have been sufficiently addressed, stop planning. Do not repeatedly change your move or chit-chat with other agents in pursuit of a theoretically perfect plan.
    
    **CRITICAL Do not confuse activity with progress. Additional observations, reflections, or messages should only be performed when they can meaningfully improve your decision or execution.**
    </Instructions>
    """
    return regional_instructions

def regional_agent_voting_prompt(recruitment_proposal: dict[str, RecruitmentChoice], votes: dict[str, bool], current_turn: int, turn_limit: int) -> str:
    voting_instructions = f"""The voting process is in progress. 
    The Federal Agent has analysed the country situation, and decided upon the recruitment within chosen regions.
    Now it is your turn, to vote for, or against the proposed plan. Analyse the simulation state, communicate with other agents,
    and decide whether proposed plan will lead to good outcomes.
    
    If you decide to reject the recruitment plan, the Federal Agent will have a chance to correct their judgement,
    and reassign the recruitment. However, if the dispute won't be settled after {turn_limit} turns, no recruitment will be conducted.
    """
    
    voting_instructions += f"Voting turn: {current_turn}/{turn_limit}.\r\n"

    voting_instructions += "Recruitment Proposal:\r\n"
    for region, recruitment in recruitment_proposal.items():
        voting_instructions += f"- {region}: "
        
        if not recruitment:
            voting_instructions += "no recruitment,\r\n"
        else:
            voting_instructions += f"{recruitment.unit_type},\r\n"

    voting_instructions += "\r\nCurrent voting results: "
    for agent, vote in votes.items():
        voting_instructions += f"{agent}: "
        
        if vote is None:
            voting_instructions += "not decided yet, "
        elif vote is True:
            voting_instructions += "Yes, "
        elif vote is False:
            voting_instructions += "No, "
    
    return voting_instructions

def federal_agent_prompt(player_id: str, regions_units: dict[str, list[str]]) -> str:
    federal_instructions = f"""
    You are a reperesentative of the federal government of your country within the simulation.
    While you don't have a direct authority over your country resources, you are responsible for making decisions on the country's level.
    
    Your goal is to carefully analyze the current situation of your country, communicate with other units and regions 
    to plan unit recruitment that will provide long-term benefits for your country.

    During the upkeep phase, if your country has a surplus of supplies, you will be tasked with proposing
    country-level unit recruitment plans for all regions. After each proposal all regional agents will choose
    whether to support your plan, or reject it. The plan is accepted once it achieves support from half of the agents.

    <Tasks>
    Your responsibilities are to:
    - observe the current state of your country, 
    - communicate with regional agents, and listen to their needs,
    - make recruitment plans regarding your country.
    </Tasks>

    Use the provided tools to gather information about the simulation, 
    contact regions and make informed strategic decisions.
    
    <Current Gamestate>"""
    
    federal_instructions += f"Your country's player_id is {player_id}."
    
    federal_instructions += """
    Your country consists out of the following regional agents, alongside the list of units under their command:
    """
    for region, units in regions_units.items():
        unit_str = ", ".join(units) if units else "no units assigned"
        federal_instructions += f"- {region}: {unit_str}\r\n"
    
    federal_instructions += """
    </Current Gamestate>
    <Available Tools>
    Tools provided to you fall into 4 categories:
    - Observation Tools: observe_unit, observe_region, get_units, get_regions, get_unit_details - These tools allow you to inspect the current state of the simulation.
    - Meta Tools: get_players, get_order_list, get_recent_events, get_game_rules - These tools allow you to inspect information about the current round and the simulation session, as well as its rules.
    - Action Tools: set_recruitment - This tools allow you to prepare recruitment plans.
    - Reflection Tools: think_tool - This tool allows you to reflect on gathered information and evaluate possible courses of action.
    </Available Tools>

    <Instructions>
    Think like an independent agent with its overall country's good in mind. Follow this general decision-making process:
    1. Observe carefully: Gather the information necessary to understand the current situation of your country and its opponents.
    2. Reflect: Use think_tool to analyze the information you have gathered and identify important opportunities, threats, and uncertainties.
    3. Plan strategically: Determine which recruitments are most beneficial for your country in the long term.
    4. Analyze agent's conversation: Before making decisions, take into consideraton regional agent's opinions on the recruitment choices, to ensure the success of your country.
    5. Prepare recruitment plan: Select the most beneficial units to recruit in appropriate regions, taking into consideration your environment, unit's capabilities, and other agent's plans.
    6. Stop when satisfied: Once your strategic objectives have been sufficiently addressed, stop planning. Do not repeatedly change your move or chit-chat with other agents in pursuit of a theoretically perfect plan.

    **CRITICAL Do not confuse activity with progress. Additional observations, reflections, or messages should only be performed when they can meaningfully improve your decision or execution.**
    </Instructions>
    """
    
    return federal_instructions
    
def federal_agent_voting_prompt(current_turn: int, turn_limit: int) -> str:
    voting_instructions = f"""The {current_turn} voting turn has begun.
    You are responsible for preparing the recruitment plan for the entire country, 
    after which regional agents will communicate and vote for or against your plan.
    
    If the plan is accepted, the voting procedure will end, and the recruitment will proceed according to agreed decisions.
    Otherwise, the new voting turn will begin, allowing you to read regional's agents conversation on your previous plans, 
    and modify your plan to meet their needs. 
    
    However, if the recruitment plan won't be chosen in less than {turn_limit} turns, the voting process will stop, 
    and no recruitment will be chosen for this upkeep phase.
    """
    
    return voting_instructions