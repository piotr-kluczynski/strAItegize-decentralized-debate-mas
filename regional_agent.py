from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage

from states import RegionalState, DecentralizedDebateState
from prompts import regional_agent_prompt

from strAItegize_common.others.human import human

def get_regional_agent(tools: list, model):
    def regional_agent(state: RegionalState, config: RunnableConfig) -> DecentralizedDebateState:
        configurable = config.get("configurable", {})
        is_human = configurable.get("human_mode", False)
        
        active_tools = [tool for tool in tools if tool.name not in state["blocked_tools"]]
        
        model_with_tools = model.bind_tools(active_tools)
        
        message_list = [SystemMessage(content=regional_agent_prompt(
            player_id=state["player_id"],
            region_id=state["current_agent_id"],
            controlled_unit_orders=state["controlled_units"],
            regions_units=state["regions_units"]
        )
        )] + state["regional_messages"]
        
        if is_human:
            return {
                "regional_messages": human(
                    message_list,
                    agent_name=state["current_agent_id"],
                    tools=active_tools
                )
            }
        else:
            return {
                "regional_messages": [
                    model_with_tools.invoke(message_list)
                ]
            }
    
    return regional_agent