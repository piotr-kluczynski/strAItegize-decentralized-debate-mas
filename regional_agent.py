from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage

from states import RegionalState, DecentralizedState
from prompts import regional_agent_prompt

from strAItegize_common.others.human import human

def get_regional_agent(tools: list, model):
    def regional_agent(state: RegionalState, config: RunnableConfig) -> DecentralizedState:
        configurable = config.get("configurable", {})
        is_human = configurable.get("human_mode", False)
        
        active_tools = [tool for tool in tools if tool.name not in state["blocked_tools"]]
        
        model_with_tools = model.bind_tools(active_tools)
        
        message_list = [SystemMessage(content=regional_agent_prompt(
            
        )
        )] + state["regional_messages"]
        
        if is_human:
            return {
                "regional_messages": human(
                    message_list,
                    agent_name=state["regional_id"],
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