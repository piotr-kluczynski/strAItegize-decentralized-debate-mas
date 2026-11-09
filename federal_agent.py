from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage

from states import DecentralizedState
from prompts import federal_agent_prompt

from strAItegize_common.others.human import human

def get_federal_agent(tools: list, model):
    def federal_agent(state: DecentralizedState, config: RunnableConfig) -> DecentralizedState:
        configurable = config.get("configurable", {})
        is_human = configurable.get("human_mode", False)
        
        active_tools = [tool for tool in tools if tool.name not in state["blocked_tools"]]
        
        model_with_tools = model.bind_tools(active_tools)
        
        message_list = [SystemMessage(content=federal_agent_prompt(
            
        )
        )] + state["federal_agent_messages"]
        
        if is_human:
            return {
                "federal_agent_messages": human(
                    message_list,
                    agent_name="federal_agent",
                    tools=active_tools
                )
            }
        else:
            return {
                "federal_agent_messages": [
                    model_with_tools.invoke(message_list)
                ]
            }
    
    return federal_agent