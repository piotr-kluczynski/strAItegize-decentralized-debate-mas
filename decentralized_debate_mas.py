from strAItegize_common.others.order import Order
from strAItegize_common.others.recruitment_choice import RecruitmentChoice

from strAItegize_common.client.communication.communication_context import CommunicationContext
from strAItegize_common.client.multiagent_system import MultiAgentSystem

class DecentralizedDebateMAS(MultiAgentSystem):
    def __init__(self, communication: CommunicationContext, human_mode: bool=False, **kwargs):
        pass
    
    def initialize_graph(self,
        player_id: str, players: list[str],
        regions: list[str], neighbouring_regions: dict[str, list[str]],
        final_round: int
    ):
        pass
    
    def invoke_orders(self, 
        current_round: int, final_round: int,
        players: list[str], leaderboard: dict[str, int], 
        units: list[str], regions: list[str],
        tresspasers: dict[str, list[str]], events: list[str]
    ) -> list[Order]:
        pass
    
    def invoke_upkeep(self, supply_surplus: int, regions: list[str]) -> list[RecruitmentChoice]:
        pass