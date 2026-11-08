from dotenv import load_dotenv
load_dotenv()

import argparse

from strAItegize_common.client.game_controller import GameController
from centralized_mas import CentralizedMAS
# Setting up app console line parameters
parser = argparse.ArgumentParser(prog="StrAItegize - Centralized Multi-Agent System")

parser.add_argument("-p", "--port", help="Client port shared by the simulation program to connect to.", type=int, required=True)
parser.add_argument("-d", "--debug", help="If True, during agent invocation, the user takes its place using console line, otherwise all agent calls are directed to the OpenAI LLM.", type=bool, default=True, required=False)
# Argument for deciding which LLM model to use - TO DO

# TEMPORARY
from langchain.chat_models import init_chat_model
leader_model = init_chat_model(
    model_name="gpt-4o",
)
general_model = init_chat_model(
    model_name="gpt-4o",
)

args = parser.parse_args()

if __name__ == '__main__':
    print("Welcome to the Centralized Multi-Agent System Architecture!")
    game_controler = GameController(simulation_port=args.port, multiagent_system=CentralizedMAS, human_mode=args.debug, leader_model=leader_model, general_model=general_model)

    game_controler.initialize_simulation()

    game_controler.start_simulation()