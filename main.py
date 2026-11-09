from dotenv import load_dotenv
load_dotenv()

import argparse

from strAItegize_common.client.game_controller import GameController
from decentralized_debate_mas import DecentralizedDebateMAS

# Setting up app console line parameters
parser = argparse.ArgumentParser(prog="StrAItegize - Decentralized Multi-Agent System")

parser.add_argument("-p", "--port", help="Client port shared by the simulation program to connect to.", type=int, required=True)
parser.add_argument("-d", "--debug", help="If True, during agent invocation, the user takes its place using console line, otherwise all agent calls are directed to the OpenAI LLM.", type=bool, default=True, required=False)
parser.add_argument("-t", "--turn_limit", help="Defines how many turns of operation can system take in each invokation.", type=int, default=10, required=False)
# Argument for deciding which LLM model to use - TO DO

# TEMPORARY
from langchain.chat_models import init_chat_model
regional_model = init_chat_model(
    model_name="gpt-4o",
)
federal_model = init_chat_model(
    model_name="gpt-4o",
)

args = parser.parse_args()

if __name__ == '__main__':
    print("Welcome to the Centralized Multi-Agent System Architecture!")
    game_controler = GameController(simulation_port=args.port, multiagent_system=DecentralizedDebateMAS, human_mode=args.debug, federal_model=federal_model, regional_model=regional_model, turn_limit=args.turn_limit)

    game_controler.initialize_simulation()

    game_controler.start_simulation()