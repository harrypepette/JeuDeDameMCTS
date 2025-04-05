# mcts/agent.py
from mcts import MCTS
from Jeu_dame import Jeu_dame
import random

class MCTSAgent:
    def __init__(self, iterations=1000):
        self.iterations = iterations
    
    def get_move(self, game):
        mcts = MCTS(game, self.iterations)
        return mcts.search()