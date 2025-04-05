# mcts/agent.py
from mcts import MCTS
from dames.game import Game

class MCTSAgent:
    def __init__(self, iterations=1000):
        self.iterations = iterations
    
    def get_move(self, game):
        mcts = MCTS(game, self.iterations)
        return mcts.search()