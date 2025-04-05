# mcts/mcts.py
from node import Node
import random
class MCTS:
    def __init__(self, game, iterations=1000):
        self.game = game
        self.iterations = iterations
    
    def search(self):
        root = Node(game=self.game)
        
        for _ in range(self.iterations):
            node = root
            simulation_game = self.game.copy()
            
            # Sélection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                simulation_game.make_move(node.move)
            
            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                simulation_game.make_move(move)
                node = node.add_child(move, simulation_game)
            
            # Simulation
            while not simulation_game.is_terminal():
                move = random.choice(simulation_game.get_legal_moves())
                simulation_game.make_move(move)
            
            # Rétropropagation
            while node is not None:
                node.visits += 1
                result = simulation_game.get_winner()
                if result == node.player:
                    node.wins += 1
                elif result == 0:  # Match nul
                    node.wins += 0.5
                node = node.parent
        
        # Retourne le meilleur coup (le plus visité)
        return max(root.children, key=lambda c: c.visits).move