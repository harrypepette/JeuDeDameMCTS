# mcts/node.py
import math

class Node:
    def __init__(self, parent=None, move=None, game=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = game.get_legal_moves()
        self.player = game.current_player
    
    def uct_select_child(self):
        """Sélectionne l'enfant avec la meilleure valeur UCT"""
        return max(self.children, key=lambda c: c.wins/c.visits + math.sqrt(2*math.log(self.visits)/c.visits))
    
    def add_child(self, move, game):
        """Ajoute un nouvel enfant"""
        child = Node(parent=self, move=move, game=game)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child