import copy
import math
import random
import time
import json
import os
import multiprocessing
from functools import partial

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state  # État du jeu
        self.parent = parent  # Nœud parent
        self.move = move  # Action qui a mené à cet état
        self.children = []  # Nœuds enfants
        self.wins = 0  # Nombre de victoires 
        self.visits = 0  # Nombre de visites
        self.untried_moves = self.get_untried_moves()  # Actions non encore essayées
    
    def get_untried_moves(self):
        """Retourne la liste des mouvements possibles non encore essayés."""
        original_turn = self.game_state.turn
        
        # Vérifier s'il y a des captures obligatoires
        captures = self.game_state.get_all_captures()
        
        if captures:
            # Format des captures: (row_from, col_from, row_to, col_to, captured_pieces, is_king_after)
            return captures
        
        # Si pas de captures, collecter tous les mouvements normaux
        all_moves = []
        for row in range(10):
            for col in range(10):
                piece = self.game_state.board[row][col]
                if piece and piece.color == original_turn:
                    moves, _ = self.game_state.get_valid_moves(piece)
                    for move_row, move_col in moves:
                        all_moves.append((row, col, move_row, move_col, [], False))
        
        return all_moves
    
    def uct_select_child(self, exploration_weight=1.414):
        """Sélectionne un enfant selon la formule UCT."""
        assert self.children
        
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def uct_score(child):
            # Exploitation + exploration
            if child.visits == 0:
                return float('inf')
            exploitation = child.wins / child.visits
            exploration = exploration_weight * math.sqrt(log_visits / child.visits)
            return exploitation + exploration
        
        return max(self.children, key=uct_score)
    
    def add_child(self, move, game_state):
        """Ajoute un enfant au nœud."""
        child = Node(game_state, parent=self, move=move)
        if move in self.untried_moves:  # Vérification pour éviter les erreurs
            self.untried_moves.remove(move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """Met à jour les statistiques du nœud."""
        self.visits += 1
        self.wins += result

class MCTS:
    # Dictionnaire d'ouvertures recommandées
    OPENINGS = {
        "start": [
            (3, 4, 4, 5, [], False),  # Quelques premiers coups recommandés
            (6, 3, 5, 4, [], False),
            (2, 5, 3, 4, [], False),
            (6, 5, 5, 4, [], False),
            (2, 3, 3, 4, [], False)
        ]
    }
    
    def __init__(self, game, ai_color="black", simulation_time=3.0, max_iterations=10000, difficulty="medium", num_processes=4):
        """
        Initialise le MCTS.
        
        Args:
            game: Instance du jeu de dames
            ai_color: Couleur jouée par l'IA ("black" ou "white")
            simulation_time: Temps maximum pour la recherche en secondes
            max_iterations: Nombre maximum d'itérations
            difficulty: Niveau de difficulté ("easy", "medium", "hard")
            num_processes: Nombre de processus à utiliser (None = utiliser tous les CPU disponibles)
        """
        self.game = game
        self.ai_color = ai_color
        self.max_iterations = max_iterations
        self.difficulty = difficulty
        self.transposition_table = {}  # Table pour stocker les positions déjà évaluées
        
        # Configurer le nombre de processus pour le parallélisme
        self.num_processes = num_processes if num_processes else multiprocessing.cpu_count()
        
        # Ajuster les paramètres selon la difficulté
        if difficulty == "easy":
            self.simulation_time = 1.0
            self.exploration_weight = 2.0  # Plus d'exploration aléatoire
        elif difficulty == "medium":
            self.simulation_time = 3.0
            self.exploration_weight = 1.414  # Valeur standard
        elif difficulty == "hard":
            self.simulation_time = 5.0
            self.exploration_weight = 1.0  # Plus d'exploitation des bons coups
        else:
            self.simulation_time = simulation_time
            self.exploration_weight = 1.414
        
        # Charger les positions apprises
        self.load_learned_positions()
    
    def get_best_move(self):
        """Retourne le meilleur mouvement selon MCTS, en respectant les captures obligatoires."""
        # Vérifier d'abord s'il y a des captures obligatoires
        captures = self.game.get_all_captures()
        if captures:
            # S'il y a des captures obligatoires, on les priorise toujours
            # On peut faire un petit calcul MCTS pour choisir la meilleure capture si plusieurs sont disponibles
            if len(captures) == 1:
                return captures[0]  # Une seule capture possible, pas besoin de MCTS
            else:
                # Plusieurs captures possibles, utiliser MCTS pour choisir la meilleure
                root = Node(copy.deepcopy(self.game))
                # Définir uniquement les captures comme coups possibles
                root.untried_moves = captures
                
                # Mini recherche MCTS pour choisir la meilleure capture
                end_time = time.time() + self.simulation_time / 2  # Moins de temps car choix limité
                iterations = 0
                
                while time.time() < end_time and iterations < self.max_iterations / 2:
                    # 1. Sélection
                    node = self.select(root)
                    
                    # 2. Expansion
                    if node.untried_moves and not self.is_terminal(node.game_state):
                        node = self.expand(node)
                    
                    # 3. Simulation
                    result = self.simulate(node)
                    
                    # 4. Rétropropagation
                    self.backpropagate(node, result)
                    
                    iterations += 1
                
                if root.children:
                    return max(root.children, key=lambda c: c.visits).move
                return captures[0]  # Par défaut, première capture si aucun enfant créé
        
        # S'il n'y a pas de captures obligatoires, on peut considérer les ouvertures
        total_pieces = sum(1 for row in self.game.board for piece in row if piece is not None)
        if total_pieces >= 38 and self.game.turn == self.ai_color:  # Presque toutes les pièces sont sur le plateau
            # Utiliser une ouverture standard, mais vérifier sa validité
            for move in self.OPENINGS["start"]:
                row_from, col_from, row_to, col_to, _, _ = move
                # Vérifier si le mouvement est possible
                piece = self.game.board[row_from][col_from]
                if piece and piece.color == self.ai_color:
                    moves, _ = self.game.get_valid_moves(piece)
                    if (row_to, col_to) in moves:
                        return move
        
        # Si on n'a pas trouvé d'ouverture ou si on n'est plus en début de partie, MCTS standard
        root = Node(copy.deepcopy(self.game))
        
        # Vérifier s'il y a un seul mouvement possible
        if len(root.untried_moves) == 1:
            return root.untried_moves[0]
        
        # Lancer la recherche parallèle
        return self.parallel_search(root)
    
    def parallel_search(self, root):
        """Effectue la recherche MCTS en parallèle sur plusieurs processus."""
        # Déterminer le nombre d'itérations par processus
        iterations_per_process = self.max_iterations // self.num_processes
        
        # Créer une liste pour stocker les résultats des différents processus
        results = []
        
        # Calculer le temps de recherche pour chaque processus
        time_per_process = self.simulation_time
        
        # Préparer l'arbre initial avec quelques expansions pour éviter les doublons
        # Effectuer quelques expansions initiales pour avoir des points de départ
        for _ in range(min(self.num_processes * 2, len(root.untried_moves))):
            if root.untried_moves:
                self.expand(root)
        
        # Démarrer un pool de processus
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Créer une fonction partielle avec les paramètres communs
            search_func = partial(
                self.process_search, 
                game_state=copy.deepcopy(root.game_state),
                ai_color=self.ai_color,
                exploration_weight=self.exploration_weight,
                search_time=time_per_process,
                iterations=iterations_per_process
            )
            
            # Lancer les processus et collecter les résultats
            results = pool.map(search_func, range(self.num_processes))
        
        # Fusionner les résultats des différents processus
        # Chaque résultat est un dictionnaire {move: (visits, wins)}
        merged_results = {}
        for result in results:
            for move, (visits, wins) in result.items():
                if move in merged_results:
                    merged_visits, merged_wins = merged_results[move]
                    merged_results[move] = (merged_visits + visits, merged_wins + wins)
                else:
                    merged_results[move] = (visits, wins)
        
        # Mise à jour des statistiques de l'arbre principal
        for child in root.children:
            move_str = str(child.move)
            if move_str in merged_results:
                visits, wins = merged_results[move_str]
                child.visits += visits
                child.wins += wins
        
        # Sauvegarder les positions apprises
        self.save_learned_positions()
        
        # Retourner le mouvement avec le plus grand nombre de visites
        if root.children:
            return max(root.children, key=lambda c: c.visits).move
        
        # En cas de problème, retourner un mouvement valide au hasard
        if root.untried_moves:
            return random.choice(root.untried_moves)
        
        return None  # Au cas où il n'y a pas de coups possibles
    
    @staticmethod
    def process_search(process_id, game_state, ai_color, exploration_weight, search_time, iterations):
        """Fonction exécutée par chaque processus pour effectuer une recherche MCTS indépendante."""
        # Créer une nouvelle instance de MCTS pour ce processus avec sa propre table de transposition
        local_mcts = MCTS(game_state, ai_color=ai_color, simulation_time=search_time, 
                          max_iterations=iterations, difficulty="medium", num_processes=1)
        local_mcts.exploration_weight = exploration_weight
        
        # Créer un nœud racine pour ce processus
        root = Node(game_state)
        
        # Effectuer la recherche MCTS dans ce processus
        end_time = time.time() + search_time
        iteration_count = 0
        
        while time.time() < end_time and iteration_count < iterations:
            # 1. Sélection
            node = local_mcts.select(root)
            
            # 2. Expansion
            if node.untried_moves and not local_mcts.is_terminal(node.game_state):
                node = local_mcts.expand(node)
            
            # 3. Simulation
            result = local_mcts.simulate(node)
            
            # 4. Rétropropagation
            local_mcts.backpropagate(node, result)
            
            iteration_count += 1
        
        # Collecter les résultats de ce processus
        results = {}
        for child in root.children:
            # Utiliser str(move) comme clé pour pouvoir le passer entre processus
            results[str(child.move)] = (child.visits, child.wins)
        
        return results
    
    def select(self, node):
        """Sélectionne un nœud à développer."""
        while node.children and not node.untried_moves:
            # Ne pas sélectionner un nœud terminal
            if self.is_terminal(node.game_state):
                return node
            node = node.uct_select_child(self.exploration_weight)
        return node
    
    def expand(self, node):
        """Développe le nœud en ajoutant un enfant."""
        if not node.untried_moves:  # Vérification supplémentaire
            return node
            
        move = random.choice(node.untried_moves)
        
        # Copie l'état du jeu et applique le mouvement
        new_state = copy.deepcopy(node.game_state)
        
        # Format du mouvement: (row_from, col_from, row_to, col_to, captured_pieces, is_king_after)
        row_from, col_from, row_to, col_to, captured, is_king_after = move
        piece = new_state.board[row_from][col_from]
        
        if piece is None:  # Vérification de sécurité
            node.untried_moves.remove(move)
            return node
        
        # Appliquer le mouvement
        move_result = new_state.move_piece(piece, row_to, col_to, captured, is_king_after)
        
        # Si c'est toujours au même joueur de jouer (capture multiple), 
        # synchroniser le tour avec la couleur de l'IA pour la simulation
        if new_state.must_capture and new_state.selected_piece:
            new_state.turn = piece.color
        
        return node.add_child(move, new_state)
    
    def simulate(self, node):
        """Simule une partie et retourne le résultat avec évaluation heuristique améliorée."""
        state = copy.deepcopy(node.game_state)
        
        # Vérifier si l'état est terminal avant de simuler
        winner = self.is_terminal(state)
        if winner:
            # Score du point de vue de l'IA
            if winner == self.ai_color:
                result = 1.0  # Victoire
            elif winner is None or winner == "draw":
                result = 0.5  # Match nul
            else:
                result = 0.0  # Défaite
                
            # Stocker le résultat dans la table de transposition
            state_hash = self.hash_state(state)
            self.transposition_table[state_hash] = result
            return result
        
        # Si cette position a déjà été évaluée, utilisez la valeur stockée
        state_hash = self.hash_state(state)
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]
        
        # Simule jusqu'à un état terminal
        depth = 0
        max_depth = 100  # Évite les boucles infinies
        
        while not self.is_terminal(state) and depth < max_depth:
            # Terminaison anticipée pour avantage matériel écrasant
            if depth > 20:  # Après un certain nombre de coups
                ai_material = 0
                opp_material = 0
                for row in range(10):
                    for col in range(10):
                        piece = state.board[row][col]
                        if piece:
                            value = 3 if piece.is_king else 1
                            if piece.color == self.ai_color:
                                ai_material += value
                            else:
                                opp_material += value
                
                # Si l'avantage est écrasant, terminer la simulation
                if ai_material > opp_material * 2:
                    result = 0.9  # Presque gagné
                    self.transposition_table[state_hash] = result
                    return result
                if opp_material > ai_material * 2:
                    result = 0.1  # Presque perdu
                    self.transposition_table[state_hash] = result
                    return result
            
            # Collecte les mouvements possibles
            moves = []
            
            # Vérifier les captures obligatoires
            captures = state.get_all_captures()
            if captures:
                moves = captures
            else:
                # Collecter tous les mouvements normaux
                for row in range(10):
                    for col in range(10):
                        piece = state.board[row][col]
                        if piece and piece.color == state.turn:
                            normal_moves, _ = state.get_valid_moves(piece)
                            for move_row, move_col in normal_moves:
                                moves.append((row, col, move_row, move_col, [], False))
            
            if not moves:
                break
                
            # Choisir un mouvement aléatoire
            move = random.choice(moves)
            row_from, col_from, row_to, col_to, captured, is_king_after = move
            piece = state.board[row_from][col_from]
            
            if piece is None:  # Vérification de sécurité
                break
                
            # Appliquer le mouvement
            state.move_piece(piece, row_to, col_to, captured, is_king_after)
            depth += 1
        
        # Détermine le gagnant
        winner = state.is_game_over()
        
        # Si le jeu n'est pas terminé après max_depth ou si on utilise l'évaluation heuristique
        if not winner or depth >= max_depth:
            result = self.evaluate_position(state)
            self.transposition_table[state_hash] = result
            return result
        
        # Score du point de vue de l'IA
        if winner == self.ai_color:
            result = 1.0  # Victoire
        elif winner is None:
            result = 0.5  # Match nul
        else:
            result = 0.0  # Défaite
            
        # Stocker le résultat dans la table de transposition
        self.transposition_table[state_hash] = result
        return result
    
    def backpropagate(self, node, result):
        """Rétropropage le résultat dans l'arbre."""
        while node:
            node.update(result)
            node = node.parent
            # Inverser le résultat pour le nœud parent (point de vue opposé)
            if node:  # S'assurer que node n'est pas None avant d'inverser
                result = 1.0 - result
    
    def is_terminal(self, state):
        """Vérifie si l'état est terminal."""
        return state.is_game_over()
    
    def hash_state(self, state):
        """Crée un hash unique pour l'état du jeu actuel."""
        board_hash = ""
        for row in range(10):
            for col in range(10):
                piece = state.board[row][col]
                if piece:
                    if piece.color == "white":
                        board_hash += "w" if not piece.is_king else "W"
                    else:
                        board_hash += "b" if not piece.is_king else "B"
                else:
                    board_hash += "."
        return board_hash + state.turn[0]  # Ajouter le joueur actuel
    
    def evaluate_position(self, state):
        """Évalue heuristiquement une position du jeu."""
        ai_color = self.ai_color
        opponent_color = "white" if ai_color == "black" else "black"
        
        # Compter les pièces avec des poids
        ai_men = 0
        ai_kings = 0
        opp_men = 0
        opp_kings = 0
        
        # Bonus pour les positions stratégiques
        ai_edge_bonus = 0
        opp_edge_bonus = 0
        ai_center_bonus = 0
        opp_center_bonus = 0
        ai_advancement = 0
        opp_advancement = 0
        
        for row in range(10):
            for col in range(10):
                piece = state.board[row][col]
                if piece:
                    # Bonus pour les pièces sur les bords (plus difficiles à capturer)
                    is_edge = row == 0 or row == 9 or col == 0 or col == 9
                    
                    # Bonus pour les pièces au centre (contrôle le plateau)
                    is_center = 3 <= row <= 6 and 3 <= col <= 6
                    
                    # Bonus pour l'avancement (pions qui avancent vers la promotion)
                    advancement_value = 0
                    if not piece.is_king:
                        if piece.color == "white":
                            advancement_value = (10 - row) / 10  # Plus proche de la rangée 0
                        else:
                            advancement_value = row / 10  # Plus proche de la rangée 9
                    
                    if piece.color == ai_color:
                        if piece.is_king:
                            ai_kings += 1
                        else:
                            ai_men += 1
                        
                        if is_edge:
                            ai_edge_bonus += 0.2
                        if is_center:
                            ai_center_bonus += 0.3
                        
                        ai_advancement += advancement_value
                    else:
                        if piece.is_king:
                            opp_kings += 1
                        else:
                            opp_men += 1
                        
                        if is_edge:
                            opp_edge_bonus += 0.2
                        if is_center:
                            opp_center_bonus += 0.3
                        
                        opp_advancement += advancement_value
        
        # Vérifier s'il reste des pièces
        if ai_men + ai_kings == 0:
            return 0.0  # L'IA a perdu
        if opp_men + opp_kings == 0:
            return 1.0  # L'IA a gagné
        
        # Valeurs des pièces
        ai_value = (ai_men * 1.0) + (ai_kings * 3.0) + ai_edge_bonus + ai_center_bonus + (ai_advancement * 0.5)
        opp_value = (opp_men * 1.0) + (opp_kings * 3.0) + opp_edge_bonus + opp_center_bonus + (opp_advancement * 0.5)
        
        # Bonus pour les pièces mobiles et les menaces
        ai_mobility = self.count_mobility(state, ai_color)
        opp_mobility = self.count_mobility(state, opponent_color)
        
        ai_value += ai_mobility * 0.1
        opp_value += opp_mobility * 0.1
        
        # Normaliser le score entre 0 et 1
        if ai_value + opp_value > 0:
            return ai_value / (ai_value + opp_value)
        return 0.5  # Position égale
    
    def count_mobility(self, state, color):
        """Compte le nombre de mouvements possibles pour une couleur."""
        mobility = 0
        for row in range(10):
            for col in range(10):
                piece = state.board[row][col]
                if piece and piece.color == color:
                    moves, captures = state.get_valid_moves(piece)
                    mobility += len(moves) + (2 * len(captures))  # Les captures valent plus
        return mobility
    
    def save_learned_positions(self):
        """Sauvegarde les positions apprises dans un fichier."""
        try:
            # Limitons le nombre de positions pour éviter des fichiers trop grands
            positions_to_save = {}
            count = 0
            
            # Prendre les 10000 premières positions (ou moins s'il y en a moins)
            for key, value in self.transposition_table.items():
                positions_to_save[key] = value
                count += 1
                if count >= 10000:  # Limiter à 10000 positions
                    break
            
            with open("learned_positions.json", "w") as f:
                json.dump(positions_to_save, f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des positions: {e}")
    
    def load_learned_positions(self):
        """Charge les positions apprises depuis un fichier."""
        try:
            self.transposition_table = {}  # Réinitialisez la table
            if os.path.exists("learned_positions.json"):
                with open("learned_positions.json", "r") as f:
                    data = json.load(f)
                    # Valider les données
                    for key, value in data.items():
                        if isinstance(key, str) and isinstance(value, (int, float)) and 0 <= value <= 1:
                            self.transposition_table[key] = value
        except Exception as e:
            print(f"Erreur lors du chargement des positions: {e}")
            self.transposition_table = {}