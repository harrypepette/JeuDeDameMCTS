import pygame
import asyncio
import platform
import copy
from mcts_ai import MCTS  # Corriger l'importation

# Initialisation de Pygame
pygame.init()

# Constantes
BOARD_SIZE = 10
SQUARE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE
FPS = 60

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
WHITE_PIECE = (245, 245, 220)
BLACK_PIECE = (105, 105, 105)
SELECTED = (255, 255, 0)
HIGHLIGHT = (0, 255, 0)
AI_MOVE_HIGHLIGHT = (255, 105, 180)  # Rose pour montrer le dernier coup de l'IA

class Piece:
    def __init__(self, color, row, col, is_king=False):
        self.color = color
        self.row = row
        self.col = col
        self.is_king = is_king

class Game:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.turn = "white"
        self.selected_piece = None
        self.must_capture = False
        self.capture_moves = []
        self.last_ai_move = None  # Pour visualiser le dernier coup de l'IA
        self.setup_board()

    def setup_board(self):
        for row in range(4):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row][col] = Piece("black", row, col)
        for row in range(6, BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row][col] = Piece("white", row, col)

    def get_valid_moves(self, piece):
        """Retourne tous les mouvements valides pour une pièce (mouvements normaux et captures)"""
        moves = []
        captures = []
        row, col = piece.row, piece.col

        # Si des captures sont possibles dans le jeu, on doit capturer
        if self.must_capture:
            # Vérifier si cette pièce est dans les mouvements de capture obligatoires
            is_piece_in_capture_moves = any(
                capture_move[0] == row and capture_move[1] == col 
                for capture_move in self.capture_moves
            )
            if not is_piece_in_capture_moves:
                return [], []

        # Mouvements normaux pour les rois (en diagonale, distance illimitée)
        if piece.is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    if self.board[r][c] is None:
                        if not self.must_capture:
                            moves.append((r, c))
                    else:
                        break
                    r += dr
                    c += dc

        # Mouvements normaux pour les pions (en avant seulement)
        else:
            move_directions = [(-1, -1), (-1, 1)] if piece.color == "white" else [(1, -1), (1, 1)]
            for dr, dc in move_directions:
                r, c = row + dr, col + dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] is None:
                    if not self.must_capture:
                        moves.append((r, c))

        # Recherche des captures possibles
        capture_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if piece.is_king:
            for dr, dc in capture_directions:
                r, c = row + dr, col + dc
                found_opponent = False
                opponent_pos = None
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    if self.board[r][c] is not None:
                        if self.board[r][c].color != piece.color and not found_opponent:
                            found_opponent = True
                            opponent_pos = (r, c)
                        else:
                            break
                    elif found_opponent:
                        is_king_after = piece.is_king or (piece.color == "white" and r == 0) or (piece.color == "black" and r == BOARD_SIZE - 1)
                        captures.append((r, c, [opponent_pos], is_king_after))
                        break
                    r += dr
                    c += dc
        else:
            for dr, dc in capture_directions:
                r, c = row + dr, col + dc
                r2, c2 = row + 2 * dr, col + 2 * dc
                if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and
                    0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and
                    self.board[r][c] is not None and self.board[r][c].color != piece.color and
                    self.board[r2][c2] is None):
                    is_king_after = piece.is_king or (piece.color == "white" and r2 == 0) or (piece.color == "black" and r2 == BOARD_SIZE - 1)
                    captures.append((r2, c2, [(r, c)], is_king_after))

        # Recherche des captures multiples (rafle)
        if captures:
            all_captures = []
            max_capture_count = 0
            for capture in captures:
                r, c, captured, is_king = capture
                temp_board = [row[:] for row in self.board]
                temp_piece = Piece(piece.color, r, c, is_king)
                temp_board[piece.row][piece.col] = None
                temp_board[r][c] = temp_piece
                for cr, cc in captured:
                    temp_board[cr][cc] = None
                new_captures = self.get_rafle_moves(temp_piece, r, c, set(tuple(pos) for pos in captured), temp_board)
                all_captures.append(capture)
                all_captures.extend(new_captures)
            
            # Filtrer pour ne garder que les captures avec le maximum de pièces capturées
            for capture in all_captures:
                capture_count = len(capture[2])
                if capture_count > max_capture_count:
                    captures = [capture]
                    max_capture_count = capture_count
                elif capture_count == max_capture_count:
                    captures.append(capture)
            
            # Éliminer les doublons
            captures = self._remove_duplicate_captures(captures)
        
        return moves, captures

    def _remove_duplicate_captures(self, captures):
        """Élimine les captures en double en comparant leurs destinations et pièces capturées"""
        unique_captures = []
        capture_signatures = set()
        
        for capture in captures:
            r, c, captured_list, is_king = capture
            # Créer une signature unique pour cette capture
            captured_tuples = tuple(sorted((cr, cc) for cr, cc in captured_list))
            signature = (r, c, captured_tuples, is_king)
            
            if signature not in capture_signatures:
                capture_signatures.add(signature)
                unique_captures.append(capture)
                
        return unique_captures

    def get_rafle_moves(self, piece, row, col, captured, board, max_depth=20):
        """Trouve récursivement les captures multiples (rafle)
        Utilise un set pour captured pour éviter les doublons"""
        if max_depth <= 0:  # Limiter la profondeur pour éviter les boucles infinies
            return []
        
        captures = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        if piece.is_king:
            for dr, dc in directions:
                r, c = row + dr, col + dc
                found_opponent = False
                opponent_pos = None
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    if board[r][c] is not None:
                        if board[r][c].color != piece.color and not found_opponent and (r, c) not in captured:
                            found_opponent = True
                            opponent_pos = (r, c)
                        else:
                            break
                    elif found_opponent:
                        new_captured = captured.union({opponent_pos})
                        temp_board = [row[:] for row in board]
                        is_king_after = piece.is_king or (piece.color == "white" and r == 0) or (piece.color == "black" and r == BOARD_SIZE - 1)
                        temp_piece = Piece(piece.color, r, c, is_king_after)
                        temp_board[row][col] = None
                        temp_board[r][c] = temp_piece
                        temp_board[opponent_pos[0]][opponent_pos[1]] = None
                        captured_list = list(new_captured)
                        sub_captures = self.get_rafle_moves(temp_piece, r, c, new_captured, temp_board, max_depth - 1)
                        captures.append((r, c, captured_list, is_king_after))
                        captures.extend(sub_captures)
                        break
                    r += dr
                    c += dc
        else:
            for dr, dc in directions:
                r, c = row + dr, col + dc
                r2, c2 = row + 2 * dr, col + 2 * dc
                if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and
                    0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and
                    board[r][c] is not None and board[r][c].color != piece.color and
                    board[r2][c2] is None and (r, c) not in captured):
                    new_captured = captured.union({(r, c)})
                    temp_board = [row[:] for row in board]
                    is_king_after = piece.is_king or (piece.color == "white" and r2 == 0) or (piece.color == "black" and r2 == BOARD_SIZE - 1)
                    temp_piece = Piece(piece.color, r2, c2, is_king_after)
                    temp_board[row][col] = None
                    temp_board[r2][c2] = temp_piece
                    temp_board[r][c] = None
                    captured_list = list(new_captured)
                    sub_captures = self.get_rafle_moves(temp_piece, r2, c2, new_captured, temp_board, max_depth - 1)
                    captures.append((r2, c2, captured_list, is_king_after))
                    captures.extend(sub_captures)

        return captures
    
    def get_all_captures(self):
        """Retourne toutes les captures possibles pour le joueur actuel,
        en gardant seulement celles qui capturent le maximum de pièces"""
        captures = []
        max_capture_count = 0

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece and piece.color == self.turn:
                    _, piece_captures = self.get_valid_moves(piece)
                    for capture in piece_captures:
                        capture_count = len(capture[2])
                        if capture_count > max_capture_count:
                            captures = [(piece.row, piece.col, capture[0], capture[1], capture[2], capture[3])]
                            max_capture_count = capture_count
                        elif capture_count == max_capture_count:
                            captures.append((piece.row, piece.col, capture[0], capture[1], capture[2], capture[3]))
        
        return captures

    def move_piece(self, piece, row, col, captured=None, is_king_after_move=False):
        """Déplace une pièce sur le plateau et gère les captures.
        Retourne toujours une tuple (from_pos, to_pos) pour le suivi des mouvements."""
        # Enregistrer le mouvement pour la visualisation
        from_pos = (piece.row, piece.col)
        to_pos = (row, col)
        
        # Déplacer la pièce
        self.board[piece.row][piece.col] = None
        piece.row, piece.col = row, col
        self.board[row][col] = piece

        # Supprimer les pièces capturées
        if captured:
            for r, c in captured:
                self.board[r][c] = None

        # Vérifier la promotion en dame
        if not piece.is_king:
            if (piece.color == "white" and row == 0) or (piece.color == "black" and row == BOARD_SIZE - 1):
                piece.is_king = True
            # Appliquer la promotion si spécifiée dans les paramètres
            elif is_king_after_move:
                piece.is_king = True

        # Vérifier s'il y a d'autres captures possibles pour cette pièce
        if captured:
            _, captures = self.get_valid_moves(piece)
            if captures:
                self.must_capture = True
                self.capture_moves = [(piece.row, piece.col, cap[0], cap[1], cap[2], cap[3]) for cap in captures]
                self.selected_piece = piece
                return from_pos, to_pos  # Toujours retourner la position

        # Sinon, passer au tour suivant
        self.turn = "black" if self.turn == "white" else "white"
        self.selected_piece = None
        self.must_capture = False
        self.capture_moves = []
        
        return from_pos, to_pos  # Toujours retourner la position

    def check_for_captures(self):
        """Vérifie s'il y a des captures obligatoires pour le joueur actuel"""
        captures = self.get_all_captures()
        if captures:
            self.must_capture = True
            self.capture_moves = captures
            return True
        else:
            self.must_capture = False
            self.capture_moves = []
            return False

    def is_game_over(self):
        """Vérifie si le jeu est terminé et retourne le gagnant ou None"""
        white_pieces = sum(1 for row in self.board for piece in row if piece and piece.color == "white")
        black_pieces = sum(1 for row in self.board for piece in row if piece and piece.color == "black")

        # Vérifier si un joueur n'a plus de pièces
        if white_pieces == 0:
            return "black"
        if black_pieces == 0:
            return "white"

        # Vérifier si le joueur actuel a des mouvements possibles
        has_moves = False
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece and piece.color == self.turn:
                    moves, captures = self.get_valid_moves(piece)
                    if moves or captures:
                        has_moves = True
                        break
            if has_moves:
                break

        # Si le joueur actuel n'a pas de mouvements, l'autre joueur gagne
        return None if has_moves else ("black" if self.turn == "white" else "white")

def draw_board(screen, game):
    """Dessine le plateau de jeu et les pièces"""
    # Dessiner les cases du plateau
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    # Surligner le dernier mouvement de l'IA
    if game.last_ai_move:
        from_row, from_col, to_row, to_col = game.last_ai_move
        pygame.draw.rect(screen, AI_MOVE_HIGHLIGHT, (
            to_col * SQUARE_SIZE, to_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE
        ), 3)

    # Dessiner les pièces
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = game.board[row][col]
            if piece:
                color = WHITE_PIECE if piece.color == "white" else BLACK_PIECE
                pygame.draw.circle(screen, color, (
                    col * SQUARE_SIZE + SQUARE_SIZE // 2,
                    row * SQUARE_SIZE + SQUARE_SIZE // 2
                ), SQUARE_SIZE // 2 - 5)
                
                # Marquer les rois avec une couronne dorée
                if piece.is_king:
                    pygame.draw.circle(screen, (255, 215, 0), (
                        col * SQUARE_SIZE + SQUARE_SIZE // 2,
                        row * SQUARE_SIZE + SQUARE_SIZE // 2
                    ), SQUARE_SIZE // 4, 3)

    # Surligner la pièce sélectionnée
    if game.selected_piece:
        row, col = game.selected_piece.row, game.selected_piece.col
        pygame.draw.rect(screen, SELECTED, (
            col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE
        ), 3)

        # Afficher les mouvements possibles pour la pièce sélectionnée
        moves, captures = game.get_valid_moves(game.selected_piece)
        for r, c in moves:
            pygame.draw.circle(screen, HIGHLIGHT, (
                c * SQUARE_SIZE + SQUARE_SIZE // 2,
                r * SQUARE_SIZE + SQUARE_SIZE // 2
            ), 10)
            
        for capture in captures:
            r, c = capture[:2]
            pygame.draw.circle(screen, HIGHLIGHT, (
                c * SQUARE_SIZE + SQUARE_SIZE // 2,
                r * SQUARE_SIZE + SQUARE_SIZE // 2
            ), 10)

async def main():
    """Fonction principale qui gère la boucle de jeu"""
    # Configuration
    player_color = "white"  # Le joueur joue avec les pièces blanches
    ai_color = "black"      # L'IA joue avec les pièces noires
    ai_thinking_time = 5.0  # Temps de réflexion de l'IA en secondes
    ai_difficulty = "hard"  # Niveau de difficulté: "easy", "medium", "hard"
    num_cores = 4  # Nombre de cœurs pour le MCTS (parallélisation)
    
    # Initialisation de l'affichage
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Jeu de Dames avec IA")
    clock = pygame.time.Clock()
    game = Game()
    
    # Créer l'IA MCTS
    ai = MCTS(game, ai_color=ai_color, simulation_time=ai_thinking_time, difficulty=ai_difficulty, num_processes=num_cores)
    
    running = True
    game_over = False
    
    # Message et police pour le message de fin de jeu
    font = pygame.font.SysFont('Arial', 30)
    game_over_text = ""
    
    while running:
        # Tour de l'IA
        if game.turn == ai_color and not game_over:
            # Afficher un message indiquant que l'IA réfléchit
            screen.fill(WHITE)
            draw_board(screen, game)
            thinking_text = font.render("L'IA réfléchit...", True, BLACK)
            screen.blit(thinking_text, (WINDOW_SIZE // 2 - thinking_text.get_width() // 2, 
                                       WINDOW_SIZE // 2 - thinking_text.get_height() // 2))
            pygame.display.flip()
            
            # Laisser le temps au rendu de s'afficher
            await asyncio.sleep(0.1)
            
            # Créer une copie profonde du jeu pour l'IA
            ai_game_copy = copy.deepcopy(game)
            ai.game = ai_game_copy
            
            # Obtenir le meilleur coup selon MCTS
            best_move = ai.get_best_move()
            
            if best_move:
                row_from, col_from, row_to, col_to, captured, is_king_after = best_move
                
                # Vérifier si la pièce existe encore sur le plateau réel
                piece = game.board[row_from][col_from]
                if piece and piece.color == ai_color:
                    # Appliquer le mouvement
                    from_pos, to_pos = game.move_piece(piece, row_to, col_to, captured, is_king_after)
                    
                    # Mettre à jour la visualisation du dernier coup de l'IA
                    game.last_ai_move = (from_pos[0], from_pos[1], to_pos[0], to_pos[1])
                else:
                    # Si la pièce n'existe plus, c'est probablement dû à une désynchronisation
                    # On passe au tour du joueur
                    game.turn = player_color
        
        # Traitement des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not game_over and game.turn == player_color:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // SQUARE_SIZE
                    row = event.pos[1] // SQUARE_SIZE
                    
                    # Vérifier les limites du plateau
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                        # Vérifier s'il y a des captures obligatoires
                        game.check_for_captures()
                        
                        piece = game.board[row][col]
                        
                        # Sélectionner une pièce
                        if piece and piece.color == game.turn:
                            # Vérifier si cette pièce peut se déplacer selon les règles actuelles
                            moves, captures = game.get_valid_moves(piece)
                            
                            if game.must_capture and not captures:
                                # Si des captures sont obligatoires mais cette pièce n'en a pas
                                continue
                                
                            if piece == game.selected_piece:
                                # Désélectionner la pièce
                                game.selected_piece = None
                            else:
                                # Sélectionner la pièce
                                game.selected_piece = piece
                                
                        # Déplacer une pièce sélectionnée
                        elif game.selected_piece:
                            moves, captures = game.get_valid_moves(game.selected_piece)
                            
                            # Mouvement normal
                            if (row, col) in moves:
                                is_king_after = game.selected_piece.is_king or (
                                    (game.selected_piece.color == "white" and row == 0) or 
                                    (game.selected_piece.color == "black" and row == BOARD_SIZE - 1)
                                )
                                game.move_piece(game.selected_piece, row, col, None, is_king_after)
                            
                            # Capture
                            elif any((row, col) == cap[:2] for cap in captures):
                                for capture in captures:
                                    if (row, col) == capture[:2]:
                                        game.move_piece(game.selected_piece, row, col, capture[2], capture[3])
                                        break
                
        # Vérifier si le jeu est terminé
        winner = game.is_game_over()
        if winner and not game_over:
            game_over = True
            if winner == player_color:
                game_over_text = "Vous avez gagné !"
            elif winner == ai_color:
                game_over_text = "L'IA a gagné !"
            else:
                game_over_text = "Match nul !"
            
            print(game_over_text)

        # Rendu
        screen.fill(WHITE)
        draw_board(screen, game)
        
        # Afficher le message de fin de partie
        if game_over:
            text_surface = font.render(game_over_text, True, BLACK)
            text_rect = text_surface.get_rect(center=(WINDOW_SIZE // 2, 20))
            screen.blit(text_surface, text_rect)
            
        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)

    pygame.quit()

if __name__ == "__main__":
    # Exécuter la fonction main() asynchrone
    if platform.system() == "Emscripten":
        # Exécution dans un navigateur (WebAssembly)
        import asyncio
        asyncio.ensure_future(main())
    else:
        # Exécution normale sur desktop
        if platform.system() == "Windows":
            # Windows nécessite un traitement particulier pour asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())