import pygame
from Jeu import Jeu

# Dimensions de la fenêtre
WINDOW_WIDTH = 800  # Largeur totale (plateau + interface)
WINDOW_HEIGHT = 600  # Hauteur de la fenêtre
BOARD_WIDTH = 500  # Largeur du plateau de jeu
INTERFACE_WIDTH = WINDOW_WIDTH - BOARD_WIDTH  # Largeur de l'interface

def main():
    pygame.init()
    
    # Création de la fenêtre
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Jeu de Dame")
    
    # Couleurs
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    
    # Police pour l'interface
    font = pygame.font.Font(None, 36)
    
    # Initialisation du jeu
    jeu = Jeu()
    running = True
    
    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Dessin du plateau (appel à votre méthode existante)
        screen.fill(WHITE)
        jeu.draw_board(screen)  # Assurez-vous que cette méthode dessine le plateau dans la zone dédiée
        
        # Dessin de l'interface
        pygame.draw.rect(screen, GRAY, (BOARD_WIDTH, 0, INTERFACE_WIDTH, WINDOW_HEIGHT))  # Zone de l'interface
        
        # Affichage des informations
        turn_text = font.render(f"À qui le tour : {jeu.current_player}", True, BLACK)
        screen.blit(turn_text, (BOARD_WIDTH + 20, 20))
        
        # Exemple : Afficher un message du terminal
        log_text = font.render("Dernière action : Déplacement", True, BLACK)
        screen.blit(log_text, (BOARD_WIDTH + 20, 60))
        
        # Mise à jour de l'affichage
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()