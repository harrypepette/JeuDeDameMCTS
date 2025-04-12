from Jeu import Jeu

import sys

from IA import IA  # Import the IA class
import argparse

def main():
    parser = argparse.ArgumentParser(description='Jeu de Dames avec IA')
    parser.add_argument('--ia', action='store_true', help='Activer l\'IA')
    parser.add_argument('--couleur-ia', choices=['blanc', 'noir'], default='noir',
                        help='Couleur de l\'IA (blanc ou noir)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Nombre d\'itérations pour l\'algorithme MCTS')
    parser.add_argument('--exploration', type=float, default=0.7,
                        help='Constante d\'exploration pour UCT')
    
    args = parser.parse_args()

    print("Lancement du jeu graphique")
    jeu = Jeu(ia_active=args.ia, ia_couleur=args.couleur_ia, ia_iterations=args.iterations)
    jeu.run()

if __name__ == "__main__":
    main()