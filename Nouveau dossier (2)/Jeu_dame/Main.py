# Dans Main.py
from Jeu import Jeu
from Joueur import Joueur
import sys
from Instance import SimulationAvancee
from Ia import Ia

def main():
    print("Bienvenue dans le jeu de dames !")
    print("Choisissez le joueur à simuler :")
    print("1. Noir")
    print("2. Blanc")
    choix = input("Entrez votre choix (1 ou 2) : ")

    if choix == "1":
        couleur_simulee = "noir"
    elif choix == "2":
        couleur_simulee = "blanc"
    else:
        print("Choix invalide, le joueur noir sera simulé par défaut.")
        couleur_simulee = "noir"
    
    
    jeu = Jeu(couleur_simulee, iterations=5000)
    jeu.run()

if __name__ == "__main__":
    main()