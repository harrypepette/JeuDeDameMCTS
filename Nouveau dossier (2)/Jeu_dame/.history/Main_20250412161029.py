from Jeu import Jeu
from Joueur import Joueur
import sys
from Instance import Instance
from Ia import Ia

def main():
        
    jeu = Jeu()
    jeu.run()

##if __name__ == "__main__":
##    positions_initiales = [
##        ("blanc", (1, 5)), ("blanc", (3, 5)), ("blanc", (5, 5)), ("blanc", (7, 5)),
##        ("noir", (0, 2)), ("noir", (2, 2)), ("noir", (4, 2)), ("noir", (6, 2))
##    ]
##    instance = Instance(positions_initiales)
##    instance.run_to_the_end()

if __name__ == "__main__":
    
    main()


