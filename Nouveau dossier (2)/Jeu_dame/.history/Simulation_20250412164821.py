from Instance import Instance
from Mouvement import Mouvement
from copy import deepcopy

class Simulation:
    def __init__(self, plateau, couleur_joueur):
        """
        Initialise la simulation.
        :param plateau: L'état actuel du plateau.
        :param couleur_joueur: La couleur du joueur à simuler ("noir" ou "blanc").
        """
        self.plateau = plateau
        self.couleur_joueur = couleur_joueur

        
    def simuler_meilleur_mouvement(self, iterations=10):
        """
        Simule tous les mouvements possibles pour les noirs et détermine le meilleur.
        :param iterations: Nombre d'instances à exécuter pour chaque mouvement.
        :return: Le meilleur mouvement pour les noirs.
        """
        meilleurs_resultats = {}
        pions_noirs = self._get_pions_noirs()

        for pion in pions_noirs:
            mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=True)
            for mouvement in mouvements:
                victoire_noir = 0
                for _ in range(iterations):
                    # Étape 1 : Créer une copie du plateau
                    plateau_copie = self.plateau.copie_sans_surface()

                    # Étape 2 : Générer les positions initiales à partir du plateau copié
                    positions_initiales = []
                    for y, ligne in enumerate(plateau_copie.cases):
                        for x, p in enumerate(ligne):
                            if p:
                                positions_initiales.append((p.couleur, (x, y)))

                    # Étape 3 : Créer une instance avec les positions initiales
                    instance = Instance(positions_initiales)

                    # Étape 4 : Appliquer le mouvement initial
                    instance.plateau.deplacer_pion(Mouvement(pion.position, mouvement))

                    # Étape 5 : Simuler la partie jusqu'à la fin
                    gagnant = instance.run_to_the_end()
                    if gagnant == "noir":
                        victoire_noir += 1

                # Enregistrer les résultats pour ce mouvement
                meilleurs_resultats[(pion.position, mouvement)] = victoire_noir

        # Trouver le mouvement avec le maximum de victoires
        meilleur_mouvement = max(meilleurs_resultats, key=meilleurs_resultats.get)
        print(f"Meilleur mouvement : {meilleur_mouvement} avec {meilleurs_resultats[meilleur_mouvement]} victoires.")
        return meilleur_mouvement
    
    def _get_pions_noirs(self):
        """
        Récupère tous les pions noirs sur le plateau.
        :return: Liste des pions noirs.
        """
        pions_noirs = []
        for ligne in self.plateau.cases:
            for pion in ligne:
                if pion and pion.couleur == "noir":
                    pions_noirs.append(pion)
        return pions_noirs
