from Instance import Instance
from Mouvement import Mouvement
from copy import deepcopy

class Simulation:
    def __init__(self, plateau, joueur_noir):
        """
        Initialise la simulation.
        :param plateau: L'état actuel du plateau.
        :param joueur_noir: Le joueur noir.
        """
        self.plateau = plateau
        self.joueur_noir = joueur_noir

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
                    plateau_copie = deepcopy(self.plateau)
                    instance = Instance(plateau_copie, Mouvement(pion.position, mouvement))
                    gagnant = instance.run_to_the_end()
                    if gagnant == "noir":
                        victoire_noir += 1
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
