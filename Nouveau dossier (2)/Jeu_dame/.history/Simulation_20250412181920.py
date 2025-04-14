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

    def simuler_meilleur_mouvement(self, iterations=1):
        """
        Simule tous les mouvements possibles pour le joueur sélectionné et détermine le meilleur.
        :param iterations: Nombre d'instances à exécuter pour chaque mouvement.
        :return: Le meilleur mouvement pour le joueur sélectionné.
        """
        meilleurs_resultats = {}
        pions = self._get_pions_par_couleur()
    
        # Étape 1 : Vérifier si au moins un pion peut capturer
        capture_possible = any(self.plateau.peut_manger(pion) for pion in pions)
    
        for pion in pions:
            # Étape 2 : Récupérer les mouvements possibles
            mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=capture_possible)
            for mouvement in mouvements:
                # Vérifier si le mouvement est une capture multiple
                if self.plateau.peut_manger(pion):
                    captures_multiples = self.plateau.obtenir_capture_multiple(pion, Mouvement(pion.position, mouvement))
                    for capture_sequence in captures_multiples:
                        victoire_joueur = 0
                        for _ in range(iterations):
                            # Simuler la séquence complète de captures
                            plateau_copie = self.plateau.copie_sans_surface()
                            instance = Instance(plateau_copie.cases)
                            for capture in capture_sequence:
                                instance.plateau.deplacer_pion(capture)
                            gagnant = instance.run_to_the_end()
                            if gagnant == self.couleur_joueur:
                                victoire_joueur += 1
                        meilleurs_resultats[tuple(capture_sequence)] = victoire_joueur
                else:
                    # Simuler un mouvement normal ou une capture simple
                    victoire_joueur = 0
                    for _ in range(iterations):
                        plateau_copie = self.plateau.copie_sans_surface()
                        instance = Instance(plateau_copie.cases)
                        instance.plateau.deplacer_pion(Mouvement(pion.position, mouvement))
                        gagnant = instance.run_to_the_end()
                        if gagnant == self.couleur_joueur:
                            victoire_joueur += 1
                    meilleurs_resultats[(pion.position, mouvement)] = victoire_joueur
    
        # Trouver le mouvement ou la séquence avec le maximum de victoires
        meilleur_mouvement = max(meilleurs_resultats, key=meilleurs_resultats.get)
        print(f"Meilleur mouvement : {meilleur_mouvement} avec {meilleurs_resultats[meilleur_mouvement]} victoires.")
        return meilleur_mouvement

    def _get_pions_par_couleur(self):
        """
        Récupère tous les pions de la couleur sélectionnée sur le plateau.
        :return: Liste des pions de la couleur sélectionnée.
        """
        pions = []
        for ligne in self.plateau.cases:
            for pion in ligne:
                if pion and pion.couleur == self.couleur_joueur:
                    pions.append(pion)
        return pions