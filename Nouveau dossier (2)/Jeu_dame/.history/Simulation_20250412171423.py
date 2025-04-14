from Instance import Instance
from Mouvement import Mouvement
from concurrent.futures import ProcessPoolExecutor

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
        Simule tous les mouvements possibles pour le joueur sélectionné et détermine le meilleur.
        :param iterations: Nombre d'instances à exécuter pour chaque mouvement.
        :return: Le meilleur mouvement pour le joueur sélectionné.
        """
        meilleurs_resultats = {}
        pions = self._get_pions_par_couleur()

        # Étape 1 : Vérifier si au moins un pion peut capturer
        capture_possible = any(self.plateau.peut_manger(pion) for pion in pions)

        def simuler_mouvement(pion, mouvement):
            """
            Fonction pour simuler un mouvement donné plusieurs fois.
            """
            victoire_joueur = 0
            for _ in range(iterations):
                # Étape 3 : Créer une copie du plateau
                plateau_copie = self.plateau.copie_sans_surface()

                # Étape 4 : Générer les positions initiales à partir du plateau copié
                positions_initiales = []
                for y, ligne in enumerate(plateau_copie.cases):
                    for x, p in enumerate(ligne):
                        if p:
                            positions_initiales.append((p.couleur, (x, y)))

                # Étape 5 : Créer une instance avec les positions initiales
                instance = Instance(positions_initiales)

                # Étape 6 : Appliquer le mouvement initial
                instance.plateau.deplacer_pion(Mouvement(pion.position, mouvement))

                # Étape 7 : Simuler la partie jusqu'à la fin
                gagnant = instance.run_to_the_end()
                if gagnant == self.couleur_joueur:
                    victoire_joueur += 1
            return (pion.position, mouvement, victoire_joueur)

        # Étape 2 : Préparer les tâches pour chaque pion et mouvement
        tasks = []
        for pion in pions:
            mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=capture_possible)
            for mouvement in mouvements:
                tasks.append((pion, mouvement))

        # Étape 3 : Exécuter les simulations en parallèle
        with ProcessPoolExecutor() as executor:
            results = executor.map(lambda task: simuler_mouvement(*task), tasks)

        # Étape 4 : Collecter les résultats
        for position, mouvement, victoires in results:
            meilleurs_resultats[(position, mouvement)] = victoires

        # Trouver le mouvement avec le maximum de victoires
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