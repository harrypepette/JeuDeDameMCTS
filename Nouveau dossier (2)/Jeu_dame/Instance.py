import random
from Plateau import Plateau
from Joueur import Joueur
from Mouvement import Mouvement
from Pion import Pion

class SimulationAvancee:
    def __init__(self, positions_initiales):
        """
        Initialise une instance du jeu avec des positions spécifiques.
        :param positions_initiales: Liste des positions des pions (format : [(couleur, (x, y)), ...]).
        """
        self.plateau = Plateau()
        self.plateau.cases = [[None for _ in range(8)] for _ in range(8)]  # Réinitialiser le plateau
        for couleur, position in positions_initiales:
            x, y = position
            if couleur == "blanc":
                self.plateau.cases[y][x] = Pion("blanc", position)
            elif couleur == "noir":
                self.plateau.cases[y][x] = Pion("noir", position)
        self.joueurs = [Joueur("blanc"), Joueur("noir")]
        self.joueur_actuel = 0

    def evaluer_mouvement(self, pion, mouvement, forcer_manger=False):
        """
        Évalue la qualité d'un mouvement selon plusieurs critères améliorés.
        Plus le score est élevé, meilleur est le coup.
        """
        score = 0
        x_depart, y_depart = pion.position
        x_arrivee, y_arrivee = mouvement
        
        # 1. Prioriser les captures (score élevé)
        if abs(x_arrivee - x_depart) >= 2 and abs(y_arrivee - y_depart) >= 2:
            score += 100  # Score élevé pour les captures
            
            # Bonus pour les captures de dames
            milieu_x = (x_depart + x_arrivee) // 2
            milieu_y = (y_depart + y_arrivee) // 2
            pion_capture = self.plateau.get_pion(milieu_x, milieu_y)
            if pion_capture and pion_capture.est_dame:
                score += 50  # Bonus pour capturer une dame
        
        # 2. Favoriser les mouvements vers la promotion
        if not pion.est_dame:
            if (pion.couleur == "blanc" and y_arrivee < y_depart):
                score += 5  # Favoriser l'avancée vers la promotion
                # Bonus plus important quand on approche de la promotion
                if y_arrivee == 0:  # Position de promotion pour les blancs
                    score += 30
                elif y_arrivee == 1:  # À une case de la promotion
                    score += 15
            elif (pion.couleur == "noir" and y_arrivee > y_depart):
                score += 5  # Favoriser l'avancée vers la promotion
                # Bonus plus important quand on approche de la promotion
                if y_arrivee == 7:  # Position de promotion pour les noirs
                    score += 30
                elif y_arrivee == 6:  # À une case de la promotion
                    score += 15
        
        # 3. AMÉLIORATION: Évaluation améliorée de la vulnérabilité (spécifique pour les dames)
        plateau_temp = self.plateau.copie_sans_surface()
        plateau_temp.deplacer_pion(Mouvement(pion.position, mouvement))
        pion_deplace = plateau_temp.get_pion(*mouvement)
        
        if pion_deplace:
            # Vérifier si le pion peut être capturé après le mouvement
            couleur_adverse = "noir" if pion.couleur == "blanc" else "blanc"
            peut_etre_capture = False
            
            # Calcul amélioré pour la vulnérabilité
            for y in range(8):
                for x in range(8):
                    pion_adverse = plateau_temp.get_pion(x, y)
                    if pion_adverse and pion_adverse.couleur == couleur_adverse:
                        # Obtenir les positions de capture possibles pour le pion adverse
                        captures = plateau_temp.cases_fin_manger(pion_adverse)
                        
                        if pion_adverse.est_dame:
                            # AMÉLIORATION: Vérification spécifique pour les dames adverses
                            # Les dames peuvent capturer sur toute la diagonale
                            for capture in captures:
                                capture_x, capture_y = capture
                                dx = capture_x - x
                                dy = capture_y - y
                                
                                # Vérifier toutes les positions intermédiaires
                                distance = abs(dx)
                                direction_x = 1 if dx > 0 else -1
                                direction_y = 1 if dy > 0 else -1
                                
                                for i in range(1, distance):
                                    mid_x = x + (i * direction_x)
                                    mid_y = y + (i * direction_y)
                                    if (mid_x, mid_y) == mouvement:
                                        peut_etre_capture = True
                                        # Si le pion déplacé est une dame, pénalité plus élevée
                                        if pion_deplace.est_dame:
                                            score -= 60  # Pénalité plus sévère pour exposer une dame
                                        else:
                                            score -= 40  # Pénalité pour un pion normal
                                        break
                        else:
                            # Gestion standard pour les pions normaux
                            for capture in captures:
                                capture_x, capture_y = capture
                                dx = capture_x - x
                                dy = capture_y - y
                                
                                # Position intermédiaire qui serait capturée
                                mid_x = x + (dx // 2)
                                mid_y = y + (dy // 2)
                                if (mid_x, mid_y) == mouvement:
                                    peut_etre_capture = True
                                    if pion_deplace.est_dame:
                                        score -= 60
                                    else:
                                        score -= 40
                                    break
                        
                        if peut_etre_capture:
                            break
                if peut_etre_capture:
                    break
            
            # AMÉLIORATION: Vérifier les positions sécurisées (adossées au bord)
            if pion_deplace.est_dame:
                if x_arrivee == 0 or x_arrivee == 7 or y_arrivee == 0 or y_arrivee == 7:
                    score += 15  # Bonus pour une dame près du bord (plus difficile à capturer)
        
        # 4. AMÉLIORATION: Évaluation améliorée des dames
        if pion.est_dame:
            # Évaluation plus sophistiquée du positionnement des dames
            
            # Contrôle du centre avec valeur progressive
            centre_x = abs(3.5 - x_arrivee)  # Distance au centre horizontal (3.5)
            centre_y = abs(3.5 - y_arrivee)  # Distance au centre vertical (3.5)
            distance_centre = (centre_x + centre_y) / 2.0
            
            # Plus la dame est proche du centre, meilleur est le score
            score += max(0, 20 - (5 * distance_centre))
            
            # AMÉLIORATION: Évaluer la mobilité de la dame
            mobilite = self._calculer_mobilite(plateau_temp, x_arrivee, y_arrivee)
            score += mobilite * 3  # 3 points par case accessible
            
            # AMÉLIORATION: Évaluer le contrôle des diagonales
            controle_diagonales = self._evaluer_controle_diagonales(plateau_temp, x_arrivee, y_arrivee)
            score += controle_diagonales * 2
        
        return score
    
    def _calculer_mobilite(self, plateau, x, y):
        """
        NOUVELLE MÉTHODE: Calcule combien de cases la dame peut atteindre depuis sa position.
        """
        pion = plateau.get_pion(x, y)
        if not pion or not pion.est_dame:
            return 0
            
        # Obtenir tous les mouvements possibles pour cette dame
        mouvements = plateau.mouvements_possibles(pion, forcer_manger=False)
        return len(mouvements)
    
    def _evaluer_controle_diagonales(self, plateau, x, y):
        """
        NOUVELLE MÉTHODE: Évalue le contrôle des diagonales par une dame.
        Retourne le nombre de cases contrôlées sur les diagonales principales.
        """
        pion = plateau.get_pion(x, y)
        if not pion or not pion.est_dame:
            return 0
            
        controle = 0
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # Parcourir chaque direction diagonale
        for dx, dy in directions:
            for dist in range(1, 8):
                nx, ny = x + (dx * dist), y + (dy * dist)
                # Si la position est valide et vide, la dame contrôle cette case
                if 0 <= nx < 8 and 0 <= ny < 8:
                    pion_sur_case = plateau.get_pion(nx, ny)
                    if pion_sur_case is None:  # Case vide
                        controle += 1
                    else:
                        # Si on rencontre un pion, on arrête dans cette direction
                        break
                else:
                    break
                    
        return controle

    # AMÉLIORATION: Gestion améliorée des captures multiples pour les dames
    def _evaluer_sequence_capture(self, pion, captures):
        """
        NOUVELLE MÉTHODE: Évalue une séquence de captures pour choisir la meilleure.
        """
        meilleures_captures = []
        for capture in captures:
            plateau_temp = self.plateau.copie_sans_surface()
            pion_temp = plateau_temp.get_pion(*pion.position)
            
            # Simuler la séquence de captures
            score_total = 0
            position_courante = pion.position
            
            # Pour les dames, prendre en compte toutes les pièces capturées
            for mouvement in capture:
                depart, arrivee = mouvement.depart, mouvement.arrivee
                score_mouvement = self.evaluer_mouvement(pion_temp, arrivee, True)
                score_total += score_mouvement
                
                # Mettre à jour la position pour la prochaine capture
                plateau_temp.deplacer_pion(mouvement)
                position_courante = arrivee
                pion_temp = plateau_temp.get_pion(*position_courante)
            
            meilleures_captures.append((capture, score_total))
        
        # Trier par score et retourner la meilleure séquence
        if meilleures_captures:
            meilleures_captures.sort(key=lambda x: x[1], reverse=True)
            return meilleures_captures[0][0]
        return []

    def run_to_the_end(self):
        """
        Lance le jeu avec des mouvements semi-aléatoires jusqu'à ce qu'un joueur gagne.
        Retourne la couleur du gagnant.
        """
        max_tours = 200  # Limite pour éviter les parties infinies
        tour_actuel = 0
        
        while tour_actuel < max_tours:
            tour_actuel += 1
            joueur_couleur = self.joueurs[self.joueur_actuel].couleur
            pions = [
                pion for ligne in self.plateau.cases for pion in ligne
                if pion and pion.couleur == joueur_couleur
            ]

            # Trouver tous les mouvements possibles pour les pions du joueur actuel
            mouvements_possibles = []
            
            # D'abord vérifier s'il y a des captures possibles
            captures_possibles = []
            for pion in pions:
                captures = self.plateau.cases_fin_manger(pion)
                if captures:
                    for capture in captures:
                        captures_possibles.append((pion, capture, self.evaluer_mouvement(pion, capture, True)))
            
            # Si des captures sont possibles, les prioriser obligatoirement
            if captures_possibles:
                mouvements_possibles = captures_possibles
            else:
                # Sinon, considérer tous les mouvements possibles
                for pion in pions:
                    mouvements = self.plateau.mouvements_possibles(pion, False)
                    for mouvement in mouvements:
                        score = self.evaluer_mouvement(pion, mouvement, False)
                        mouvements_possibles.append((pion, mouvement, score))

            if not mouvements_possibles:
                # Si aucun mouvement n'est possible, l'autre joueur gagne
                gagnant = "noir" if joueur_couleur == "blanc" else "blanc"
                return gagnant

            # Trier les mouvements par score et ajouter un élément aléatoire
            # pour éviter que l'IA joue toujours les mêmes coups
            mouvements_possibles.sort(key=lambda x: x[2], reverse=True)
            
            # Prendre les 3 meilleurs mouvements si disponibles, sinon tous
            top_n = min(3, len(mouvements_possibles))
            meilleurs_mouvements = mouvements_possibles[:top_n]
            
            # Choisir aléatoirement parmi les meilleurs mouvements
            # avec une probabilité plus élevée pour les meilleurs scores
            poids = [max(1, mvt[2]) for mvt in meilleurs_mouvements]
            total = sum(poids)
            if total > 0:  # Éviter division par zéro
                poids = [p/total for p in poids]
            else:
                poids = [1/len(meilleurs_mouvements)] * len(meilleurs_mouvements)
            
            pion, mouvement, _ = random.choices(meilleurs_mouvements, weights=poids, k=1)[0]
            
            # Déplacer le pion
            self.plateau.deplacer_pion(Mouvement(pion.position, mouvement))
            
            # Gérer les captures multiples
            pion_deplace = self.plateau.get_pion(*mouvement)
            while pion_deplace and self.plateau.peut_encore_manger(pion_deplace):
                captures = self.plateau.cases_fin_manger(pion_deplace)
                if not captures:
                    break
                    
                # Évaluer les captures suivantes
                captures_evaluees = [(capture, self.evaluer_mouvement(pion_deplace, capture, True)) 
                                    for capture in captures]
                captures_evaluees.sort(key=lambda x: x[1], reverse=True)
                
                # Prendre la meilleure capture
                meilleure_capture = captures_evaluees[0][0]
                self.plateau.deplacer_pion(Mouvement(pion_deplace.position, meilleure_capture))
                pion_deplace = self.plateau.get_pion(*meilleure_capture)

            # Vérifier si la partie est terminée
            est_terminee, gagnant = self.plateau.partie_terminee()
            if est_terminee:
                return gagnant

            # Passer au joueur suivant
            self.joueur_actuel = 1 - self.joueur_actuel
            
        # Si le nombre maximum de tours est atteint, déterminer le gagnant
        # par le nombre de pièces
        pions_blancs = sum(1 for ligne in self.plateau.cases for pion in ligne 
                          if pion and pion.couleur == "blanc")
        pions_noirs = sum(1 for ligne in self.plateau.cases for pion in ligne 
                         if pion and pion.couleur == "noir")
        
        if pions_blancs > pions_noirs:
            return "blanc"
        elif pions_noirs > pions_blancs:
            return "noir"
        else:
            # En cas d'égalité, retourner une couleur aléatoire
            return random.choice(["blanc", "noir"])

    def run_limited_depth(self, max_depth):
        """
        Version optimisée qui limite la profondeur de simulation
        """
        depth = 0
        while depth < max_depth:
            # Vérifier fin de partie
            est_terminee, gagnant = self.plateau.partie_terminee()
            if est_terminee:
                return gagnant
                
            # Obtenir les mouvements possibles
            joueur_actuel = "blanc" if self.joueur_actuel == 0 else "noir"
            pions = [p for ligne in self.plateau.cases for p in ligne if p and p.couleur == joueur_actuel]
            
            # Si plus de pions, partie terminée
            if not pions:
                return "noir" if joueur_actuel == "blanc" else "blanc"
                
            # Faire un mouvement aléatoire
            # Version simplifiée pour la vitesse
            self._faire_mouvement_simple()
            depth += 1
            
        # Si max_depth atteint, évaluer la position finale
        return self._evaluer_position_finale()

    def _faire_mouvement_simple(self):
        """
        Effectue un mouvement simple et rapide pour les simulations
        de profondeur limitée.
        """
        # Obtenir le joueur actuel
        joueur_actuel = "blanc" if self.joueur_actuel == 0 else "noir"
        
        # Récupérer tous les pions du joueur actuel
        pions = [p for y in range(8) for x in range(8) 
                 if (p := self.plateau.get_pion(x, y)) and p.couleur == joueur_actuel]
        
        # S'il n'y a pas de pions, retourner
        if not pions:
            return
        
        # Vérifier s'il y a des captures possibles
        pions_capturants = [p for p in pions if self.plateau.peut_manger(p)]
        
        if pions_capturants:
            # Choisir un pion aléatoire parmi ceux qui peuvent capturer
            pion = random.choice(pions_capturants)
            # Obtenir les mouvements de capture possibles
            mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=True)
        else:
            # Filtrer les pions qui peuvent bouger
            pions_mobiles = []
            for pion in pions:
                mouvements = self.plateau.mouvements_possibles(pion, forcer_manger=False)
                if mouvements:
                    pions_mobiles.append((pion, mouvements))
            
            # S'il n'y a pas de pions qui peuvent bouger, retourner
            if not pions_mobiles:
                return
            
            # Choisir un pion et un mouvement aléatoire
            pion, mouvements = random.choice(pions_mobiles)
        
        # Choisir un mouvement aléatoire
        if mouvements:
            mouvement = random.choice(mouvements)
            # Créer et effectuer le mouvement
            mvt = Mouvement(pion.position, mouvement)
            self.plateau.deplacer_pion(mvt)
            
            # Gérer les captures multiples
            pion_deplace = self.plateau.get_pion(*mouvement)
            if pion_deplace and self.plateau.peut_encore_manger(pion_deplace):
                prochaines_positions = self.plateau.cases_fin_manger(pion_deplace)
                if prochaines_positions:
                    position = random.choice(prochaines_positions)
                    mvt = Mouvement(pion_deplace.position, position)
                    self.plateau.deplacer_pion(mvt)
        
        # Passer au joueur suivant
        self.joueur_actuel = 1 - self.joueur_actuel

    def _evaluer_position_finale(self):
        """
        Évalue la position finale pour déterminer le gagnant probable
        quand la profondeur maximale est atteinte.
        """
        # Compter les pièces
        pieces_blanc = 0
        pieces_noir = 0
        dames_blanc = 0
        dames_noir = 0
        
        for y in range(8):
            for x in range(8):
                pion = self.plateau.get_pion(x, y)
                if pion:
                    if pion.couleur == "blanc":
                        pieces_blanc += 1
                        if hasattr(pion, 'est_dame') and pion.est_dame:
                            dames_blanc += 1
                    else:
                        pieces_noir += 1
                        if hasattr(pion, 'est_dame') and pion.est_dame:
                            dames_noir += 1
        
        # Calculer un score pondéré (une dame vaut 3 pions)
        score_blanc = pieces_blanc + 3 * dames_blanc
        score_noir = pieces_noir + 3 * dames_noir
        
        # Déterminer le gagnant probable
        if score_blanc > score_noir:
            return "blanc"
        elif score_noir > score_blanc:
            return "noir"
        else:
            # En cas d'égalité, retourner le joueur qui n'est pas actuel
            # (supposant qu'avoir le coup est un avantage)
            return "noir" if self.joueur_actuel == 0 else "blanc"