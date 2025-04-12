import random
import math
import copy
from Mouvement import Mouvement
from Noeud import Noeud

class MCTS:
    def __init__(self, iterations=1000, constante_exploration=0.7):
        """
        Initialise l'algorithme MCTS.
        :param iterations: Nombre d'itérations à exécuter
        :param constante_exploration: Constante pour ajuster l'exploration vs exploitation
        """
        self.iterations = iterations
        self.constante_exploration = constante_exploration
        self.racine = None
        
    def rechercher_meilleur_mouvement(self, etat):
        self.racine = Noeud(etat=etat.copier())
        joueur_couleur = etat.joueurs[etat.joueur_actuel].couleur
        self.racine.mouvements_non_explores = self.obtenir_mouvements_possibles(self.racine.etat, joueur_couleur)
        
        # Si aucun mouvement possible, retourner None
        if not self.racine.mouvements_non_explores:
            print(f"Aucun mouvement possible pour {joueur_couleur} dans rechercher_meilleur_mouvement")
            return None
        
        for _ in range(self.iterations):
            noeud = self.selection(self.racine)
            if not noeud.est_terminal() and not noeud.est_entierement_explore():
                noeud = self.expansion(noeud)
            resultat = self.simulation(noeud)
            self.retropropagation(noeud, resultat)
        
        if not self.racine.enfants:
            print(f"Aucun enfant créé pour {joueur_couleur}")
            return None
        
        meilleur_enfant = max(self.racine.enfants, key=lambda enfant: enfant.visites)
        self.racine = meilleur_enfant
        self.racine.parent = None
        return meilleur_enfant.mouvement

    def selection(self, noeud):
        while not noeud.est_terminal():
            if not noeud.est_entierement_explore():
                return noeud
            if not noeud.enfants:
                print("Aucun enfant disponible dans selection")
                return noeud
            noeud = max(noeud.enfants, key=lambda enfant: enfant.UCT(self.constante_exploration))
        return noeud
      

    def expansion(self, noeud):
        """
        Développe un nœud en ajoutant un enfant.
        :param noeud: Nœud à développer
        :return: Nœud enfant ajouté
        """
        # Si c'est la première fois qu'on visite ce nœud, initialiser les mouvements non explorés
        if noeud.mouvements_non_explores is None:
            joueur_couleur = noeud.etat.joueurs[noeud.etat.joueur_actuel].couleur
            noeud.mouvements_non_explores = self.obtenir_mouvements_possibles(noeud.etat, joueur_couleur)
        
        # S'il n'y a plus de mouvements à explorer, retourner le nœud tel quel
        if not noeud.mouvements_non_explores:
            return noeud
        
        # Choisir un mouvement aléatoire parmi les non explorés
        pion, mouvement_arrivee = random.choice(noeud.mouvements_non_explores)
        noeud.mouvements_non_explores.remove((pion, mouvement_arrivee))
        
        # Créer un nouvel état pour l'enfant
        nouvel_etat = noeud.etat.copier()
        mouvement = Mouvement(pion.position, mouvement_arrivee)
        
        # Appliquer le mouvement
        nouvel_etat.plateau.deplacer_pion(mouvement)
        
        # Vérifier s'il y a une capture supplémentaire possible
        pion_apres_mouvement = nouvel_etat.plateau.get_pion(*mouvement_arrivee)
        capture_supplementaire = False
        
        if pion_apres_mouvement and nouvel_etat.plateau.peut_encore_manger(pion_apres_mouvement):
            # Ne pas changer de joueur, car le même joueur continue
            capture_supplementaire = True
        else:
            # Passer au joueur suivant
            nouvel_etat.joueur_actuel = 1 - nouvel_etat.joueur_actuel
        
        # Créer et ajouter le nœud enfant
        enfant = Noeud(etat=nouvel_etat, parent=noeud, mouvement=mouvement)
        noeud.enfants.append(enfant)
        
        return enfant

    def simulation(self, noeud):
        etat_simulation = noeud.etat.copier()
        joueur_depart = etat_simulation.joueur_actuel
        couleur_depart = etat_simulation.joueurs[joueur_depart].couleur
        
        est_terminee, gagnant = etat_simulation.plateau.partie_terminee()
        profondeur = 0
        max_profondeur = 10  # Limiter pour accélérer
        
        while not est_terminee and profondeur < max_profondeur:
            joueur_couleur = etat_simulation.joueurs[etat_simulation.joueur_actuel].couleur
            mouvements_possibles = self.obtenir_mouvements_possibles(etat_simulation, joueur_couleur)
            
            if not mouvements_possibles:
                gagnant = "noir" if joueur_couleur == "blanc" else "blanc"
                break
            
            # Prioriser captures et promotions
            mouvements_prioritaires = []
            mouvements_normaux = []
            for pion, arrivee in mouvements_possibles:
                if etat_simulation.plateau.est_capture(pion, arrivee):
                    mouvements_prioritaires.append((pion, arrivee))
                elif pion.est_dame or etat_simulation.plateau.devient_dame(pion, arrivee):
                    mouvements_prioritaires.append((pion, arrivee))
                else:
                    mouvements_normaux.append((pion, arrivee))
            
            choix = random.choice(mouvements_prioritaires or mouvements_normaux)
            mouvement = Mouvement(choix[0].position, choix[1])
            etat_simulation.plateau.deplacer_pion(mouvement)
            
            pion_apres_mouvement = etat_simulation.plateau.get_pion(*mouvement.arrivee)
            if pion_apres_mouvement and etat_simulation.plateau.peut_encore_manger(pion_apres_mouvement):
                pass
            else:
                etat_simulation.joueur_actuel = 1 - etat_simulation.joueur_actuel
            
            est_terminee, gagnant = etat_simulation.plateau.partie_terminee()
            profondeur += 1
        
        if est_terminee:
            return 1 if gagnant == couleur_depart else 0
        
        # Évaluation rapide
        score = 0
        for ligne in etat_simulation.plateau.cases:
            for pion in ligne:
                if pion:
                    if pion.couleur == couleur_depart:
                        score += 1
                        if pion.est_dame:
                            score += 3
                    else:
                        score -= 1
                        if pion.est_dame:
                            score -= 3
        return 1 if score > 0 else 0

    def retropropagation(self, noeud, resultat):
        """
        Met à jour les statistiques des nœuds.
        :param noeud: Nœud à partir duquel commencer la rétropropagation
        :param resultat: Résultat de la simulation
        """
        while noeud is not None:
            noeud.visites += 1
            noeud.victoires += resultat
            noeud = noeud.parent
            # Inverser le résultat pour le point de vue du joueur opposé
            resultat = 1 - resultat

    def obtenir_mouvements_possibles(self, etat, joueur_couleur):
        """
        Obtient tous les mouvements possibles pour un joueur.
        :param etat: État du jeu
        :param joueur_couleur: Couleur du joueur ('blanc' ou 'noir')
        :return: Liste des mouvements possibles [(pion, position_arrivee), ...]
        """
        
       
        mouvements = etat.obtenir_mouvements_legaux()
        print(f"Mouvements possibles pour {joueur_couleur}: {mouvements}")  # Débogage
        return mouvements