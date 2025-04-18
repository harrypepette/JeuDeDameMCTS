import math
import random
import concurrent.futures
from Mouvement import Mouvement
from copy import deepcopy
from Node import Node
from Instance import SimulationAvancee
from Plateau import Plateau
from Dame import Dame
from Regles import Regles

class Simulation:
    def __init__(self, plateau, couleur_joueur):
        """
        Initialise la simulation MCTS.
        :param plateau: L'état actuel du plateau.
        :param couleur_joueur: La couleur du joueur à simuler ("noir" ou "blanc").
        """
        self.plateau = plateau
        self.couleur_joueur = couleur_joueur
        self.racine = None
        # Table de transposition pour stocker les états déjà explorés
        self.table_transposition = {}
    
    def hash_plateau(self, plateau):
        """Version plus rapide du hash"""
        # Créer un tableau de bits au lieu d'une liste de chaînes
        hash_value = 0
        for y in range(8):
            for x in range(8):
                pion = plateau.get_pion(x, y)
                if pion:
                    # Position (6 bits) + couleur (1 bit) + dame (1 bit)
                    position_bits = (y * 8 + x) & 0x3F
                    couleur_bit = 1 if pion.couleur == "blanc" else 0
                    dame_bit = 1 if hasattr(pion, 'est_dame') and pion.est_dame else 0
                    
                    # Combiner les bits et ajouter au hash
                    pion_bits = (position_bits << 2) | (couleur_bit << 1) | dame_bit
                    hash_value ^= (1 << pion_bits)
        
        return hash_value
    
    def _selection(self, noeud):
        """
        Sélectionne un nœud à explorer selon la politique UCT.
        :param noeud: Le nœud racine.
        :return: Le nœud sélectionné.
        """
        while not noeud.est_feuille() and not noeud.mouvements_non_explores:
            # Vérifier dans la table de transposition pour chaque enfant
            for enfant in noeud.enfants:
                hash_etat = self.hash_plateau(enfant.plateau)
                if hash_etat in self.table_transposition:
                    stats = self.table_transposition[hash_etat]
                    # Mettre à jour les statistiques avec celles de la table
                    enfant.visites = max(enfant.visites, stats['visites'])
                    enfant.victoires = max(enfant.victoires, stats['victoires'])
            
            # Sélectionner le meilleur enfant selon UCT
            noeud = max(noeud.enfants, key=lambda n: n.calculer_uct())
            
        return noeud
    
    def _expansion(self, noeud):
        """
        Étend un nœud en ajoutant un enfant.
        :param noeud: Le nœud à étendre.
        :return: Le nouveau nœud enfant.
        """
        if not noeud.mouvements_non_explores:
            return noeud
        
        # Choisir un mouvement non exploré
        mouvement = random.choice(noeud.mouvements_non_explores)
        noeud.mouvements_non_explores.remove(mouvement)
        
        # Créer une copie du plateau et appliquer le mouvement
        plateau_copie = noeud.plateau.copie_sans_surface()
        mouv = Mouvement(mouvement[0], mouvement[1])
        plateau_copie.deplacer_pion(mouv)
        
        # Calculer le hash du nouvel état
        hash_etat = self.hash_plateau(plateau_copie)
        
        # Créer le nouveau nœud
        enfant = Node(plateau_copie, mouvement, noeud)
        
        # Vérifier si cet état existe déjà dans la table de transposition
        if hash_etat in self.table_transposition:
            stats = self.table_transposition[hash_etat]
            # Initialiser avec les statistiques existantes
            enfant.visites = stats['visites']
            enfant.victoires = stats['victoires']
        
        # Ajout de la condition pour les dames
        pos_x, pos_y = mouvement[0]
        pion = noeud.plateau.get_pion(pos_x, pos_y)
        if pion and hasattr(pion, 'est_dame') and pion.est_dame:
            # Traitement spécial pour les dames
            pass
        
        noeud.enfants.append(enfant)
        self._initialiser_mouvements_non_explores(enfant)
        return enfant
    
    def _backpropagation(self, noeud, resultat):
        """
        Rétropropage le résultat de la simulation dans l'arbre.
        :param noeud: Le nœud à partir duquel rétropropager.
        :param resultat: Le résultat de la simulation (1 pour victoire, 0 pour défaite).
        """
        while noeud is not None:
            noeud.visites += 1
            noeud.victoires += resultat
            
            # Mettre à jour la table de transposition
            hash_etat = self.hash_plateau(noeud.plateau)
            self.table_transposition[hash_etat] = {
                'visites': noeud.visites,
                'victoires': noeud.victoires
            }
            
            noeud = noeud.parent
    
    def simuler_meilleur_mouvement(self, iterations=5000):
        """
        Exécute MCTS pour trouver le meilleur mouvement avec early stopping.
        :param iterations: Nombre maximum d'itérations MCTS.
        :return: Le meilleur mouvement.
        """
        # Vérifier s'il n'y a qu'un seul mouvement possible
        pions = self._get_pions_par_couleur(self.plateau)
        capture_possible = any(self.plateau.peut_manger(pion) for pion in pions)
        
        mouvements_possibles = []
        for pion in pions:
            if capture_possible and not self.plateau.peut_manger(pion):
                continue
            moves = self.plateau.mouvements_possibles(pion, forcer_manger=capture_possible)
            for move in moves:
                mouvements_possibles.append((pion.position, move))
        
        # S'il n'y a qu'un seul mouvement, le retourner immédiatement
        if len(mouvements_possibles) == 1:
            print("Un seul mouvement possible, retour immédiat")
            return mouvements_possibles[0]
        
        # Si capture obligatoire et un seul pion peut capturer
        if capture_possible:
            pions_pouvant_capturer = [p for p in pions if self.plateau.peut_manger(p)]
            if len(pions_pouvant_capturer) == 1:
                # Si ce pion n'a qu'une seule façon de capturer
                moves = self.plateau.mouvements_possibles(pions_pouvant_capturer[0], True)
                if len(moves) == 1:
                    move = (pions_pouvant_capturer[0].position, moves[0])
                    print("Une seule capture possible, retour immédiat")
                    return move
        
        # Initialisation
        self.racine = Node(self.plateau.copie_sans_surface())
        self._initialiser_mouvements_non_explores(self.racine)
        
        # Paramètres pour early stopping
        verification_interval = 5  # Vérifier plus fréquemment (était 10)
        iterations_minimum = 20    # Réduire le minimum d'itérations (était 30)
        seuil_confiance = 0.7      # Seuil de confiance plus bas (était 0.8)
        dominance_requise = 1      # Une seule vérification dominante suffit (était 2)
        
        iteration = 0
        meilleur_precedent = None
        dominance_count = 0       # Compteur de dominance consécutive
        
        while iteration < iterations:
            # Exécuter une itération complète de MCTS
            noeud = self._selection(self.racine)
            if noeud.mouvements_non_explores:
                noeud = self._expansion(noeud)
            resultat = self._simulation(noeud)
            self._backpropagation(noeud, resultat)
            
            iteration += 1
            
            # Vérifier si on peut arrêter tôt (après un minimum d'itérations)
            if iteration >= iterations_minimum and iteration % verification_interval == 0:
                if self.racine.enfants:
                    # Utiliser le ratio victoires/visites pour trouver le meilleur mouvement
                    meilleur_enfant = max(self.racine.enfants, 
                                         key=lambda n: n.victoires/n.visites if n.visites > 0 else 0)
                    total_visites = sum(n.visites for n in self.racine.enfants)
                    
                    # Calculer la proportion de visites du meilleur mouvement
                    if total_visites > 0:
                        dominance = meilleur_enfant.visites / total_visites
                        
                        # Vérifier si le même mouvement reste dominant
                        if meilleur_precedent == meilleur_enfant.mouvement:
                            dominance_count += 1
                        else:
                            dominance_count = 0
                            meilleur_precedent = meilleur_enfant.mouvement
                        
                        # Si le mouvement est clairement dominant
                        if dominance > seuil_confiance and dominance_count >= dominance_requise:
                            print(f"Early stopping à l'itération {iteration}/{iterations}")
                            return meilleur_enfant.mouvement
        
        # Retourner le meilleur mouvement après toutes les itérations
        # en utilisant le ratio victoires/visites plutôt que juste les visites
        if self.racine.enfants:
            meilleur_enfant = max(self.racine.enfants, 
                                 key=lambda n: n.victoires/n.visites if n.visites > 0 else 0)
            
            # Afficher les statistiques du meilleur mouvement pour information
            taux_victoire = meilleur_enfant.victoires / meilleur_enfant.visites if meilleur_enfant.visites > 0 else 0
            print(f"Meilleur mouvement sélectionné: {meilleur_enfant.mouvement} "
                  f"(Taux de victoire: {taux_victoire:.2%}, Visites: {meilleur_enfant.visites})")
            
            return meilleur_enfant.mouvement
        
        print("Aucun mouvement valide trouvé!")
        return None

    def _initialiser_mouvements_non_explores(self, noeud):
        """
        Initialise la liste des mouvements non explorés pour un nœud.
        """
        pions = self._get_pions_par_couleur(noeud.plateau)
        capture_possible = any(noeud.plateau.peut_manger(pion) for pion in pions)
        mouvements = []
        for pion in pions:
            if capture_possible and not noeud.plateau.peut_manger(pion):
                continue
            moves = noeud.plateau.mouvements_possibles(pion, forcer_manger=capture_possible)
            for move in moves:
            # Valider chaque mouvement avant de l'ajouter
                mouvement = Mouvement(pion.position, move)
                if Regles.mouvement_valide(pion, mouvement, noeud.plateau):
                    mouvements.append((pion.position, move))
                else:
                    print(f"Mouvement invalide filtré: {pion.position} -> {move}")
        noeud.mouvements_non_explores = mouvements

    def _simulation(self, noeud):
        """Version optimisée avec profondeur limitée"""
        # Limiter la profondeur de simulation
        max_depth = 40  # Maximum 40 coups par joueur
        
        # Créer une instance à partir de l'état du plateau
        positions_initiales = []
        for y, ligne in enumerate(noeud.plateau.cases):
            for x, p in enumerate(ligne):
                if p:
                    positions_initiales.append((p.couleur, (x, y)))

        # Utiliser la simulation avancée avec profondeur limitée
        simulation = SimulationAvancee(positions_initiales)
        gagnant = simulation.run_limited_depth(max_depth)  # Nouvelle méthode à implémenter
        return 1 if gagnant == self.couleur_joueur else 0

    def _get_pions_par_couleur(self, plateau):
        """
        Récupère tous les pions de la couleur sélectionnée sur le plateau.
        """
        pions = []
        for ligne in plateau.cases:
            for pion in ligne:
                if pion and pion.couleur == self.couleur_joueur:
                    pions.append(pion)
        return pions