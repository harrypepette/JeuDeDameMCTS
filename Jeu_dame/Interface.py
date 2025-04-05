from abc import ABC, abstractmethod

class Interface(ABC):
    """
    Classe abstraite représentant une interface pour le jeu de dames.
    """

    @abstractmethod
    def afficher_plateau(self, plateau):
        """
        Méthode abstraite pour afficher le plateau de jeu.
        :param plateau: Le plateau de jeu à afficher.
        """
        pass

    @abstractmethod
    def demander_mouvement(self):
        """
        Méthode abstraite pour demander un mouvement au joueur.
        :return: Les coordonnées de départ et d'arrivée du mouvement.
        """
        pass

    @abstractmethod
    def afficher_message(self, message):
        """
        Méthode abstraite pour afficher un message à l'utilisateur.
        :param message: Le message à afficher.
        """
        pass