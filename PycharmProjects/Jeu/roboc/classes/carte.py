# -*-coding:Utf-8 -*

"""Ce module contient la classe Carte."""
from classes.labyrinthe import Labyrinthe


class Carte:

    """Objet de transition entre un fichier et un labyrinthe.
    - nom
    - labyrinthe """

    def __init__(self, nom, chaine):
        self.nom = nom
        self.labyrinthe = self.creer_labyrinthe(chaine)
        self.labyrinthe.generer_robots()
    def __repr__(self):
        return "<Carte {}>".format(self.nom)
    def creer_labyrinthe(cls, chaine):
        """ Fonction permeetant de crée une Labyrinthe à partir d'une chaine de caractére
        - chaine : une chaine de caractére
        """
        robot = []
        liste_lignes = chaine.split("\n")
        matrix = []
        sortie_find = False
        for ligne in liste_lignes:
            matrix.append(list(ligne))
            compteur_parcour = 0
            while sortie_find == False and compteur_parcour < len(ligne):
                if ligne[compteur_parcour] == "U":
                    sortie = (len(matrix) -1, compteur_parcour)
                compteur_parcour += 1
        labyrinthe = Labyrinthe (matrix, robot, sortie)
        return labyrinthe
