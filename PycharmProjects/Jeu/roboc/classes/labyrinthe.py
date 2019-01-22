# -*-coding:Utf-8 -*
import random
"""Ce module contient la classe Labyrinthe."""

class Labyrinthe:

    """Classe représentant un labyrinthe.
        - robot
        - grille
        - sortie """

    def __init__(self, grille, robot, sortie):
        self.robots = [[],[]]
        self.grille = grille
        self.sortie = sortie
    def print_grille(self):
        """Fonction permettant d'afficher la grille"""
        for ligne in self.grille:
            print(''.join(ligne))
    def grilletochaine(self):
        self.grille[self.robots[0][0]][self.robots[0][1]] = self.robots[0][2]
        self.grille[self.robots[1][0]][self.robots[1][1]] = self.robots[1][2]
        chaine =b""
        for ligne in self.grille:
            chaine.__add__('.'.join(ligne).encode())
        return chaine
    def generer_robots(self):
        positions_valides = []
        hauteur_grille = 0
        while hauteur_grille < len(self.grille):
            largeur_grille = 0
            while largeur_grille < len(self.grille [0]):
                if self.grille[hauteur_grille][largeur_grille] == " ":
                    positions_valides.append([hauteur_grille, largeur_grille])
                largeur_grille+=1
            hauteur_grille +=1
        position_1 = random.randint(0, len(positions_valides))
        position_2 = position_1
        while position_2 == position_1:
            position_2 = random.randint(0, len(positions_valides)-1)
        positions_valides[position_1].append("X")
        positions_valides[position_2].append("x")
        self.robots[0] = positions_valides[position_1]
        self.robots[1] = positions_valides[position_2]
        self.grille[self.robots[0][0]][self.robots[0][1]] = self.robots[0][2]
        self.grille[self.robots[1][0]][self.robots[1][1]] = self.robots[1][2]