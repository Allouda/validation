
class Partie:

    """Objet permettant de sauvegarder une partie
        - nom
        - labyrinthe
        - is_porte
        - sortie"""

    def __init__(self, nom, labyrinthe):
        self.nom = nom
        self.labyrinthe = labyrinthe
        self.is_porte = False
        self.sortie = False
    def deplacement(self, chaine_deplacement, liste_reponses):
        """Fonction permettant de deplacer le robot dans Labyrinthe
        - chaine_deplacement : une chane de caractére contenant la direction et la longeur de deplacement
        """
        for reponse in liste_reponses:
            direction = reponse[0]
            nombre_deplacement = reponse[1]
            if self.verifier_deplacement(direction, nombre_deplacement):
                if self.is_porte:
                    self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] = "."
                    self.is_porte = False
                else:
                    self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] = " "
                if direction == "s":
                    self.labyrinthe.robot[0] += nombre_deplacement
                elif direction == "n":
                    self.labyrinthe.robot[0] -= nombre_deplacement
                elif direction == "o":
                    self.labyrinthe.robot[1] -= nombre_deplacement
                elif direction == "e":
                    self.labyrinthe.robot[1] += nombre_deplacement
                if self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] == ".":
                    self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] = "X"
                    self.is_porte = True
                    self.labyrinthe.print_grille()
                    return False
                elif self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] == "U":
                    self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] = "X"
                    self.labyrinthe.print_grille()
                    return True
                else:
                    self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1]] = "X"
                    self.labyrinthe.print_grille()
                    return False
    def verifier_deplacement(self, direction, nombre_deplacement):
        """ Fonction permettant de vérifier le deplacement demandé par l'utilisateur
        - direction : la direction de deplacement ; s, n, o, e
        - nombre_deplacement : les nombre des pas
        """
        # verifier la direction
        if direction not in {"s", "e", "n", "o"}:
            print("choix incorrecte!")
            return False
        # verifier la longeur de deplacement, l'emplacement apres le deplacement et le trajet de deplacement

        if direction == "s":
            if nombre_deplacement > len(self.labyrinthe.grille)- self.labyrinthe.robot[0]:
                print("Le deplacement depasse la labyrinthe!")
                return False
            elif self.labyrinthe.grille[self.labyrinthe.robot[0] + nombre_deplacement][self.labyrinthe.robot[1]] == "O":
                print("L'emplacement aprés le deplacement non valide!")
                return False
            else:
                trajet = True
                compteur = 0
                while trajet and compteur < nombre_deplacement:
                    if self.labyrinthe.grille[self.labyrinthe.robot[0] + compteur] [self.labyrinthe.robot[1]]== "O":
                        trajet = False
                    compteur+=1
                return trajet
        elif direction == "n":
            if nombre_deplacement > self.labyrinthe.robot[0]:
                print("Le deplacement depasse la labyrinthe!")
                return False
            elif self.labyrinthe.grille[self.labyrinthe.robot[0]- nombre_deplacement][self.labyrinthe.robot[1]] == "O":
                print("L'emplacement aprés le deplacement non valide!")
                return False
            else:
                trajet = True
                compteur = 0
                while trajet and compteur < nombre_deplacement:
                    if self.labyrinthe.grille[self.labyrinthe.robot[0] - compteur][self.labyrinthe.robot[1]] == "O":
                        trajet = False
                    compteur += 1
                return trajet
        elif direction == "o":
            if nombre_deplacement > self.labyrinthe.robot[1]:
                print("Le deplacement depasse la labyrinthe!")
                return False
            elif self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1] - nombre_deplacement] == "O":
                print("L'emplacement aprés le deplacement non valide!")
                return False
            else:
                trajet = True
                compteur = 0
                while trajet and compteur < nombre_deplacement:
                    if self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1] - compteur] == "O":
                        trajet = False
                    compteur += 1
                return trajet
        elif direction == "e":
            if nombre_deplacement > len(self.labyrinthe.grille[0]) - self.labyrinthe.robot[1]:
                print("Le deplacement depasse la labyrinthe!")
                return False
            elif self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1] + nombre_deplacement] == "O":
                print("L'emplacement aprés le deplacement non valide!")
                return False
            else:
                trajet = True
                compteur = 0
                while trajet and compteur < nombre_deplacement:
                    if self.labyrinthe.grille[self.labyrinthe.robot[0]][self.labyrinthe.robot[1] + compteur] == "O":
                        trajet = False
                    compteur += 1
                return trajet
