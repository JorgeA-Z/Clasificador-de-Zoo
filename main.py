import clasificador as cl

class MainUserInterface:

    def __init__(self) -> None:
        self.zoo = cl.Clasificador()
        self.zoo.read_Dataset()

    def Text(self):
        print("-------------------------")
        print("1. Regresión logística")
        print("2. K-Vecinos Cercanos")
        print("3. Maquinas Vector Soporte")
        print("4. Naive Bayes")
        print("5. Red Neuronal")
        print("6. Salir")
        print("-------------------------")

        
    def Menu(self):
        llave = True

        while llave:
            self.Text();

            try:
                op = int(input("Ingresa una opcion: "))

            except Exception as e:
                print(f"Error: {e}\n\n\n\n\n\n\n\n\n\n\n\n")

            else:
                if op == 1:
                    #Regresión logistica
                    self.zoo.Logistic_Regression()

                elif op == 2:
                    #K vecinos
                    self.zoo.K_Nearest_Neighbors()



                elif op == 3:
                    #Maquinas vector soporte
                    self.zoo.Support_Vector_Machines()


                elif op == 4:
                    #Naive Bayes
                    self.zoo.Naive_Bayes()

                elif op == 5:

                    self.zoo.redNeuronal()


                elif op == 6:
                    
                    llave = False

                    print("Gracias por usar el programa")

                else:
                    print("Opcion invalida\n\n\n\n\n\n\n\n\n\n\n\n")

if __name__ == "__main__":
    menu = MainUserInterface()

    menu.Menu()