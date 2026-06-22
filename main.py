#Programa principal para el uso del sistema de clasificación visual con aprendizaje por refuerzo (RL)
import os
import color_rl_system


if __name__ == "__main__":
    print("Bienvenido al sistema de clasificación visual con aprendizaje por refuerzo (RL)!")
    while True:
        print("Seleccione la opcion que desea utilizar:")
        print("1. Entrenar el modelo con datos de entrenamiento")
        print("2. Utilizar el modelo entrenado para clasificar colores en tiempo real")
        print("3. Ver metricas de entrenamiento y resultados")
        print("4. Salir")
        #Verificacion de coneccion de la WonderMV antes de ejecutar cualquier opcion
        puerto = color_rl_system.detectar_puerto_wondermv()
        if puerto:
            print(f"✅ WonderMV detectada en {puerto} — lista para entrenar o clasificar")
        else:
            print("⚠ WonderMV no encontrada — por favor conecte la cámara para usar el sistema")
            break
        opcion = input("Ingrese el numero de la opcion deseada: ")
        if opcion == "1":
            print("Ha seleccionado entrenar el modelo")
            print("Cuantos episodios de entrenamiento desea realizar? (Default: 100)")
            try:
                episodios = int(input("Ingrese el numero de episodios: "))
            except ValueError:
                print("Entrada no valida. Se usara el valor por defecto de 100 episodios.")
                episodios = 100
            while True:
                nombre_modelo = input("Que nombre desea darle al modelo entrenado: ")
                #vefidicamos que el nombre del modelo no este vacio
                if not nombre_modelo:
                    print("El nombre del modelo no puede estar vacio.")
                #verificamos que el nombre del modelo no este repetido ya
                elif os.path.isfile(f"modelos_entrenados/{nombre_modelo}.pth"):
                    print("Ya existe un modelo con ese nombre. Por favor, elija otro nombre.")
                #si el nombre es apropiado salimos del ciclo
                else:
                    break
            #Ejecutamos el script color_rl_system.py para entrenar el modelo
            color_rl_system.entrenar(episodios = episodios, guardar_como = nombre_modelo)
        elif opcion == "2":
            print("Utilizando el modelo entrenado para clasificar colores en tiempo real...")
            #Cargamos el modelo entrenado desde la carpeta modelos_entrenados
            modelos_disponibles = [f[:-4] for f in os.listdir("modelos_entrenados") if f.endswith(".pkl")]
            if not modelos_disponibles:
                print("No hay modelos entrenados disponibles. Por favor, entrene un modelo primero.")
                continue
            print("Modelos entrenados disponibles:")
            for i, modelo in enumerate(modelos_disponibles):
                print(f"{i+1}. {modelo}")
            while True:
                try:
                    seleccion = int(input("Seleccione el numero del modelo que desea utilizar: "))
                    if 1 <= seleccion <= len(modelos_disponibles):
                        modelo_seleccionado = modelos_disponibles[seleccion - 1]
                        break
                    else:
                        print(f"Por favor, ingrese un numero entre 1 y {len(modelos_disponibles)}.")
                except ValueError:
                    print("Entrada no valida. Por favor, ingrese un numero.")
            color_rl_system.inferencia(filepath=f"modelos_entrenados/{modelo_seleccionado}.pkl")

        elif opcion == "3":
            print("Generando gráficos de métricas de entrenamiento del último modelo...")
            color_rl_system.mostrar_metricas()
        elif opcion == "4":
            print("Saliendo del programa. ¡Hasta luego!")
            break
        else:
            print("Opcion no valida. Por favor, ingrese un numero del 1 al 4.")
