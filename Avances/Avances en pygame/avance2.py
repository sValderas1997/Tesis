import pygame
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Dimensiones de la ventana de simulación
WIDTH = 800
HEIGHT = 600

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Configuración del robot
RADIO_ROBOT = 20
COLOR_ROBOT = RED

# Parámetros del controlador PID
KP_RANGE = (0.1, 1.0)
KI_RANGE = (0.01, 0.1)
KD_RANGE = (0.05, 0.5)

# Función para generar un punto de muestreo aleatorio
def generar_punto_aleatorio():
    return (random.uniform(0, WIDTH), random.uniform(0, HEIGHT))

# Función para calcular la distancia euclidiana entre dos puntos
def calcular_distancia(coord1, coord2):
    return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)

# Función para mover el robot en función de las velocidades lineal y angular
def mover_robot(pos_actual, v_lineal, v_angular, delta_tiempo):
    x = pos_actual[0] + v_lineal * math.cos(pos_actual[2]) * delta_tiempo
    y = pos_actual[1] + v_lineal * math.sin(pos_actual[2]) * delta_tiempo
    theta = pos_actual[2] + v_angular * delta_tiempo
    return x, y, theta

# Función principal de simulación
def simulacion(coordenadas_objetivo):
    pygame.init()
    pantalla = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulación de Robot de 2 Ruedas")

    reloj = pygame.time.Clock()
    hecho = False

    fuente = pygame.font.Font(None, 24)  # Fuente para el texto
    color_texto = BLACK  # Color del texto

    indice_objetivo = 0
    pos_robot = coordenadas_objetivo[0] + (0.0,)  # Inicializa la posición del robot (x, y, theta)
    objetivo_alcanzado = False

    # Variables para almacenar los valores antiguos de velocidad
    v_lineal_antigua = 0.0
    v_angular_antigua = 0.0

    # Variables para almacenar los valores objetivo de velocidad
    v_lineal_objetivo = 0.0
    v_angular_objetivo = 0.0

    # Variables para almacenar los datos de velocidad
    tiempos = []
    velocidades_lineal = []
    velocidades_angular = []

    while not hecho:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                hecho = True

        if not objetivo_alcanzado:
            # Calcula el error de posición entre la posición actual y el objetivo actual
            error_x = coordenadas_objetivo[indice_objetivo][0] - pos_robot[0]
            error_y = coordenadas_objetivo[indice_objetivo][1] - pos_robot[1]

            # Calcula las velocidades lineal y angular objetivo utilizando el controlador PID
            v_lineal_objetivo = KP * math.sqrt(error_x ** 2 + error_y ** 2)
            orientacion_objetivo = math.atan2(error_y, error_x)
            error_theta = orientacion_objetivo - pos_robot[2]
            error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))  # Limita el error entre -pi y pi
            v_angular_objetivo = KI * error_theta + KD * error_theta

            # Limitar la velocidad lineal a un máximo de 1
            v_lineal_objetivo = min(v_lineal_objetivo, 1.0)

            # Limitar la velocidad angular entre -1 y 1
            v_angular_objetivo = max(min(v_angular_objetivo, 1.0), -1.0)

            # Suaviza las transiciones de velocidad utilizando ramping
            ramping_factor = 0.1  # Ajusta este valor para controlar la suavidad de las transiciones
            v_lineal = v_lineal_antigua + (v_lineal_objetivo - v_lineal_antigua) * ramping_factor
            v_angular = v_angular_antigua + (v_angular_objetivo - v_angular_antigua) * ramping_factor

            # Actualiza los valores antiguos de velocidad
            v_lineal_antigua = v_lineal
            v_angular_antigua = v_angular

            # Añadir las velocidades al registro
            tiempos.append(pygame.time.get_ticks() / 1000.0)  # Tiempo en segundos
            velocidades_lineal.append(v_lineal)
            velocidades_angular.append(v_angular)

            # Mueve el robot hacia el objetivo actual
            pos_robot = mover_robot(pos_robot, v_lineal, v_angular, 0.1)

            # Verifica si el robot ha alcanzado el objetivo actual
            if calcular_distancia((pos_robot[0], pos_robot[1]), coordenadas_objetivo[indice_objetivo]) < RADIO_ROBOT:
                indice_objetivo += 1
                if indice_objetivo >= len(coordenadas_objetivo):
                    objetivo_alcanzado = True
        else:
            # El robot ha alcanzado todos los objetivos
            break

        # Dibuja los puntos de las coordenadas objetivo con números
        pantalla.fill(WHITE)
        for i, coord in enumerate(coordenadas_objetivo):
            pygame.draw.circle(pantalla, BLUE, (int(coord[0]), int(coord[1])), 5)
            # Agrega el número de la coordenada junto a la posición
            texto_coord = fuente.render(f"{i + 1}", True, BLACK)
            pantalla.blit(texto_coord, (int(coord[0]) + 10, int(coord[1]) - 10))

        # Dibuja el robot
        pygame.draw.circle(pantalla, COLOR_ROBOT, (int(pos_robot[0]), int(pos_robot[1])), RADIO_ROBOT)
        pygame.draw.line(
            pantalla,
            BLACK,
            (pos_robot[0], pos_robot[1]),
            (
                pos_robot[0] + RADIO_ROBOT * math.cos(pos_robot[2]),
                pos_robot[1] + RADIO_ROBOT * math.sin(pos_robot[2]),
            ),
            2,
        )

        # Muestra la velocidad lineal y angular en la pantalla
        texto = fuente.render(f"Velocidad: Lineal={v_lineal:.2f}, Angular={v_angular:.2f}", True, color_texto)
        pantalla.blit(texto, (10, 10))

        pygame.display.flip()

        reloj.tick(0)

    pygame.quit()

    # Graficar las velocidades
    plt.figure()
    plt.plot(tiempos, velocidades_lineal, label="Velocidad Lineal")
    plt.plot(tiempos, velocidades_angular, label="Velocidad Angular")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad")
    plt.title("Velocidades durante la simulación")
    plt.legend()
    plt.show()

# Función para la función objetivo utilizada en la optimización mediante PSO
def funcion_objetivo(ganancias, coordenadas_objetivo):
    # Obtiene las ganancias del controlador PID
    KP, KI, KD = ganancias

    # Inicializa la posición del robot
    pos_robot = coordenadas_objetivo[0] + (0.0,)  # Inicializa la posición del robot (x, y, theta)

    # Calcula el error cuadrático medio entre la posición objetivo y la posición actual del robot
    error_cuadratico_medio = 0.0
    for objetivo in coordenadas_objetivo[1:]:
        # Calcula el error de posición entre la posición actual y el objetivo actual
        error_x = objetivo[0] - pos_robot[0]
        error_y = objetivo[1] - pos_robot[1]

        # Calcula las velocidades lineal y angular utilizando el controlador PID
        v_lineal = KP * math.sqrt(error_x ** 2 + error_y ** 2)
        orientacion_objetivo = math.atan2(error_y, error_x)
        error_theta = orientacion_objetivo - pos_robot[2]
        error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))  # Limita el error entre -pi y pi
        v_angular = KI * error_theta + KD * error_theta

        # Limitar la velocidad lineal a un máximo de 1
        v_lineal = min(v_lineal, 1.0)

        # Limitar la velocidad angular entre -1 y 1
        v_angular = max(min(v_angular, 1.0), -1.0)

        # Mueve el robot hacia el objetivo actual
        pos_robot = mover_robot(pos_robot, v_lineal, v_angular, 0.1)

        # Calcula el error cuadrático medio
        error_cuadratico_medio += calcular_distancia((pos_robot[0], pos_robot[1]), objetivo) ** 2

    error_cuadratico_medio /= len(coordenadas_objetivo[1:])

    return error_cuadratico_medio

# Función para optimizar el controlador PID utilizando PSO
def optimizacion_pso(coordenadas_objetivo):
    num_particulas = 10
    num_iteraciones = 50
    dimensiones = 3  # KP, KI, KD

    # Rangos de búsqueda para cada dimensión de ganancias
    rango_min = np.array([KP_RANGE[0], KI_RANGE[0], KD_RANGE[0]])
    rango_max = np.array([KP_RANGE[1], KI_RANGE[1], KD_RANGE[1]])

    # Inicialización aleatoria de las posiciones y velocidades de las partículas
    posiciones = np.random.uniform(rango_min, rango_max, (num_particulas, dimensiones))
    velocidades = np.zeros((num_particulas, dimensiones))
    mejor_posicion_particula = np.copy(posiciones)
    mejor_desempenho_particula = np.ones(num_particulas) * np.inf
    mejor_posicion_global = np.zeros(dimensiones)
    mejor_desempenho_global = np.inf

    # Variables para almacenar los datos de las ganancias durante la optimización
    iteraciones = []
    ganancias_KP = []
    ganancias_KI = []
    ganancias_KD = []

    # Bucle principal de optimización
    for iteracion in range(num_iteraciones):
        for i in range(num_particulas):
            # Evaluación del desempeño de la partícula
            desempenho = funcion_objetivo(posiciones[i], coordenadas_objetivo)

            # Actualización de la mejor posición de la partícula
            if desempenho < mejor_desempenho_particula[i]:
                mejor_desempenho_particula[i] = desempenho
                mejor_posicion_particula[i] = np.copy(posiciones[i])

            # Actualización de la mejor posición global
            if desempenho < mejor_desempenho_global:
                mejor_desempenho_global = desempenho
                mejor_posicion_global = np.copy(posiciones[i])

            # Actualización de las velocidades y posiciones de las partículas
            inercia = 0.5
            coeficiente_cognitivo = 1.0
            coeficiente_social = 1.0
            r1 = np.random.random(dimensiones)
            r2 = np.random.random(dimensiones)
            velocidades[i] = (
                inercia * velocidades[i]
                + coeficiente_cognitivo * r1 * (mejor_posicion_particula[i] - posiciones[i])
                + coeficiente_social * r2 * (mejor_posicion_global - posiciones[i])
            )
            posiciones[i] += velocidades[i]

            # Limitar las posiciones dentro de los rangos de búsqueda
            posiciones[i] = np.clip(posiciones[i], rango_min, rango_max)

            # Añadir los datos de las ganancias al registro
            iteraciones.append(iteracion)
            ganancias_KP.append(mejor_posicion_global[0])
            ganancias_KI.append(mejor_posicion_global[1])
            ganancias_KD.append(mejor_posicion_global[2])

    # Graficar la variación de las ganancias
    plt.figure()
    plt.plot(iteraciones, ganancias_KP, label="KP")
    plt.plot(iteraciones, ganancias_KI, label="KI")
    plt.plot(iteraciones, ganancias_KD, label="KD")
    plt.xlabel("Iteraciones")
    plt.ylabel("Ganancias")
    plt.title("Variación de las ganancias durante la optimización")
    plt.legend()
    plt.show()

    return mejor_posicion_global

# Coordenadas objetivo
cantidad_coordenadas = max(5, random.randint(5, 20))
coordenadas_objetivo = [generar_punto_aleatorio() for _ in range(cantidad_coordenadas)]

# Optimizar controlador PID
mejores_ganancias = optimizacion_pso(coordenadas_objetivo)

# Ejecutar simulación con las mejores ganancias
KP, KI, KD = mejores_ganancias
simulacion(coordenadas_objetivo)
