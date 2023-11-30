import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_circles(image_path):
    img = cv.imread(image_path)

    # Separando os canais de cor da imagem
    blue_channel = img[:,:,0]  # Canal azul

    # Aplicando operações de pré-processamento no canal azul
    kernel = np.ones((5,5), np.uint8)
    blue_channel = cv.GaussianBlur(blue_channel, (9,9), 0)
    blue_channel = cv.morphologyEx(blue_channel, cv.MORPH_OPEN, kernel)
    blue_channel = cv.morphologyEx(blue_channel, cv.MORPH_CLOSE, kernel)
    blue_channel = cv.threshold(blue_channel, 60, 255, cv.THRESH_BINARY_INV)[1]
    blue_channel = cv.morphologyEx(blue_channel, cv.MORPH_CLOSE, kernel)

    canny = cv.Canny(blue_channel, 100, 200)

    plt.imshow(blue_channel, cmap="gray")
    plt.show()

    circles = cv.HoughCircles(blue_channel,
                              cv.HOUGH_GRADIENT,
                              dp=1.1,
                              minDist=300,
                              param1=200,
                              param2=40,
                              minRadius=50,
                              maxRadius=400)

    if circles is not None:
        circles = circles[0].astype(int)  # Convertendo para inteiros
        return circles  # Retornando coordenadas x, y e raio dos círculos

    return None


def find_random_point_from_circle(circle_coordinates, max_iterations=1000):
    circle_x, circle_y, circle_radius = circle_coordinates

    tolerance = circle_radius  # Tolerância para ajustar a busca

    start_x, end_x = circle_x - circle_radius, circle_x + circle_radius
    start_y, end_y = circle_y - circle_radius, circle_y + circle_radius

    attempts_x = [start_x]
    attempts_y = [start_y]
    
    iterations = 0

    while iterations < max_iterations:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        attempts_x.append(mid_x)
        attempts_y.append(mid_y)

        # Verifica se o ponto aleatório está dentro do círculo
        if (mid_x - circle_x) ** 2 + (mid_y - circle_y) ** 2 < circle_radius ** 2 - tolerance:
            # Se estiver dentro do círculo, ajusta os limites da busca
            start_x = mid_x - tolerance
            end_x = mid_x + tolerance
            start_y = mid_y - tolerance
            end_y = mid_y + tolerance

        else:
            # Se estiver fora do círculo, atualiza os limites da busca
            if random.random() < 0.5:
                end_x = mid_x
            else:
                start_x = mid_x
            
            if random.random() < 0.5:
                end_y = mid_y
            else:
                start_y = mid_y

        if end_x - start_x < tolerance and end_y - start_y < tolerance:
            return (mid_x, mid_y), attempts_x, attempts_y

        iterations += 1

    return None, attempts_x, attempts_y

# Chamando a função com a imagem desejada
circles_coordinates = detect_circles("CN/Teste2.jpg")
print(circles_coordinates)

for circle_coordinates in circles_coordinates:
    random_point, attempts_x, attempts_y = find_random_point_from_circle(circle_coordinates)

    if random_point is not None:
        plt.plot(attempts_x, attempts_y, 'bo-', label='Attempts')
        plt.plot(random_point[0], random_point[1], 'ro', label='Random Point')
        circle = plt.Circle((circle_coordinates[0], circle_coordinates[1]), circle_coordinates[2], color='green', fill=False, label='Detected Circle')
        plt.gca().add_patch(circle)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Não foi possível encontrar um ponto aleatório dentro do círculo.")