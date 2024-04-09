import cv2
import matplotlib.pyplot as plt


''' Consigna

Elija una de las imágenes color que tomó para la clase y aplique separación de canales
y elija un método para transformarla en escala de grises.
Muestre por pantalla los resultados obtenidos. 

'''


# Load the image
imagen = cv2.imread('Unidad1\imagenes\cv2_2024_04_09.jpg')

# Separate color channels
canal_azul, canal_verde, canal_rojo = cv2.split(imagen)

# Convert the image to grayscale by taking the maximum of the channels
gris_maximo = cv2.max(imagen[:, :, 0], cv2.max(imagen[:, :, 1], imagen[:, :, 2]))

# Create the figure and subplots
plt.figure(figsize=(15, 10))

# Original image and maximum of channels
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gris_maximo, cmap='gray')
plt.title('Máximo de canales')
plt.axis('off')

# Red channel
plt.subplot(2, 3, 4)
plt.imshow(canal_rojo, cmap='Reds')
plt.title('Canal Rojo')
plt.axis('off')

# Green channel
plt.subplot(2, 3, 5)
plt.imshow(canal_verde, cmap='Greens')
plt.title('Canal Verde')
plt.axis('off')

# Blue channel
plt.subplot(2, 3, 6)
plt.imshow(canal_azul, cmap='Blues')
plt.title('Canal Azul')
plt.axis('off')

plt.show()
