# Imports
import numpy as np
import cv2 as cv
import time

def blob_print(blob):
    # Slicing blob image and transposing to make channels come at the end
    blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
    print(blob_to_show.shape)  # (416, 416, 3)

    # Showing Blob Image
    # Giving name to the window with Blob Image
    # And specifying that window is resizable
    cv.namedWindow('Blob Image', cv.WINDOW_NORMAL)

    # Pay attention! 'cv.imshow' takes images in BGR format
    # Consequently, we DO need to convert image from RGB to BGR firstly
    # Because we have our blob in RGB format
    cv.imshow('Blob Image', cv.cvtColor(blob_to_show, cv.COLOR_RGB2BGR))

    # Waiting for any key being pressed
    cv.waitKey(0)

    # Destroying opened window with name 'Blob Image'
    cv.destroyWindow('Blob Image')
    
    
    

# Part 01
image_BGR = cv.imread('../images/Gatinho_01.jpeg')
cv.namedWindow('Title of the original image', cv.WINDOW_NORMAL)
cv.imshow('Title of the original image', image_BGR)
cv.waitKey(0)

# Part 02
h, w = image_BGR.shape[:2] # We don't need the third position, which stands for the channels of the image
print(f'Height: {h}, Width: {w}')

# Part 03
# O próximo passo é converter a imagem em um "blob". Um blob é uma forma de pré-processamento da imagem,
# que permite que ela seja alimentada na rede.
# O método blobFromImage realiza várias operações, como escalonar os pixels (aqui, eles são escalados para um intervalo de 0 a 1,
# usando 1/255.0), redimensionar a imagem para 416x416 pixels (um requisito do YOLOv3)
# e trocar os canais de azul-vermelho para vermelho-azul.
blob = cv.dnn.blobFromImage(
    image_BGR,  # image we want to convert into a blob
    1 / 255.0,  # normalization factor
    (416, 416), # image size after preprocessing (we need this shape in YOLOv3)
    swapRB=True,# Opencv works with BGR channels, and YoLo with RGB
    crop=False  # we don't crop image before passing it to network
)
blob_print(blob)


# Part 04 - Loading YOLO from disk
print("Loading YOLO from disk...")
network = cv.dnn.readNetFromDarknet('./yolo-coco-data/yolov3.cfg', './yolo-coco-data/yolov3.weights')
print("YOLO loaded successfully!")


# Part 05

layers_names_all = network.getLayerNames()
print(layers_names_all)

print("Unconnected out layers:", network.getUnconnectedOutLayers())

layers_names_output = [\
    layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

print("Unconnected out layers names:", layers_names_output)

# Opening file
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# Part 06
probability_minimum = 0.2 # Minimum probability to eliminate weak predictions
threshold = 0.3 # Threshold when applying non-maximum suppression
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8') # Generating colors for bounding boxes

network.setInput(blob) # Implementing forward pass
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

print(f'Objects Detection took {(end - start):.5f} seconds')

# Part 
bounding_boxes = []
confidences = []
class_numbers = []

for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]
        
        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)


results = cv.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

counter = 1

# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    # Going through indexes of results
    for i in results.flatten():
        # Showing labels of the detected objects
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

        # Incrementing counter
        counter += 1

        # Getting current bounding box coordinates,
        # its width and height
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        # Preparing colour for current bounding box
        # and converting from numpy array to list
        colour_box_current = colors[class_numbers[i]].tolist()

        # # # Check point
        # print(type(colour_box_current))  # <class 'list'>
        # print(colour_box_current)  # [172 , 10, 127]

        # Drawing bounding box on the original image
        cv.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        # Preparing text with label and confidence for current bounding box
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        # Putting text with label and confidence on the original image
        cv.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)


print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)


# Showing Original Image with Detected Objects
# Giving name to the window with Original Image
# And specifying that window is resizable
cv.namedWindow('Detections', cv.WINDOW_NORMAL)
# Pay attention! 'cv.imshow' takes images in BGR format
cv.imshow('Detections', image_BGR)
# Waiting for any key being pressed
cv.waitKey(0)
# Destroying opened window with name 'Detections'
cv.destroyWindow('Detections')

#     leaky: Esta é uma referência à função de ativação "Leaky ReLU" (Rectified Linear Unit). A função ReLU é comumente usada em redes neurais para introduzir não-linearidade no modelo. Basicamente, para qualquer entrada negativa, ela retorna zero, e para qualquer entrada positiva, ela retorna a própria entrada. A versão "Leaky" dessa função permite uma pequena saída mesmo para entradas negativas. Isso pode ajudar a evitar o problema conhecido como "neurônios morrendo", onde certos neurônios nunca são ativados.

#     bn: Essa é uma abreviação de "Batch Normalization", uma técnica usada para aumentar a estabilidade de uma rede neural. Ela normaliza a entrada para cada camada para ter uma média de zero e uma variância de um. Isso tem o efeito de acelerar o treinamento da rede.

#     permute: Essa camada é usada para reordenar as dimensões de uma entrada.

#     conv: Isso se refere a uma "camada convolucional". As camadas convolucionais são o bloco de construção principal de uma Convolutional Neural Network (CNN), que é o tipo de rede neural usada em YOLO. Essas camadas são responsáveis por aprender e aplicar um conjunto de filtros à imagem de entrada para criar "mapas de recursos" que destacam diferentes características da imagem.

#     yolo: Essas são as camadas de saída da rede, responsáveis por fazer as previsões finais. Cada camada YOLO faz previsões em uma escala diferente, permitindo que a rede detecte objetos de vários tamanhos.

# A razão pela qual essas camadas têm esses nomes é porque cada uma realiza uma função específica dentro da rede. Juntas, essas camadas permitem que a rede YOLOv3 detecte objetos em uma imagem e determine suas classes e localizações.
















