
import cv2
import numpy as np

net = cv2.dnn.readNet("C:/Users/asus/Downloads/MaskRecog/yolov3_training_4000.weights",
                      "C:/Users/asus/Downloads/MaskRecog/yolov3_testing.cfg")

classes = []
with open("C:/Users/asus/Downloads/MaskRecog/classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x + 10, y + 40), (x + w - 10, y + h - 20), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + h - 30), font, 2, (255, 255, 255), 2)

    cv2.imshow('tushar',img)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
