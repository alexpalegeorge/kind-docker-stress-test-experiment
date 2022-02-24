# import the necessary packages
import numpy as np
import sys
import time
import cv2
import os
from flask import Flask, jsonify, request
import base64
import json

app = Flask(__name__)

# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath = os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def do_prediction(image, net, LABELS):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]
            # print("confidence: ", confidence)

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    objects = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            rectangle = {
                "X": boxes[i][0],
                "Y": boxes[i][1],
                "width": boxes[i][2],
                "height": boxes[i][3]
            }
            detection_entry = {
                "detected item": LABELS[classIDs[i]],
                "accuracy": confidences[i],
                "rectangle": rectangle
            }
            objects.append(detection_entry)
            # "detected item:{}, accuracy:{}, X:{}, Y:{}, width:{}, height:{}".format(LABELS[classIDs[i]],
            # confidences[i],
            # boxes[i][0],
            # boxes[i][1],
            # boxes[i][2],
            # boxes[i][3]))
    else:
        objects = ["No detected items"]
    return objects


yolo_path = "yolo_tiny_configs/"
labelsPath = "coco.names"
cfgpath = "yolov3-tiny.cfg"
wpath = "yolov3-tiny.weights"

Lables = get_labels(labelsPath)
CFG = get_config(cfgpath)
Weights = get_weights(wpath)


@app.route('/api/object_detection', methods=['POST'])
def main():
    try:
        # get object from post and load it as json
        input_data = request.get_json()
        input_json = json.loads(input_data)

        # extract id and image values
        input_id = input_json['id']
        input_image = input_json['image']

        # decode image
        image_decode = base64.b64decode(input_image)
        np_array = np.asarray(bytearray(image_decode), dtype="uint8")
        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        image = cv_image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load the neural net.  Should be local to this method as its multi-threaded endpoint
        nets = load_model(CFG, Weights)

        # return the jsonify object
        objects = do_prediction(image, nets, Lables)

        return jsonify(id=input_id, objects=objects, indent=4)



    except Exception as e:

        print("Exception  {}".format(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
