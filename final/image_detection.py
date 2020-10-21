def detecteion_bienso():
    # library
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    import sys
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    sys.path.append("..")
    IMAGE_NAME = '5.jpg'
    forder_path = "C:\\Users\\DELL\\Desktop\\final"
    PATH_TO_CKPT = os.path.join(forder_path,'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(forder_path,'labelmap.pbtxt')

    PATH_TO_IMAGE = os.path.join(forder_path,IMAGE_NAME)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image = cv2.imread(PATH_TO_IMAGE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    true_box = boxes[0][scores[0] > 0.8]
    h, w = image.shape[1], image.shape[0]
    ymin = int(true_box[0,0]*w)
    xmin = int(true_box[0,1]*h)
    ymax = int(true_box[0,2]*w)
    xmax = int(true_box[0,3]*h)
    # crop and save
    image_crop = image[ymin : ymax, xmin : xmax]
    cv2.imshow('Object detector', image_crop)
    cv2.imwrite("bienso5.JPG",image_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # convert from image to text
    image_to_text = cv2.imread('bienso5.jpg')[:,:,0]
    w, h =  image_to_text.shape
    bigimage = np.ones((w*4, h*6), np.uint8)*255
    bigimage[w*2:w*3, h*2:h*3] = 0
    bigimage[w*2:w*3, h*2:h*3] = bigimage[w*2:w*3, h*2:h*3] + image_to_text
    text = pytesseract.image_to_string(bigimage, lang = 'eng', config = '--psm 11')
    print(text)
    print(scores[0][0])

    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 5)
    cv2.imwrite('test.jpg', image)

detecteion_bienso()