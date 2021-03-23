import numpy as np
import tensorflow as tf
import cv2
import datetime

from centroidtracker.centroidtracker import CentroidTracker

VDO_SOURCE = "./video/traffic_day.mp4"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Required TensorFLow 2+

# Read the graph.
detection_graph = tf.Graph()
with detection_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('./ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        
        # open video
        cap = cv2.VideoCapture(VDO_SOURCE)

        sys_init = False
        traffic_light = []
        mon_line = []
        cap_line = []
        
        in_mon_list = []

        ct = CentroidTracker()

        while cap.isOpened():

            rect = []

            # read frame
            (ret, frame) = cap.read()

            # video end
            if not ret:
                break

            if sys_init == False:
                traff_sel_roi = cv2.selectROI("Select Traffic Light", frame, False, False)
                cv2.destroyWindow("Select Traffic Light")
                traffic_light = [(traff_sel_roi[0], traff_sel_roi[1]), (traff_sel_roi[0] + traff_sel_roi[2], traff_sel_roi[1] + traff_sel_roi[3])]

                sel_roi = cv2.selectROI("Select Monitor Line", frame, False, False)
                cv2.destroyWindow("Select Monitor Line")
                mon_line = [(sel_roi[0], sel_roi[1]), (sel_roi[0] + sel_roi[2], sel_roi[1] + sel_roi[3])]

                sel_roi = cv2.selectROI("Select Capture Line", frame, False, False)
                cv2.destroyWindow("Select Capture Line")
                cap_line = [(sel_roi[0], sel_roi[1]), (sel_roi[0] + sel_roi[2], sel_roi[1] + sel_roi[3])]

                sys_init = True

            cols = frame.shape[1]
            rows = frame.shape[0]

            image_np_expanded = np.expand_dims(frame, axis=0)

            # we use this to detect what is the current signal of traffic light
            traffic_light_crop = frame[int(traff_sel_roi[1]):int(traff_sel_roi[1]+traff_sel_roi[3]), int(traff_sel_roi[0]):int(traff_sel_roi[0]+traff_sel_roi[2])]
            hsv_traffic_light_crop = cv2.cvtColor(traffic_light_crop, cv2.COLOR_BGR2HSV)
            # color code right now is not the right one
            lower_red = np.array([0,100,100])
            upper_red = np.array([10,255,255])
            
            traffic_signal_mask = cv2.inRange(hsv_traffic_light_crop, lower_red, upper_red)

            print(np.sum(traffic_signal_mask))

            cv2.imshow('traffic_light_color', traffic_signal_mask)

            # check if found red color and size
            if np.sum(traffic_signal_mask) > 5000:
                print("signal is now red")
                
                # run detection

            # Run the model
            out = sess.run([detection_graph.get_tensor_by_name('num_detections:0'),
                            detection_graph.get_tensor_by_name('detection_scores:0'),
                            detection_graph.get_tensor_by_name('detection_boxes:0'),
                            detection_graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': image_np_expanded})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])

            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows

                    rect.append((x, y, right, bottom))
                    cv2.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=1)

            # update tracking
            # we need unique id for each detection to find if it is the same car
            # that is under watching
            objects = ct.update(rect)

            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)

                if centroid[1] > (mon_line[0][1]+mon_line[1][1])/2 and objectID not in in_mon_list:
                    in_mon_list.append(objectID)

                if centroid[1] < (cap_line[0][1]+cap_line[1][1])/2 and objectID in in_mon_list:
                    
                    # debug purpose
                    print("violate traffic: {}".format(objectID))
                    
                    in_mon_list.remove(objectID)
            
            # debug purpose
            print(in_mon_list)

            cv2.rectangle(frame, traffic_light[0], traffic_light[1], (0, 0, 255), thickness=1)
            cv2.line(frame, mon_line[0], mon_line[1], (255, 0, 0), thickness=1)
            cv2.line(frame, cap_line[0], cap_line[1], (255, 0, 255), thickness=1)

            cv2.imshow('vehicle detection', frame)
            cv2.imshow('traffic light', traffic_light_crop)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
