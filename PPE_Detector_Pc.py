#!/usr/bin/env python3

import sys, os, cv2, time
import numpy as np, math

from openvino.inference_engine import IENetwork, IEPlugin #for PC
#from armv7l.openvino.inference_engine import IENetwork, IEPlugin #for Rpi

#from picamera.array import PiRGBArray
#from picamera import PiCamera
#from sense_hat import SenseHat

# Define confidence thresholds.
confidence_threshold_hat = 0.1
confidence_threshold_person = 0.4
confidence_threshold_vest = 0.2

# Define intersection over union threshold.
IOU_threshold = 0.1

# Sw version. 
version = "v4.03"

# Yolo parameters. 
m_input_size = 416
yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52
classes = 3
coords = 4
num = 3
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

# Labels parameters. 
LABELS = ("person", "hat", "vest")
label_text_color = (255, 255, 255) # Text color
label_background_color = (0, 0, 0)
box_color = (255, 128, 0)

# Box parameters. 
color1 = [0,0,255] # Person box color
color2 = [0,255,0] # Hat box color
color3 = [255,0,0] # Vest box color
box_thickness = 2

#Sense hat display code.



def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval

def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects

# initialize the camera and grab a reference to the raw camera capture
#print("initializating PiCamera")
#camera = PiCamera()
#camera.resolution = (416, 416)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(416, 416))
#time.sleep(0.5) # allow the camera to warmup

# Initialize camera and resolution constants. 

camera_width = 416
camera_height = 416

new_w = int(camera_width * min(m_input_size/camera_width, m_input_size/camera_height))
new_h = int(camera_height * min(m_input_size/camera_width, m_input_size/camera_height))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    

# ----- Coment this section if you dont want to record the output. 
FILE_OUTPUT='output2.avi' # The output is stored in 'outpy.avi' file.
fourcc = cv2.VideoWriter_fourcc(*'XVID') # Define the codec and create VideoWriter object.
out = cv2.VideoWriter('output.avi', fourcc, 15, (416, 416))


def main_IE_infer():
    
    # Define Constants.

    t1 = 0
    fps = ""
    #framepos = 0
    #frame_count = 0
    #vidfps = 0
    #skip_frame = 0
    #elapsedTime = 0
    
    
    
    detected_people_frames=[0,0,0,0,0]
    detected_hat_frames=[0,0,0,0,0]
    detected_vest_frames=[0,0,0,0,0]
    
    
    print("loading the model...")
#    args = build_argparser().parse_args()
    model_xml = "tiny_yolo_IR_500000_FP32.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    time.sleep(1)
    print("loading plugin on Intel NCS2...")
    
    plugin = IEPlugin(device="CPU")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    
    # Define a window to show the cam stream on it
    window_title= "PPE Detector"   
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        current_time=time.ctime();

        # initialize lists to store objects in frame
        People=[]
        Hats=[]
        Vests=[]
        objects = []
        
        det_person =0
        det_hat=0
        det_vest=0
        # Shift register for detected objects. 
        detected_people_frames[1:]=detected_people_frames[0:9]
        detected_hat_frames[1:]=detected_hat_frames[0:9]
        detected_vest_frames[1:]=detected_vest_frames[0:9]

        #lload the image from the camera. 
        ret, image = cap.read()
        if not ret:
            break
        #resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC) #resize image to 416x416 
        resized_image = cv2.resize(image, (new_w, new_h)) #resize image to 416x416 
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        canvas = np.full((m_input_size, m_input_size, 3), 0)
        canvas[(m_input_size-new_h)//2:(m_input_size-new_h)//2 + new_h,(m_input_size-new_w)//2:(m_input_size-new_w)//2 + new_w,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        outputs = exec_net.infer(inputs={input_blob: prepimg})


        

        for output in outputs.values():
            objects = ParseYOLOV3Output(output, new_h, new_w, camera_height, camera_width, confidence_threshold_hat, objects)

        # Filtering overlapping boxes same class
        
        # Separate classes detected. 
        objlen = len(objects)

        for i in range(objlen):
            if(objects[i].class_id == 0):
                People.append(objects[i])
            
            if(objects[i].class_id == 1):
                Hats.append(objects[i])
            if(objects[i].class_id == 2):
                Vests.append(objects[i])

        # Elimitate overlaping Hats. 
        objlen = len(Hats)
        for i in range(objlen):
            if (Hats[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (IntersectionOverUnion(Hats[i], Hats[j]) >= IOU_threshold):
                    if Hats[i].confidence < Hats[j].confidence:
                        Hats[i], Hats[j] = Hats[j], Hats[i]
                    Hats[j].confidence = 0.0
        
        # Drawing hats boxes. 
        for obj in Hats:
            if obj.confidence < confidence_threshold_hat:
                continue
            label = obj.class_id
            confidence = obj.confidence
            #if confidence >= 0.2:
            label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            #label_text = LABELS[label]
            cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), color2, box_thickness)
            cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
            det_hat=det_hat+1
        detected_hat_frames[0] = det_hat
        
        # Eliminate overlaping vests.
        objlen = len(Vests)
        for i in range(objlen):
            if (Vests[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (IntersectionOverUnion(Vests[i], Vests[j]) >= IOU_threshold):
                    if Vests[i].confidence < Vests[j].confidence:
                        Vests[i], Vests[j] = Vests[j], Vests[i]
                    Vests[j].confidence = 0.0
        
        # Drawing vests boxes
        for obj in Vests:
            #print(str(confidence))
            if obj.confidence < confidence_threshold_vest:
                continue
            label = obj.class_id
            confidence = obj.confidence
            #print(str(confidence))
            label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            #label_text = LABELS[label]
            cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), color3, box_thickness)
            cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
            det_vest=det_vest+1
        detected_vest_frames[0] = det_vest
        
        # Eliminate overlaping people
        objlen = len(People)
        for i in range(objlen):
            if (People[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (IntersectionOverUnion(People[i], People[j]) >= IOU_threshold):
                    if People[i].confidence < People[j].confidence:
                        People[i], People[j] = People[j], People[i]
                    People[j].confidence = 0.0
        
        # Drawing people's boxes
        for obj in People:
            if obj.confidence < confidence_threshold_person:
                continue
            label = obj.class_id
            confidence = obj.confidence
            #if confidence >= 0.2:
            label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            #label_text = LABELS[label]
            cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), color1, box_thickness)
            cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
            det_person=det_person+1
        detected_people_frames[0]=det_person

        conclusion_color = [0,0,255]
                
        if max(detected_people_frames)!=0:
            if max(detected_hat_frames)!=0:
                if max(detected_vest_frames)!=0:
                    conclusion="PPE: Detected!"
                    conclusion_color = [0,0,255]
                    print (current_time[11:19]+ ' - ' + '\x1b[6;30;42m' + '[✔] %s [✔]' % (conclusion) + '\x1b[0m')
                                    
                    #sh.set_pixels(Person_Vest_Hat)
                    conclusion_color = [0,255,0]
                else:
                    conclusion="PPE: Not Pass, Missing Vest"
                    print (current_time[11:19] + ' - ' + '\x1b[1;29;41m' + '[✘]' + conclusion + '[✘]' + '\x1b[0m')
                    #sh.set_pixels(Person_hat)
            else:
                if max(detected_vest_frames)!=0:
                    conclusion="PPE: Not Pass, Missing Hat"
                    print (current_time[11:16] + ' - ' +  '\x1b[1;29;41m' + '[✘]' + conclusion + '[✘]' + '\x1b[0m')
                    #sh.set_pixels(Person_Vest)
                else:
                    conclusion="PPE: Not Pass, Missing Vest and hat"
                    print (current_time[11:19]+ ' - ' + '\x1b[1;29;41m' + '[✘]' + conclusion + '[✘]' + '\x1b[0m')
                    #sh.set_pixels(Person)
        else:
            conclusion = "No Person Detected"
            #sh.set_pixels(question_mark) #display a question mark
            print (current_time[11:19] + ' - ' + conclusion)

        # Write Performance information. 
        elapsedTime = time.time() - t1
        fps = "{:.1f} FPS".format(1/elapsedTime)
        
        #   print((fps+ " - " + conclusion)) 

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        
        # Restart the time. 
        #t1 = time.time()


        cv2.putText(image, (fps), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conclusion_color, 1, cv2.LINE_AA)
        
        #Display Image
        cv2.imshow(window_title, image)
        # wrute the output
        for f in range(3):
            out.write(image)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        
        # Restart the time. 
        t1 = time.time()

        #temp = sh.get_temperature()
        #print("Temp: %s ºC" % str(round(temp,1)))               # Show temp on console
        #rawCapture.truncate(0)
    out.release()
    cv2.destroyAllWindows()
    
    #sh.clear(0,0,0)
    del net
    del exec_net
    del plugin


if __name__ == '__main__':
  
    
    sys.exit(main_IE_infer() or 0)
