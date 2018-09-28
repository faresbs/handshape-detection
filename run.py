from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer

import torch 
import torch.nn as nn
from torch.autograd import Variable
import cnn_architectures as cnn
import numpy as np
import cv2 

import sys
import time
import timeit
import joblib



def prep_image(image, isGray, input_dim, CUDA):

	img = cv2.resize(image, (input_dim, input_dim)) 
	

	if(isGray):
		img = np.resize(img, (input_dim, input_dim, 3))

	#Normalize test example
	if (isGray):

		mean = np.array([0.5, 0.5, 0.5])
		std = np.array([0.5, 0.5, 0.5])

	else:
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])


	img_ = std * img + mean

	img_ =  img_.transpose((2, 0, 1))
	img_ = img_[np.newaxis,:,:,:]/255.0

	#values outside the interval are clipped to the interval edges
	img_ = np.clip(img_, 0, 1)

	img_ = torch.from_numpy(img_).float()
	img_ = Variable(img_)

	if CUDA:
		img_ = img_.cuda()

	#Take only one channel for Gray since they are all duplicated channels
	if(isGray):
		img_ = img_[:, 0, :, :].unsqueeze(1)

	return img_


if __name__ == '__main__':

	if len(sys.argv) < 3:
		print('Usage: python run.py <detection model path>  <detection labels path>  <classification model path>')
		sys.exit(0)
	detection_path = sys.argv[1]
	label_detection_path = sys.argv[2]
	class_path = sys.argv[3]

	class_names_detection = [name.strip() for name in open(label_detection_path).readlines()]


	#Extract class names for the classification task
	with open(class_path+'/class_names', "rb") as file:
		class_names = joblib.load(file)

	num_classes = len(class_names)

	print('Handshape labels: '+str(class_names))

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	if device=='cpu':
		print ("Running on CPU.")
	elif device=='cuda:0':
		print ("Running on GPU.")

	#Loading classification model
	print("Loading networks...")
	class_model = cnn.Inception3(num_classes=24, channels=1, aux_logits=True)
	class_model.load_state_dict(torch.load(class_path+'/weights.h5'))
	print("Classification Network successfully loaded.")
	    
	#put the model in eval mode to disable dropout
	class_model = class_model.eval()

	#load model into the gpu
	class_model = class_model.to(device)


	#Loading detection model
	detect_model = create_mobilenetv1_ssd(2, is_test=True)
	detect_model.load(detection_path)
	predictor = create_mobilenetv1_ssd_predictor(detect_model, candidate_size=200)
	print("Detection Network successfully loaded.")

	video_capture = cv2.VideoCapture(0)

	#Image size for classification must identical to network input
	image_size = 299

	#Classification threshold
	threshold = 0.2

	#Empty label
	empty='None'

	#add some space for the detected bounding box
	add_bbox = 30

	#classify grayscale images or rgb
	isGray = True

	while True:

		#Start timer
		start = timeit.default_timer()

		# Capture frame-by-frame
		ret, frame = video_capture.read()

		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		#Predict bounding boxes using the first sub network
		boxes, labels, probs = predictor.predict(image, 10, 0.4)

		#Extract images from bounding box
		images = []
		x1s = []
		y1s = []

		for i in range(boxes.size(0)):
		    box = boxes[i, :]

		    #Transform from tensor to int
		    x1 = int(box[0]) - add_bbox
		    y1 = int(box[1]) - add_bbox
		    x2 = int(box[2]) + add_bbox
		    y2 = int(box[3]) + add_bbox

		    #coords must not exceed the limit of the frame or be negative

		    if x1 < 0:
		    	x1 = 0

		    if x2 > frame.shape[1]:
		    	x2 = frame.shape[1]

		    if y1 < 0:
		    	y1 = 0

		    if y2 > frame.shape[0]:
		    	y2 = frame.shape[0]


		    image = frame[y1:y2, x1:x2]

		    if (isGray):
		    	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		    #Resize captured image to be identical with the image size of the training data
		    img = prep_image(image, isGray, image_size, CUDA=True)

		    #Prediction
		    output = class_model(img)
		    prediction = output.data.argmax()
		    value, predicted = torch.max(output.data, 1)

		    #Tranform logits to probablities
		    m = nn.Softmax()
		    input = output.data
		    output = m(input)
		    output = np.around(output,2)
		    value, predicted = torch.max(output.data, 1)

		    #if prediction is not accurate returns empty

		    if (value >= threshold):
		    	prediction = class_names[predicted]
		    	fontColor = (255, 0, 0)
		    	lineType = 4
		    else:
		    	prediction = empty
		    	fontColor = (255, 255, 0)
		    	lineType = 2

		    #THERE IS A PROBLEM HERE 
		    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), fontColor, lineType)

		    cv2.putText(frame, prediction, (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  


		#print(f"Found {len(probs)}")

		stop = timeit.default_timer()
		running_time = 1/(stop - start)
		print ("FPS: {:.2f}, Found {:d} objects".format(running_time, len(probs)))
		#cv2.putText(frame, running_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)  

		
		#Show results
		#frame = cv2.flip(frame, 1)
		cv2.imshow("Video Stream", frame)


		#PRESS Q TO QUIT
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()

		