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



def prep_image(image, input_dim, CUDA):

	img = cv2.resize(image, (input_dim, input_dim)) 

	#Normalize and apply std for test example
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

	return img_


if __name__ == '__main__':

	if len(sys.argv) < 3:
		print('Usage: python run_ssd_example.py <detection model path>  <detection labels path>  <classification model path>')
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
	class_model = cnn.network_vgg16(num_classes)
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
	image_size = 224

	#Classification threshold
	threshold = 0.3

	#Empty label
	empty='None'

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
		    x1 = int(box[0])
		    y1 = int(box[1])
		    x2 = int(box[2])
		    y2 = int(box[3])

		    images.append(frame[y1:y2, x1:x2])
		    x1s.append(x1)
		    y1s.append(y1)

		#apply second sub network to every detected bounding box
		for i in range(len(images)):    
			#Resize captured image to be identical with the image size of the training data
			img = prep_image(images[i], image_size, True)

			#load in gpu
			img = img.to(device)

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
		    
			cv2.putText(frame, prediction, (x1s[i] + 10, y1s[i] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  




		#print(f"Found {len(probs)}")

		stop = timeit.default_timer()
		running_time = 1/(stop - start)
		print ("FPS: {:.2f}, Found {:d} objects".format(running_time, len(probs)))
		#cv2.putText(frame, running_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)  

		#we can flip the frame here so that it wont be weird
		#Show results
		cv2.imshow("Video Stream", frame)

		#PRESS Q TO QUIT
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		