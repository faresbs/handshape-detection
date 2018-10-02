##This is for creating new data images for sign language through sessions##
##Work for both continued and static signs

import cv2
import datetime as dt
import time
import os
import numpy as np

#TO DO 
#APPLY BOUNDING BOX IF NEEDED ELSE TAKE ONLY WHATS REQUIRED FOR THE SIGN


if __name__ == "__main__":

	window_size = 400

	fontColor = (255, 255, 0)
	lineType = 2

	#folder where we save data
	if not os.path.exists('data'):
	   os.makedirs('data')


	sl = input("What's the sign language you'll be using? ")

	#create a folder for the sign language
	if not os.path.exists('data/'+sl):
	   os.makedirs('data/'+sl)


	#Save the images in a folder session
	date = dt.datetime.now().strftime("%Y-%m-%d")
	if not os.path.exists('data/'+sl+'/'+date):
	   os.makedirs('data/'+sl+'/'+date)


	#Type of the signs you'll be using static signs or temporal signs that use movement
	Type = input("Temporal/static signs (t/s)? ")


	video_capture = cv2.VideoCapture(0)

	#For static signs
	if (Type=='s'):

		#sign that you'll use
		sign = input("What's the sign that you'll be using ? ")

		#small bbox is for signs that only use handshape and big bbox will have multiple components
		bbox = input("What's the bbox dimensions that you'll be using (big/small) ? ")

		if (bbox == 'big'):
			#Dimension of the rectangle for the bounding box for multi-components sign
			x = 300
			y = 100
			h = 300
			w = 300
		elif (bbox == 'small'):
			#Dimension of the rectangle for the bounding box for handshape
			x = 100
			y = 150
			h = 200
			w = 200


		save = 'data/'+sl+'/'+date+'/static/'+sign

		#Save in the sign folder in the current session folder
		date = dt.datetime.now().strftime("%Y-%m-%d")
		if not os.path.exists(save):
		   os.makedirs(save)

		if not os.path.exists(save):
		   os.makedirs(save)


		files = os.listdir(save)
		pic_num = len(files)

		#Initialize captured image as a black image
		image = np.zeros((window_size,window_size, 3), np.uint8)

		while True:

			# Capture frame-by-frame
			ret, frame = video_capture.read()

			#PRESS Space to capture the image
			k = cv2.waitKey(33)
			if k == 32:
				#Extract bounding box image
				image = frame[y:y+h,x:x+w]
				cv2.imwrite(save+'/'+str(pic_num)+'.png', image)
				pic_num += 1

			cv2.rectangle(frame, (x, y), (x+w, y+h), fontColor, lineType)

			frame = cv2.resize(frame, (window_size, window_size))
			image = cv2.resize(image, (window_size, window_size))
			cv2.imshow("Video Stream", np.hstack([frame, image]))

			#PRESS Q TO QUIT
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


	#For temporal signs that require movement
	elif (Type == 't'):

		#how = input("Temporal/static signs (t/s)? ")

		#sign that you'll use
		sign = input("What's the sign that you'll be using ? ")

		save = 'data/'+sl+'/'+date+'/temporal/'+sign

		#Save in the sign folder in the current session folder
		date = dt.datetime.now().strftime("%Y-%m-%d")

		if not os.path.exists(save):
		   os.makedirs(save)


		subfolders = os.listdir(save)
		folder_num = len(subfolders)

		#Initialize captured image as a black image
		image = np.zeros((window_size,window_size,3), np.uint8)

		while True:

			# Capture frame-by-frame
			ret, frame = video_capture.read()

			#KEEP PRESSING k to save the frames in the folder 
			k = cv2.waitKey(1)
			if k == 32:
				if not os.path.exists(save+'/'+str(folder_num)):
					os.makedirs(save+'/'+str(folder_num))

				frame_num = 0
				while True:
					#Save in lower fps rate so that we dont capture unecessary frames
					k = cv2.waitKey(2000)
					if k == 32:
						cv2.imwrite(save+'/'+str(folder_num)+'/'+str(frame_num)+'.png', frame)
						frame_num += 1
						ret, frame = video_capture.read()
					else:
						folder_num += 1
						break


			cv2.imshow("Video Stream", frame)

			#KEPP PRESSING Q TO QUIT
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break










