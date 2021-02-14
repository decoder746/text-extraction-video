# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import os


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			# endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			# endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			# startX = int(endX - w)
			# startY = int(endY - h)
			# # add the bounding box coordinates and probability score
			# # to our respective lists
			# rects.append((startX, startY, endX, endY))
            # A more accurate bounding box for rotated text
			offsetX = offsetX + cos * xData1[x] + sin * xData2[x] 
			offsetY = offsetY - sin * xData1[x] + cos * xData2[x]                
					
			# calculate the UL and LR corners of the bounding rectangle
			p1x = -cos * w + offsetX
			p1y = -cos * h + offsetY
			p3x = -sin * h + offsetX
			p3y = sin * w + offsetY
									
			# add the bounding box coordinates
			rects.append((p1x, p1y, p3x, p3y))

			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optional input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-o","--output",type=str, default = "./output/op.txt",
	help="Path to the output file")
ap.add_argument("-t","--tesseract",type=str,
	help="The absolute location of the file tesseract.exe, wherever you have installed it")
args = vars(ap.parse_args())

pytesseract.pytesseract.tesseract_cmd = args["tesseract"]
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)
# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
frame_counter = 0
final_text = []

prev_text = ""
start_frame = 0
stuck_in_loop = False

while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame, maintaining the aspect ratio
	old_frame = imutils.resize(frame, width=1000)
	orig = frame.copy()
	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)
	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(old_frame, (newW, newH))
	if frame_counter%60==0:
		# construct a blob from the frame and then perform a forward pass
		# of the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)
		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		# loop over the bounding boxes
		max_endX = -1
		max_endY = -1
		min_startX = 1e9
		min_startY = 1e9
		enter_loop = False
		for image_counter,(startX, startY, endX, endY) in enumerate(boxes):
			enter_loop = True
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			# print("Frame {} image counter {}: {} {} {} {}".format(frame_counter,image_counter,startY,endY,startX,endX))
			max_endX = max(max_endX,endX)
			max_endY = max(max_endY,endY)
			min_startX = min(min_startX,startX)
			min_startY = min(min_startY,startY)
			# draw the bounding box on the frame
			# roi = orig[startY:endY, startX:endX]
			# # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
			# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			# # gray = cv2.medianBlur(gray, 5) 
			# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			# filename = ".\images\{}_{}.png".format(frame_counter,image_counter)
			# cv2.imwrite(filename, gray)
			# text = pytesseract.image_to_string(Image.open(filename))
			# f = open("op.txt",'a')
			# f.write(text)
			# f.close

		if enter_loop :
			# print(min_startY,max_endY,min_startX,max_endX)
			roi = orig[min_startY:max_endY, min_startX:max_endX]
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			# gray = cv2.medianBlur(gray, 5) 
			gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			filename = "{}.png".format(frame_counter)
			cv2.imwrite(filename, gray)
			text = pytesseract.image_to_string(Image.open(filename))
			os.remove(filename)
			# f = open("op.txt",'a')
			# f.write(text)
			# f.close
			if prev_text == text :
				stuck_in_loop = True
				# continue
			elif prev_text != "":
				stuck_in_loop = False
				f = open(args["output"],'a')
				f.write("\n\n\n###################################### Frame {} : {} ###########################################\n\n\n".format(start_frame,frame_counter))
				f.write(prev_text)
				f.close
				prev_text = text
				start_frame = frame_counter
			else:
				prev_text = text
			# print(text)
			# final_text.append(text)
	# update the FPS counter
	fps.update()
	# show the output frame
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF
	frame_counter += 1
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# stop the timer and display FPS information
if stuck_in_loop:
	f = open(args["output"],'a')
	f.write("\n\n\n###################################### Frame {} : {} ###########################################\n\n\n".format(start_frame,frame_counter))
	f.write(prev_text)
	f.close
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()

print(final_text)