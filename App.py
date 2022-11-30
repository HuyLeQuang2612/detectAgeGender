import numpy as np
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, Response
from PIL import Image
from werkzeug.utils import secure_filename
from mtcnn.mtcnn import MTCNN
from detect import predict, draw_box_faces, crop_faces
import tensorflow as tf


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getGender(pred_sex):
    result = ""
    index_maxValue = pred_sex.argmax()
    pred_gender = pred_sex[0][0]
    gender = "Male"

    if index_maxValue == 1:
        pred_gender = pred_sex[0][1]
        gender = "Female"

    percent = str(round(pred_gender * 100, 2))
    result = gender + "(" + percent + "%)"

    return result


def getRangeAge(pred_age):
    arrRangeAge = ["1-14", "15-25", "26-40", "41-60", "61-116"]
    maxIndex = list(pred_age[0]).index(max(pred_age[0]))
    rangeAge = arrRangeAge[maxIndex]
    percent = str(round(pred_age[0][maxIndex] * 100, 2))
    
    return rangeAge + "(" + percent + "%)"


def crop_face_video(image, result):
    nb_detected_faces = len(result)

    cropped_face = np.empty((nb_detected_faces, 48, 48))
    boxes = []
    # loop through detected face    
    for i in range(nb_detected_faces):
        # coordinates of boxes
        bounding_box = result[i]['box']
        left, top = bounding_box[0], bounding_box[1]
        right, bottom = bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]

        # coordinates of cropped image
        x1_crop = max(int(left), 0)
        y1_crop = max(int(top), 0)
        x2_crop = int(right)
        y2_crop = int(bottom)

        face = image[y1_crop:y2_crop, x1_crop:x2_crop, :]
        face = cv2.resize(face, (48, 48), cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        cropped_face[i, :, :] = face
        boxes.append((x1_crop, y1_crop, x2_crop, y2_crop))

    return cropped_face, boxes


def draw_labels_and_boxes(img, boxes, labels, margin=0):
    for i in range(len(labels)):
        # get the bounding box coordinates
        left, top, right, bottom = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        width = right - left
        height = bottom - top
        img_h, img_w = img.shape[:2]

        x1 = max(int(left - margin * width), 0)
        y1 = max(int(top - margin * height), 0)
        x2 = min(int(right + margin * width), img_w - 1)
        y2 = min(int(bottom + margin * height), img_h - 1)

        # Color green
        color = (0, 255, 0)

        # classify label according to result
        age_label, gender_label = labels[i]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, gender_label, (left - 40, top - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, age_label, (left - 40, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img


def gen_frames():  # generate frame by frame from camera
	global camera
	if not camera.isOpened():
		camera = cv2.VideoCapture(0)

	while True:
        # Capture frame-by-frame
		success, frame = camera.read()  # read the camera frame
		if not success:
			print("Can't receive frame (stream end?). Exiting ...")
			break
		else:
			# detect faces
			result = detect_model.detect_faces(frame)

        	# cropped face
			cropped_face, boxes = crop_face_video(frame, result)

			# predict
			predicted_result = []
			for i in range(cropped_face.shape[0]):
				arr3d = np.zeros((48, 48, 3))
				arr3d[:, :, 0] = arr3d[:, :, 1] = arr3d[:, :, 2] = cropped_face[i]
				pixels = arr3d
				pred = multitask_model.predict(np.array([pixels]))
				predicted_result.append((getRangeAge(pred[0]), getGender(pred[1])))

			# draw label and boxes
			frame = draw_labels_and_boxes(frame, boxes, predicted_result)
			
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

INDEX_FACE = 0
filename = ""
faceFiles = []
output = []

# load model detect faces
detect_model = MTCNN()
# load model multitask learning
multitask_model = tf.keras.models.load_model('./saved_model/efficientNetB2_weight.h5')

streaming = True
camera = cv2.VideoCapture(0)

@app.route('/')
def index_view():
	return render_template('index.html', imgFile="detectedFace.jpg", faceFile="face.jpg", gender="Male", age="26-40 years old")

@app.route('/', methods = ['POST'])
def result():
	global INDEX_FACE
	global filename
	global faceFiles
	global output

	INDEX_FACE = 0
	faceFiles = []
	output = []
	keep = ("detectedFace.jpg", "face.jpg")

	for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
		for f in files:
			if f in keep:
				continue
			else:
				path = os.path.join(root, f)
				os.remove(path)

	if 'file' not in request.files:
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(path)
		
		loadImg = Image.open(path)
		pixels = cv2.imread(path)
		pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)

		faces = detect_model.detect_faces(pixels)
		if faces == []:
			loadImg.close()
			os.remove(path)
			return redirect(request.url)

		cropped_images = crop_faces(loadImg, faces)
		arrFile = filename.split(".")

		for i, face in enumerate(cropped_images):
			faceFile = arrFile[0] + "_face" + str(i) + "." + arrFile[1]
			faceFiles.append(faceFile)
			path = os.path.join(app.config['UPLOAD_FOLDER'], faceFile)
			face.save(path)

			pixels = cv2.imread(path)
			pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
			pixels = cv2.resize(pixels, (48, 48))
			
			# Predict age, gender
			age, gender = predict(pixels)
			output.append((age, gender))

		draw_box_faces(loadImg, faces)
		
		filename = arrFile[0] + "_detectedFaces" + "." + arrFile[1]
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		loadImg.save(path)

		return render_template('index.html', imgFile=filename, faceFile=faceFiles[INDEX_FACE], age=output[INDEX_FACE][0], gender=output[INDEX_FACE][1])

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/webcam', methods = ['POST'])
def webcam_view():
	return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
	if camera.isOpened():
		print("Releasing cam feed")
		camera.release()
	return ""

@app.route('/next_face')
def next_face():
	global INDEX_FACE
	if len(faceFiles) == 0:
		return render_template('index.html', imgFile="detectedFace.jpg", faceFile="face.jpg", gender="Male", age="26-40 years old")
	
	if INDEX_FACE <= (len(faceFiles) - 2):
		INDEX_FACE += 1
	else:
		INDEX_FACE = 0

	return render_template('index.html', imgFile=filename, faceFile=faceFiles[INDEX_FACE], age=output[INDEX_FACE][0], gender=output[INDEX_FACE][1])

if __name__ == '__main__':
    app.run(debug=True, port=8000)
