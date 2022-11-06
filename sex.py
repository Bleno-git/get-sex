import face_recognition_models
import face_recognition
from PIL import Image
import numpy as np
import random
import pickle
import dlib
import glob
import os

from sklearn.cluster import KMeans
import numpy as np

print("Загружаем модели для преобразования фото")
predictors = []
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
face_recognition_model = face_recognition_models.face_recognition_model_location()
face_detector = dlib.get_frontal_face_detector()

pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

predictors.append([pose_predictor_68_point, face_detector, face_encoder])
print("Модели готовы")


def get(n=5000):
	"""
	n - максимальное кол-во фотографий из каждой папки, которое будем добавлять в модель
	"""

	mails = []
	femails = []
	x = glob.glob('male/*')[:n]
	for i, filename in enumerate(x):
		print(f"Этап 1 (мужчины): {round(i/len(x)*100,2)}%")
		try:
			with open(filename, 'rb') as file:
				img = Image.open(file)
				img = img.convert('RGB')
				img = np.array(img)
			predictor, detector, encoder = predictors[0]
			face_locations = detector(img, 1)
			raw_landmarks = [predictor(img, face_location) for face_location in face_locations]
			face = [np.array(encoder.compute_face_descriptor(img, raw_landmark_set, 1)) for raw_landmark_set in raw_landmarks][0]

			mails.append(face)
		except:
			pass

	x = glob.glob('female/*')[:n]
	for i, filename in enumerate(x):
		print(f"Этап 2 (женщины): {round(i/len(x)*100,2)}%")
		try:
			with open(filename, 'rb') as file:
				img = Image.open(file)
				img = img.convert('RGB')
				img = np.array(img)
			predictor, detector, encoder = random.choice(predictors)
			face_locations = detector(img, 1)
			raw_landmarks = [predictor(img, face_location) for face_location in face_locations]
			face = [np.array(encoder.compute_face_descriptor(img, raw_landmark_set, 1)) for raw_landmark_set in raw_landmarks][0]

			femails.append(face)
		except:
			pass

	return mails, femails

def create_model():
	mails, femails = get()

	clf = KMeans(n_clusters=2)
	clf.fit(mails + femails)

	with open('model.pickle', 'wb') as file:
		pickle.dump(clf, file)

	return clf


def load_model():
	with open('model.pickle', 'rb') as file:
		clf = pickle.load(file)
	return clf


def test():
	clf = load_model()
	for fn in glob.glob('tests/*'):
		name = fn.split('\\')[-1].split('.')[0]
		with open(fn, 'rb') as file:
			img = Image.open(file)
			img = img.convert('RGB')
			img = np.array(img)

		predictor, detector, encoder = random.choice(predictors)
		face_locations = detector(img, 1)
		raw_landmarks = [predictor(img, face_location) for face_location in face_locations]
		face = [np.array(encoder.compute_face_descriptor(img, raw_landmark_set, 1)) for raw_landmark_set in raw_landmarks][0]


		print(f"{name}: {'Мужчина' if clf.predict([face]) == 1 else 'Женщина'}")

create_model()
test()

