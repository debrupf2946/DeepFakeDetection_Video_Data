from components.preprocessing.pixel_distrubution_based_kfs import sample_key_frames
from models.gru_based import get_model
import keras
import cv2
import numpy as np 

key_frames=sample_key_frames("data/Random/aagfhgtpmv.mp4")

model=get_model()

model.load_weights("artifacts/deep_fake_GRU.weights.h5")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
resized_image = cv2.resize(key_frames[0], (299, 299)) 
input_image = np.expand_dims(resized_image, axis=0)


predictions = model.predict(input_image)

print(predictions)

