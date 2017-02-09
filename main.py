import numpy as np
import cv2
from random import shuffle, randrange, sample
from data_preprocessing import process_data, isModelString, isModelFile
typ = 3
data, ind1, ind2 = process_data(
    typ=typ)
testStartIndex = ind1
testStopIndex = ind2
print(testStartIndex,
      testStopIndex)
train_file = np.concatenate(
    [data[:testStartIndex], data[testStopIndex:]])
test_file = data[
    testStartIndex:testStopIndex]
print(train_file.shape,
      test_file.shape)

X_data = []
Y_data = []
countOtherFiles = 0
countIphone = 0
countSamsung = 0
for myFile in train_file:
	if isModelFile(myFile, [['iphone']]):
		countIphone += 1
		image = cv2.imread(myFile, 1).astype(np.uint8)
		if image.shape == (150, 150, 3):
			index = randrange(len(Y_data) + 1)
			Y_data.insert(index, 1)
			X_data.insert(index, image)
	elif isModelFile(myFile, [['samsung']]):
		countSamsung += 1
		image = cv2.imread(myFile, 1).astype(np.uint8)
		if image.shape == (150, 150, 3):
			index = randrange(len(Y_data) + 1)
			Y_data.insert(index, 2)
			X_data.insert(index, image)

	else:
		if countOtherFiles > 6000:
			continue
		countOtherFiles += 1
		image = cv2.imread(myFile, 1).astype(np.uint8)
		if image.shape == (150, 150, 3):
		    index = randrange(
		        len(Y_data) + 1)
		    Y_data.insert(
		        index, 3)
		    X_data.insert(
		        index, image)

print("iphone files: ",
      countIphone)
print("samsung files: ",
      countSamsung)
print("other files: ",
      countOtherFiles)
print('X_data and Y_data length - ',
      len(X_data), len(Y_data))
