import os
import os.path
import numpy as np
import skimage
import skimage.io
import skimage.transform
import skimage.color
import random
import pandas as pd
import cv2

TRAIN_SET = 1
VALIDATION_SET = 2
TEST_SET = 3

class InputReader:
	def __init__(self, options):
		self.options = options

		# Reads pathes of images together with their labels
		# self.imageList, self.labelList = self.readImageNames(self.options.trainFileName)
		# self.imageListTest, self.labelListTest = self.readImageNames(self.options.testFileName)

		trainImagesFile = options.dataDir + "trainImages.npy"
		trainLabelsFile = options.dataDir + "trainLabels.npy"
		validationImagesFile = options.dataDir + "validationImages.npy"
		validationLabelsFile = options.dataDir + "validationLabels.npy"
		testImagesFile = options.dataDir + "testImages.npy"

		if os.path.isfile(trainImagesFile):
			print ("Loading saved data files")

			# Load the arrays from file
			self.trainImages = np.load(trainImagesFile)
			self.trainLabels = np.load(trainLabelsFile)
			self.validationImages = np.load(validationImagesFile)
			self.validationLabels = np.load(validationLabelsFile)
			self.testImages = np.load(testImagesFile)

		else:
			print ("Creating validation set")

			# Load the files from the data directory
			trainData = pd.read_csv(options.dataDir + "train.csv", sep=",", header=0)
			testData = pd.read_csv(options.dataDir + "test.csv", sep=",", header=0)

			trainMatrix = trainData.as_matrix()
			self.testImages = testData.as_matrix()

			labelData = trainMatrix[:, 0]
			self.trainImages = []
			self.validationImages = []
			self.trainLabels = []
			self.validationLabels = []

			# Create validation set using specified percentage of data
			for i in range(options.numClasses):
				# Get all the indices of examples where the class label is equal to the current class
				currentClassElements = np.nonzero(labelData == i)[0]
				valIndices = np.random.choice(len(currentClassElements), int(len(currentClassElements) * options.validationDataPercentage), replace=False)
				for j in range(len(currentClassElements)):
					idx = currentClassElements[j]
					if j in valIndices:
						self.validationImages.append(trainMatrix[idx, 1:])
						self.validationLabels.append(i)
					else:
						self.trainImages.append(trainMatrix[idx, 1:])
						self.trainLabels.append(i)

			self.trainImages = np.array(self.trainImages)
			self.trainLabels = np.array(self.trainLabels)
			self.validationImages = np.array(self.validationImages)
			self.validationLabels = np.array(self.validationLabels)

			# Save the current split for later reuse
			np.save(trainImagesFile, self.trainImages)
			np.save(trainLabelsFile, self.trainLabels)
			np.save(validationImagesFile, self.validationImages)
			np.save(validationLabelsFile, self.validationLabels)
			np.save(testImagesFile, self.testImages)

		print ("Train data shape: %s" % (str(self.trainImages.shape)))
		print ("Train label shape: %s" % (str(self.trainLabels.shape)))
		print ("Validation data shape: %s" % (str(self.validationImages.shape)))
		print ("Validation label shape: %s" % (str(self.validationLabels.shape)))
		print ("Test data shape: %s" % (str(self.testImages.shape)))

		self.currentIndex = 0
		self.currentIndexTest = 0
		self.currentIndexValidation = 0
		self.totalEpochs = 0
		self.totalImages = self.trainImages.shape[0] # len(self.imageList)
		self.totalImagesTest = self.testImages.shape[0] # len(self.imageListTest)
		self.totalImagesValidation = self.validationImages.shape[0]

		self.imgShape = [self.options.imageHeight, self.options.imageWidth, self.options.imageChannels]
		self.imgShapeTrain = [32, 32, self.options.imageChannels]
		self.epsilon = 1e-8

		if options.computeMeanImage:
			meanFile = options.dataDir + "meanImg.npy"
			stdFile = options.dataDir + "stdImg.npy"
			if os.path.isfile(meanFile) and os.path.isfile(stdFile):
				self.meanImg = np.load(meanFile)
				self.stdImg = np.load(stdFile)
				print ("Image mean: %s" % str(self.meanImg))
				print ("Image std: %s" % str(self.stdImg))
			else:
				print ("Computing mean image from training dataset")
				# Compute mean image
				meanImg = np.zeros([options.imageHeight, options.imageWidth, options.imageChannels])
				stdImg = np.zeros([options.imageHeight, options.imageWidth, options.imageChannels])
				imagesProcessed = 0
				for i in range(self.trainImages.shape[0]):
					img = self.trainImages[i]

					if img.shape != self.imgShape:
						img = np.reshape(img, self.imgShape)
					
					img = img.astype(float)
					meanImg += img
					imagesProcessed += 1

				self.meanImg = meanImg / imagesProcessed
				np.save(options.dataDir + "fullImageMean.npy", self.meanImg)

				imagesProcessed = 0
				for i in range(self.trainImages.shape[0]):
					img = self.trainImages[i]

					if img.shape != self.imgShape:
						img = np.reshape(img, self.imgShape)
					
					img = img.astype(float)
					stdImg += np.square(img - self.meanImg)
					imagesProcessed += 1

				self.stdImg = stdImg / imagesProcessed
				np.save(options.dataDir + "fullImageStd.npy", self.stdImg)

				# Convert mean to per channel mean (single channel images)
				self.meanImg = np.mean(np.mean(self.meanImg, axis=0), axis=0)
				np.save(meanFile, self.meanImg)
				self.stdImg = np.mean(np.mean(self.stdImg, axis=0), axis=0)
				self.stdImg = np.sqrt(self.stdImg)
				np.save(stdFile, self.stdImg)

				print ("Mean image and std computed")

	def readImagesFromDisk(self, indices, imageSet):
		"""Consumes a list of indices and returns images
		Args:
		  indices: List of image file index
		Returns:
		  4-D numpy array: The input images
		"""
		images = []
		for i in range(0, len(indices)):
			if self.options.verbose > 1:
				print ("Image: %s" % fileNames[i])

			if imageSet == TRAIN_SET:
				img = self.trainImages[indices[i]]
				img = img.astype(float)

				if img.shape != self.imgShapeTrain:
					img = np.reshape(img, self.imgShape)

				# Take a random crop from the image
				# randY = random.randint(0, self.imgShapeTrain[0] - self.options.imageHeight - 1)
				# randX = random.randint(0, self.imgShapeTrain[1] - self.options.imageWidth - 1)
				# img = img[randY:(randY+self.options.imageHeight), randX:(randX+self.options.imageWidth)]

				if self.options.computeMeanImage:
					img = (img - self.meanImg) / (self.stdImg)
				
				if random.random() > 0.5:
					img = np.fliplr(img)

			else:
				if imageSet == VALIDATION_SET:
					img = self.validationImages[indices[i]]
				elif imageSet == TEST_SET:
					img = self.testImages[indices[i]]
				else:
					print ("Error: Incorrect set selected")
					exit(-1)

				img = img.astype(float)

				if img.shape != self.imgShape:
					# img = skimage.transform.resize(img, self.imgShape, preserve_range=True)
					img = np.reshape(img, self.imgShape)

				if self.options.computeMeanImage:
					img = (img - self.meanImg) / (self.stdImg)

			images.append(img)

		# Convert list to ndarray
		images = np.array(images)
		return images

	def getTrainBatch(self):
		"""Returns training images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: training images and labels in batch.
		"""
		if self.totalEpochs >= self.options.trainingEpochs:
			return None, None

		endIndex = self.currentIndex + self.options.batchSize
		if self.options.sequentialFetch:
			# Fetch the next sequence of images
			self.indices = np.arange(self.currentIndex, endIndex)

			if endIndex > self.totalImages:
				# Replace the indices which overshot with 0
				self.indices[self.indices >= self.totalImages] = np.arange(0, np.sum(self.indices >= self.totalImages))
		else:
			# Randomly fetch any images
			self.indices = np.random.choice(self.totalImages, self.options.batchSize, replace=False)

		imagesBatch = self.readImagesFromDisk(self.indices, imageSet=TRAIN_SET)
		labelsBatch = self.convertLabelsToOneHot(self.indices, imageSet=TRAIN_SET)

		self.currentIndex = endIndex
		if self.currentIndex > self.totalImages:
			print ("Training epochs completed: %f" % (self.totalEpochs + (float(self.currentIndex) / self.totalImages)))
			self.currentIndex = self.currentIndex - self.totalImages
			self.totalEpochs = self.totalEpochs + 1

			# Shuffle the image list if not random sampling at each stage
			if self.options.sequentialFetch:
				np.random.shuffle(self.imageList)

		return imagesBatch, labelsBatch

	def resetValidationBatchIndex(self):
		self.currentIndexValidation = 0

	def resetTestBatchIndex(self):
		self.currentIndexTest = 0

	def getValidationBatch(self):
		"""Returns validation images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: validation images and labels in batch.
		"""
		if self.currentIndexValidation >= self.totalImagesValidation:
			return None, None

		if self.options.randomFetchTest:
			self.indices = np.random.choice(self.totalImagesValidation, self.options.batchSize, replace=False)
			imagesBatch = self.readImagesFromDisk(self.indices, imageSet=VALIDATION_SET)
			labelsBatch = self.convertLabelsToOneHot(self.indices, imageSet=VALIDATION_SET)

		else:
			endIndex = self.currentIndexValidation + self.options.batchSize
			if endIndex > self.totalImagesValidation:
				endIndex = self.totalImagesValidation
			self.indices = np.arange(self.currentIndexValidation, endIndex)
			imagesBatch = self.readImagesFromDisk(self.indices, imageSet=VALIDATION_SET)
			labelsBatch = self.convertLabelsToOneHot(self.indices, imageSet=VALIDATION_SET)
			self.currentIndexValidation = endIndex

		return imagesBatch, labelsBatch

	def getTestBatch(self):
		"""Returns testing images
		Args:
		  None
		Returns:
		  Two numpy arrays: test images in batch.
		"""
		if self.currentIndexTest >= self.totalImagesTest:
			return None

		if self.options.randomFetchTest:
			self.indices = np.random.choice(self.totalImagesTest, self.options.batchSize, replace=False)
			imagesBatch = self.readImagesFromDisk(self.indices, imageSet=TEST_SET)
			# labelsBatch = self.convertLabelsToOneHot(self.indices)

		else:
			endIndex = self.currentIndexTest + self.options.batchSize
			if endIndex > self.totalImagesTest:
				endIndex = self.totalImagesTest
			self.indices = np.arange(self.currentIndexTest, endIndex)
			imagesBatch = self.readImagesFromDisk(self.indices, imageSet=TEST_SET)
			# labelsBatch = self.convertLabelsToOneHot(self.indices)
			self.currentIndexTest = endIndex

		return imagesBatch

	def restoreCheckpoint(self, numSteps):
		"""Restores current index and epochs using numSteps
		Args:
		  numSteps: Number of batches processed
		Returns:
		  None
		"""
		processedImages = numSteps * self.options.batchSize
		self.totalEpochs = processedImages / self.totalImages
		self.currentIndex = processedImages % self.totalImages

	def convertLabelsToOneHot(self, indices, imageSet):
		oneHotLabels = []
		numElements = self.options.numClasses
		
		for idx in indices:
			oneHotVector = np.zeros(numElements)
			if imageSet == TRAIN_SET:
				oneHotVector[self.trainLabels[idx]] = 1
			elif imageSet == VALIDATION_SET:
				oneHotVector[self.validationLabels[idx]] = 1
			elif imageSet == TEST_SET:
				print ("Error: Test data has no labels")
				exit(-1)
				oneHotVector[self.testLabels[idx]] = 1
			else:
				print ("Error: Incorrect set selected")
				exit(-1)
			oneHotLabels.append(oneHotVector)

		oneHotLabels = np.array(oneHotLabels)
		return oneHotLabels