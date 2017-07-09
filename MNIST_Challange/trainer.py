import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from optparse import OptionParser
import datetime as dt

# Import CNN model
from mnet import *

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--dataDir", action="store", type="string", dest="dataDir", default="../Data/MNIST/", help="Directory containing the MNIST data")
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=28, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=28, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=1, help="Number of channels in image for feeding into the network")
parser.add_option("--sequentialFetch", action="store_true", dest="sequentialFetch", default=False, help="Sequentially fetch images for each batch")
parser.add_option("--randomFetchTest", action="store_true", dest="randomFetchTest", default=False, help="Randomly fetch images for each test batch")
parser.add_option("--computeMeanImage", action="store_false", dest="computeMeanImage", default=True, help="Compute mean image on data")
parser.add_option("--validationDataPercentage", action="store", type="float", dest="validationDataPercentage", default=0.05, help="Percentage of data to use for validation")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-3, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=50, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=5000, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=5, help="Progress evaluation step")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./checkpoints/", help="Directory for saving the model")
parser.add_option("--bestModelDir", action="store", type="string", dest="bestModelDir", default="./best-checkpoint/", help="Directory for saving the best model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="mnet", help="Name to be used for saving the model")

# Network Params
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=10, help="Number of classes")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Import custom data
import inputReader
inputReader = inputReader.InputReader(options)

if options.trainModel:
	with tf.variable_scope('Model'):
		# Data placeholders
		inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, options.imageHeight,
											options.imageWidth, options.imageChannels], name="inputBatchImages")
		inputBatchLabels = tf.placeholder(dtype=tf.float32, shape=[None, options.numClasses], name="inputBatchLabels")
		inputKeepProbability = tf.placeholder(dtype=tf.float32, name="inputKeepProbability")

	# Instantiate MNET
	with slim.arg_scope(mnet_arg_scope()):
		logits, end_points = lenet(inputBatchImages, inputKeepProbability, options.numClasses)

		tf.summary.tensor_summary("Prob", end_points["Predictions"])
		tf.summary.image("MNIST", inputBatchImages)

	with tf.name_scope('Loss'):
		# Define loss
		# slim.losses.softmax_cross_entropy(onehot_labels=inputBatchLabels, logits=logits)
		# loss = slim.losses.get_total_loss() # XE automatically added to the list by Slim
		tf.losses.softmax_cross_entropy(onehot_labels=inputBatchLabels, logits=logits)
		loss = tf.reduce_mean(tf.losses.get_losses())
		# loss = tf.losses.get_total_loss()

	with tf.name_scope('Accuracy'):
		predictedLabel = tf.argmax(end_points['Predictions'], 1, name="predictedLabel")
		correct_predictions = tf.equal(predictedLabel, tf.argmax(inputBatchLabels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

	with tf.name_scope('Optimizer'):
		# Define Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)
		# optimizer = tf.train.MomentumOptimizer(learning_rate=options.learningRate, momentum=0.9, use_nesterov=True).minimize(loss)

		# Op to calculate every variable gradient
		gradients = tf.gradients(loss, tf.trainable_variables())
		gradients = list(zip(gradients, tf.trainable_variables()))

		# Op to update all variables according to their gradient
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

	# Initializing the variables
	init = tf.global_variables_initializer()
	init_local = tf.local_variables_initializer()

	if options.tensorboardVisualization:
		# Create a summary to monitor cost tensor
		tf.summary.scalar("loss", loss)

		# Create summaries to visualize weights
		for var in tf.trainable_variables():
		    tf.summary.histogram(var.name, var)
		# Summarize all gradients
		for grad, var in gradients:
		    tf.summary.histogram(var.name + '/gradient', grad)

		# Merge all summaries into a single op
		mergedSummaryOp = tf.summary.merge_all()

	# 'Saver' op to save and restore all the variables
	saver = tf.train.Saver()
	# bestModelSaver = tf.train.Saver()

	bestAcc = 0.0
	step = 1

# Train model
if options.trainModel:
	with tf.Session() as sess:
		# Initialize all variables
		sess.run(init)
		sess.run(init_local)

		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.modelDir)
			os.system("rm -rf " + options.bestModelDir)
			os.system("mkdir " + options.modelDir)
			os.system("mkdir " + options.bestModelDir)

		# Restore checkpoint
		else:
			print ("Restoring from checkpoint")
			# saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
			saver.restore(sess, options.modelDir + options.modelName)

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

		print ("Starting network training")

		# Keep training until reach max iterations
		while True:
			batchImagesTrain, batchLabelsTrain = inputReader.getTrainBatch()
			# print ("Batch images shape: %s, Batch labels shape: %s" % (str(batchImagesTrain.shape), str(batchLabelsTrain.shape)))

			# If training iterations completed
			if batchImagesTrain is None:
				print ("Training completed")
				break

			# Run optimization op (backprop)
			if options.tensorboardVisualization:
				[trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
				# Write logs at every iteration
				summaryWriter.add_summary(summary, step)
			else:
				[trainLoss, currentAcc, _] = sess.run([loss, accuracy, applyGradients], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})

			print ("Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (step, trainLoss, currentAcc * 100))
			step += 1

			if step % options.saveStep == 0:
				# Save model weights to disk
				saver.save(sess, options.modelDir + options.modelName)
				print ("Model saved: %s" % (options.modelDir + options.modelName))

			#Check the accuracy on test data
			if step % options.evaluateStep == 0:
				numBatches = 0
				cummulativeAccuracy = 0
				inputReader.resetValidationBatchIndex()
				while True:
					# Report loss on test data
					batchImagesTest, batchLabelsTest = inputReader.getValidationBatch()
					if batchImagesTest is None:
						break

					[testLoss, testAcc] = sess.run([loss, accuracy], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})	
					print ("(Test) Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (numBatches+1, testLoss, testAcc * 100))
					numBatches += 1
					cummulativeAccuracy += testAcc

				testAcc = cummulativeAccuracy / numBatches
				# If its the best loss achieved so far, save the model
				if testAcc > bestAcc:
					bestAcc = testAcc
					# bestModelSaver.save(sess, options.bestModelDir + options.modelName, global_step=0, latest_filename=checkpointStateName)
					saver.save(sess, options.bestModelDir + options.modelName)
					print ("Best best model found with accuracy of: %.2f" % (bestAcc * 100))
					print ("Best model saved in file: %s" % (options.bestModelDir + options.modelName))
				else:
					print ("Current test accuracy: %f" % (testAcc * 100))
					print ("Previous best accuracy: %f" % (bestAcc * 100))

		# Save final model weights to disk
		saver.save(sess, options.modelDir + options.modelName)
		print ("Model saved: %s" % (options.modelDir + options.modelName))

		# Restore the best model
		saver.restore(sess, options.bestModelDir + options.modelName)

		# Compute the final validation accuracy
		numBatches = 0
		cummulativeAccuracy = 0
		inputReader.resetValidationBatchIndex()
		while True:
			# Report loss on test data
			batchImagesTest, batchLabelsTest = inputReader.getValidationBatch()
			if batchImagesTest is None:
				break

			[testLoss, testAcc] = sess.run([loss, accuracy], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})	
			print ("(Test) Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (numBatches+1, testLoss, testAcc * 100))
			numBatches += 1
			cummulativeAccuracy += testAcc

		testAcc = cummulativeAccuracy / numBatches
		print ("Cummulative validation accuracy: %f" % (testAcc * 100))

		# Train the best model on the held out validation set
		validationSetEpochs = 3 # Epochs on the validation set
		for i in range(validationSetEpochs):
			inputReader.resetValidationBatchIndex()
			while True:
				batchImagesTrain, batchLabelsTrain = inputReader.getValidationBatch()
				if batchImagesTrain is None:
					break

				# Run optimization op (backprop)
				if options.tensorboardVisualization:
					[trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
					# Write logs at every iteration
					summaryWriter.add_summary(summary, step)
				else:
					[trainLoss, currentAcc, _] = sess.run([loss, accuracy, applyGradients], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})

				print ("(Validation training) Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (step, trainLoss, currentAcc * 100))

		# Compute the final validation accuracy
		numBatches = 0
		cummulativeAccuracy = 0
		inputReader.resetValidationBatchIndex()
		while True:
			# Report loss on test data
			batchImagesTest, batchLabelsTest = inputReader.getValidationBatch()
			if batchImagesTest is None:
				break

			[testLoss, testAcc] = sess.run([loss, accuracy], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})	
			print ("(Test) Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (numBatches+1, testLoss, testAcc * 100))
			numBatches += 1
			cummulativeAccuracy += testAcc

		testAcc = cummulativeAccuracy / numBatches
		print ("Validation accuracy (after training): %f" % (testAcc * 100))

		# Save final model weights to disk
		saver.save(sess, options.modelDir + options.modelName)
		print ("Final model saved: %s" % (options.modelDir + options.modelName))

		print ("Optimization Finished!")

# Test model
if options.testModel:
	print ("Using saved model for test set predictions")
	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session() as session:
		saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
		saver.restore(session, options.modelDir + options.modelName)

		# Get reference to placeholders
		predictionsNode = session.graph.get_tensor_by_name("Predictions:0")
		predictedLabelNode = session.graph.get_tensor_by_name("Accuracy/predictedLabel:0")
		accuracyNode = session.graph.get_tensor_by_name("Accuracy/accuracy:0")
		inputBatchImages = session.graph.get_tensor_by_name("Model/inputBatchImages:0")
		inputBatchLabels = session.graph.get_tensor_by_name("Model/inputBatchLabels:0")
		inputKeepProbability = session.graph.get_tensor_by_name("Model/inputKeepProbability:0")

		# Make predictions for validation set
		numBatches = 0
		cummulativeAccuracy = 0
		inputReader.resetValidationBatchIndex()
		while True:
			# Report loss on test data
			batchImagesTest, batchLabelsTest = inputReader.getValidationBatch()
			if batchImagesTest is None:
				break

			[testAcc] = session.run([accuracyNode], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})	
			print ("(Validation) Iteration: %d, Accuracy: %f" % (numBatches+1, testAcc * 100))
			numBatches += 1
			cummulativeAccuracy += testAcc

		testAcc = cummulativeAccuracy / numBatches
		print ("Cummulative validation accuracy: %f" % (testAcc * 100))

		# Open the predictions file
		inputReader.resetTestBatchIndex()
		numBatches = 0
		fileIndex = 1
		with open("Predictions.csv", "w") as outputPredictionFile:
			outputPredictionFile.write("ImageId,Label\n")
			while True:
				batchImagesTest = inputReader.getTestBatch()
				if batchImagesTest is None:
					break
				print ("Processing batch number %d" % (numBatches + 1))
				[predictions, predictedLabel] = session.run([predictionsNode, predictedLabelNode], feed_dict={inputBatchImages: batchImagesTest, inputKeepProbability: 1.0})
				for i in range(predictions.shape[0]):
					npLabel = np.argmax(predictions[i, :])
					assert(npLabel == predictedLabel[i])
					outputPredictionFile.write(str(fileIndex) + "," + str(npLabel) + "\n")
					fileIndex += 1
				
				numBatches += 1

	print ("Predictions saved to file")
