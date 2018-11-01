import numpy as np
import itertools
import math
import time
import matplotlib.pyplot as plt
import tensorflow as tf

def shuffleArrays(arrays):
	rngState = np.random.get_state()
	for array in arrays:
		np.random.set_state(rngState)
		np.random.shuffle(array)

class myTimer:
	def __init__(self):
		self.reset()

	def reset(self):
		self.initTime = time.time()

	def getSec(self):
		return time.time() - self.initTime

	def getHMS(self):
		return self.secToHMS(self.getSec())

	def secToHMS(self, sec):
		sec = round(sec)
		h = sec//(60*60)
		sec = sec%(60*60)
		m = sec//60
		s = sec%60
		return h, m, s

class Dictionary:
	def __init__(self):
		self.punctuationSet = set('.,;:?!')
		self.noneTxt = '_none'
		self.startTxt = '_start'
		self.stopTxt = '_stop'

		self.nextIndex = 0
		self.dict = {}
		self.revDict = {}

		self(self.noneTxt)
		self.startIndex = self(self.startTxt)
		self.stopIndex = self(self.stopTxt)
		self.noneIndex = self(self.noneTxt)

		self.punctuationIndexSet = set()
		for char in self.punctuationSet:
			self.punctuationIndexSet.add(self(char))

	def __call__(self, word):
		if word not in self.dict:
			self.dict[word] = self.nextIndex
			self.revDict[self.nextIndex] = word
			self.nextIndex += 1
		return self.dict[word]

	def rev(self, index):
		return self.revDict[index]

	def wordCount(self):
		return len(self.dict)

	def txtToIndices(self, txt):
		indices = [self.startIndex]
		for c in txt:
			indices.append(self(c))
		indices.append(self.stopIndex)
		return indices

		outWords = [self.startTxt]
		innWords = txt.split()
		for word in innWords:
			if word[-1] in self.punctuationSet:
				outWords.append(word[-1])
				outWords.append(word[:-1])
		outWords.append(self.stopTxt)
		return [self(word) for word in outWords]

	def indicesToTxt(self, indices):
		txt = ''
		for i in indices:
			if i != self.startIndex and i != self.stopIndex and i != self.noneIndex:
				txt += self.rev(i)
		return txt

		outTxt = ''
		i = 0

		if indices[0] == self.startIndex:
			i+=1

		while i<len(indices) and indices[i] != self.stopIndex:
			outTxt += self.rev(indices[i])
			i += 1

			if i<len(indices) and indices[i] != self.stopIndex and indices[i] not in self.punctuationIndexSet:
				outTxt += ' '
		return outTxt

class DataSet:
	def __init__(self, filePath, batchSize, testFrac=0.0):
		self.enDict = Dictionary()
		self.noDict = Dictionary()
		enLines = []
		noLines = []

		with open(filePath, 'r', encoding='utf-8') as dataFile:
			for line in dataFile:
				en, no = line.lower().split('\t')
				enLines.append(self.enDict.txtToIndices(en))
				noLines.append(self.noDict.txtToIndices(no))

		nLines = min(len(enLines), len(noLines))
		maxEnLineLength = max([len(line) for line in enLines])
		maxNoLineLength = max([len(line) for line in noLines])

		self.encoderInputData = np.zeros([nLines, maxEnLineLength, self.enDict.wordCount()])
		self.decoderInputData = np.zeros([nLines, maxNoLineLength-1, self.noDict.wordCount()])
		self.targetOutputData = np.zeros([nLines, maxNoLineLength-1, self.noDict.wordCount()])

		for lineIndex in range(len(enLines)):
			line = enLines[lineIndex]
			for wordIndex in range(len(line)):
				word = line[wordIndex]
				self.encoderInputData[lineIndex, wordIndex, word] = 1

		for lineIndex in range(len(noLines)):
			line = noLines[lineIndex]
			for wordIndex in range(len(line)-1):
				innWord = line[wordIndex]
				self.decoderInputData[lineIndex, wordIndex, innWord] = 1
				outWord = line[wordIndex+1]
				self.targetOutputData[lineIndex, wordIndex, outWord] = 1

		nTestSamples = int(nLines*testFrac)
		self.nSamples = nLines-nTestSamples
		shuffleArrays([self.encoderInputData, self.decoderInputData, self.targetOutputData])

		self.encoderInputTest = self.encoderInputData[:nTestSamples]
		self.decoderInputTest = self.decoderInputData[:nTestSamples]
		self.targetOutputTest = self.targetOutputData[:nTestSamples]

		self.encoderInputData = self.encoderInputData[nTestSamples:]
		self.decoderInputData = self.decoderInputData[nTestSamples:]
		self.targetOutputData = self.targetOutputData[nTestSamples:]

		self.batchSize = batchSize
		self.genBatches()

	def genBatches(self):
		lastBatchLen = self.nSamples % self.batchSize
		nBatches = self.nSamples//self.batchSize
		shuffleArrays([self.encoderInputData, self.decoderInputData, self.targetOutputData])

		self.encoderInputBatches = np.split(self.encoderInputData[lastBatchLen:], nBatches)
		self.decoderInputBatches = np.split(self.decoderInputData[lastBatchLen:], nBatches)
		self.targetOutputBatches = np.split(self.targetOutputData[lastBatchLen:], nBatches)

		if lastBatchLen != 0:
			self.encoderInputBatches.append(self.encoderInputData[:lastBatchLen])
			self.decoderInputBatches.append(self.decoderInputData[:lastBatchLen])
			self.targetOutputBatches.append(self.targetOutputData[:lastBatchLen])

	def getBatch(self):
		if len(self.encoderInputBatches) == 0:
			self.genBatches()
		return self.encoderInputBatches.pop(), self.decoderInputBatches.pop(), self.targetOutputBatches.pop()

	def getTest(self):
		return self.encoderInputTest, self.decoderInputTest, self.targetOutputTest

class RNN:
	def __init__(self, dataSet, fileName=None):
		self.dataSet = dataSet
		self.enDict = self.dataSet.enDict
		self.noDict = self.dataSet.noDict
		enWordCount = self.enDict.wordCount()
		noWordCount = self.noDict.wordCount()

		if fileName is not None:
			self.model = tf.keras.models.load_model(fileName)

		else:
			layerDepth = 64

			inputLayer = tf.keras.layers.Input(shape=[None, enWordCount])
			_, encodingState = tf.keras.layers.GRU(layerDepth, return_state=True)(inputLayer)

			decoderInput = tf.keras.layers.Input(shape=[None, noWordCount])
			decoderOutput = tf.keras.layers.GRU(layerDepth, return_sequences=True)(decoderInput, initial_state=encodingState)
			outputLayer = tf.keras.layers.Dense(noWordCount, activation='softmax')(decoderOutput)

			self.model = tf.keras.models.Model([inputLayer, decoderInput], outputLayer)
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def __call__(self, enTxt, maxLen=0):
		enIndices = self.enDict.txtToIndices(enTxt.lower())
		enData = np.zeros([1, len(enIndices), self.enDict.wordCount()])
		for i, index in zip(itertools.count(), enIndices):
			enData[0, i, index] = 1

		noIndices = []
		currentIndex = self.noDict.startIndex
		while currentIndex != self.noDict.stopIndex and (len(noIndices) < maxLen or maxLen == 0):
			noIndices.append(currentIndex)
			noData = np.zeros([1, len(noIndices), self.noDict.wordCount()])
			for i, index in zip(itertools.count(), noIndices):
				noData[0, i, index] = 1
			currentIndex = np.argmax(self.model.predict([enData, noData])[0, -1])
		noIndices.append(currentIndex)

		return self.noDict.indicesToTxt(noIndices)

	def test(self):
		encoderInn, decoderInn, targetOut = self.dataSet.getTest()
		return self.model.test_on_batch([encoderInn, decoderInn], targetOut)

	def trainOneStep(self):
		encoderInn, decoderInn, targetOut = self.dataSet.getBatch()
		return self.model.train_on_batch([encoderInn, decoderInn], targetOut)

	def trainForSteps(self, steps):
		timer = myTimer()

		for step in range(1, steps+1):
			acc = self.trainOneStep()
			print("Step: {}/{}".format(step, steps))
			print("Elapsed time: {}:{:02d}:{:02d}".format(*timer.getHMS()))
			print("Training loss: {:.3f}".format(acc))
			print("")

		acc = self.test()
		print("Test loss: {:.3f}".format(acc))

	def trainForTime(self, hours, minutes, seconds):
		sec = (hours*60+minutes)*60+seconds
		timer = myTimer()
		step = 0

		while timer.getSec() < sec:
			acc = self.trainOneStep()
			step+=1
			print("Step: {}".format(step))
			print("Elapsed time: {}:{:02d}:{:02d} / {}:{:02d}:{:02d}".format(*timer.getHMS(), *timer.secToHMS(sec)))
			print("Training loss: {:.3f}".format(acc))
			print("")

		acc = self.test()
		print("Test loss: {:.3f}".format(acc))

	def save(self, fileName):
		self.model.save(fileName)

dataSet = DataSet('nob-eng/nob.txt', 64, 0.2)
rnn = RNN(dataSet)
rnn.trainForTime(1, 00, 00)
rnn.save('rnn.h5')
print(rnn("who are you?", 100))
