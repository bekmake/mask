import tensorflow as tf
import numpy as np
import pickle


class ModelMemoryManager:

    def __init__(self):
        self.weights = {}
        self.biases = {}
        self. saver = tf.train.Saver()

    def createWeights(self, name, dimensions):
        newWeights = tf.Variable(tf.random_normal(dimensions))
        self.weights[name] = newWeights

    def createBiases(self, name, dimensions):
        newBias = tf.Variable(tf.random_normal(dimensions))
        self.biases[name] = newBias

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

    def getSpecificWeight(self, name):
        return self.weights[name]

    def getSpecificBias(self, name):
        return self.biases[name]

    def saveWeights(self, fileName):
        with open(fileName + '.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle)

    def saveBiases(self, fileName):
        with open(fileName + '.pickle', 'wb') as handle:
            pickle.dump(self.biases, handle)
