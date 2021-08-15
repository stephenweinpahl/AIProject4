# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
import time
import samples
import dataClassifier
import random
import numpy
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    c = .002
    accMean = []
    accStd = []
    numbers = range(100)
    # Determine if the data is testing on digits or faces
    isDigit = False
    featureFunction = dataClassifier.enhancedFeatureExtractorFace
    if self.legalLabels[-1] == 9:
      isDigit = True
      featureFunction = dataClassifier.enhancedFeatureExtractorDigit

    # classifiy the testdata
    if isDigit == False:
      rawTestData = samples.loadDataFile("facedata/facedatatest", 100,60,70)
      testLabels = samples.loadLabelsFile("facedata/facedatatestlabels",100)
    else:
      rawTestData = samples.loadDataFile("digitdata/testimages", 100,28,28)
      testLabels = samples.loadLabelsFile("digitdata/testlabels", 100)
    testData = map(featureFunction, rawTestData)
    
    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    # we have to train the algo
    # self.weights[prediction] represents the vector of weight that we created
    # self.weights[label] represents the vectors of weight that most accuractly depict the answer
    perf = []
    acc = []
    for a in range(1, 11):
      start = time.time()
      dataLimit =  int(len(trainingData)*a/10)
      # collect training data and initial counts
      for iteration in range(self.max_iterations):  
        for i in range(int(len(trainingData)*(a-1)/10), dataLimit):
          datum = trainingData[i]
          label = trainingLabels[i]
          # make a prediction using our model and compare it to the training data label
          prediction = self.classify([datum])[0]
          # check to see if prediciton was correct, if not must correct the weight vectors
          # the util class has a very useful counter additon and subtraction that makes this easy
          # similar to perceptron, we have to update weight vecotrs however a tau is required which is shown in the documentation
          
          tau = min(c, ((self.weights[prediction] - self.weights[label])*datum + 1.0)/(2.0*(datum*datum)))
          if prediction != label:
            # must update the values of tau*datum for each feature in datum to get the correction factor for the weight vecotors
            for feat in datum:
              datum[feat] *= tau
            self.weights[label] += datum
            self.weights[prediction] -= datum
      perf.append(time.time()-start)

      # now that model has been trained, we can test it on the validation data
      # accCount = 0
      # for i in range(len(validationData)):
      #   datum = validationData[i]
      #   label = validationLabels[i]
      #   prediction = self.classify([datum])[0]
        
      #   # track number of correct predictions
      #   if prediction == label:
      #     accCount += 1
      # acc.append(100*accCount/len(validationData))

      # random sample the model 5 times with randomly selected 50 elements from test data
      currAcc = []
      for i in range(5):
        guessIndex = random.sample(numbers, k = 50)
        accCount = 0
        currTestData = []
        currTestLabels = []
        for j in range(len(guessIndex)):
          index = guessIndex[j]
          currTestData.append(testData[index])
          currTestLabels.append(testLabels[index])
        accCount = 0
        guesses = self.classify(currTestData)
        accCount = 0
        for k in range(len(guessIndex)):
          if guesses[k] == currTestLabels[k]:
            accCount += 1
        #print(100.0*accCount/len(guesses))
        currAcc.append(100.0*accCount/len(guesses))
      
      accMean.append(numpy.mean(currAcc))
      accStd.append(numpy.std(currAcc))

    
    print()
    print("Mean Accuracy for Mira on test data")
    print(accMean)
    print("Std Accuracy for Mira on test data")
    print(accStd)
    print("Time for Mira")
    print(perf)

    print()

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

