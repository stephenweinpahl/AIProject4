# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import time
import samples
import dataClassifier
import random
import numpy

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    
    
    
    acc = []
    perf = []
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
    

    # look the testing data in 10% increments
    for a in range(1, 11):
      start = time.time()
      dataLimit =  int(len(trainingData)*a/10)
      # trainingPrior is the prior probability given the label (counts the number of times a label is seen overall)
       # trainingCondProb is the conditional probability that given index (feat, label) occurs (has total value of features)
       # trainingCount is the total count for seeing given index (feat, label)
      trainingPrior = util.Counter()
      trainingCondProb = util.Counter()
      trainingCount = util.Counter()
      # collect training data and initial counts
      for i in range(dataLimit):
        datum = trainingData[i]
        label = trainingLabels[i]
        trainingPrior[label] += 1
        for feat, val in datum.items():
          trainingCount[(feat, label)] += 1
          if val > 0:
            trainingCondProb[(feat, label)] += val
      perf.append(time.time()-start)
      # observe perfomrance on actual test data

      k = .2
      dataPrior = util.Counter()
      dataCondProb = util.Counter()
      dataCount = util.Counter()

      # need to recover test data points into the counts

      for key, val in trainingPrior.items():
        dataPrior[key] += val
      for key, val in trainingCondProb.items():
        dataCondProb[key] += val
      for key, val in trainingCount.items():
        dataCount[key] += val
      
      # apply smoothing, see the berkley website for the smoothing formula, it is applied over both values of 0 - 1 hence 2k:

      for feat in self.features:
        for label in self.legalLabels:
          dataCondProb[(feat, label)] += k
          dataCount[(feat, label)] += 2*k

      # need to get the get the acutal probabilities for prior and condProb by normalizing (at this point we have counts)
      total = 0
      for key, val in dataPrior.items():
        total += val

      for key, val in dataPrior.items():
        dataPrior[key] =  dataPrior[key]/total

      for key, val in dataCondProb.items():
        dataCondProb[key] =  dataCondProb[key]/dataCount[key]

      self.dataPrior = dataPrior
      self.dataCondProb = dataCondProb

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

      
      
      # guess = self.classify(validationData)

      # accCount = 0
      # for i in range(len(guess)):
      #   if guess[i] == validationLabels[i]:
      #     accCount += 1
      # acc.append(100*accCount/len(guess))
      
    print()
    print("Mean Accuracy for Naive Bayes on test data")
    print(accMean)
    print("Std Accuracy for Naive Bayes on test data")
    print(accStd)
    print("Time for Naive Bayes")
    print(perf)

    print()
    

      
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    
    # self.priorLabel is the priorProbobability of seeing the label it is a counter[label]
    # datum is a counter with index of feature and a value where a feature is cordinates of a pixel?
    logJoint = util.Counter()

    for label in self.legalLabels:
      if self.dataPrior[label] > 0:
        logJoint[label] = math.log(self.dataPrior[label])
      else:
        logJoint[label] = 0
     
      for feat, val in datum.items():
        # also because we are taking logs, we add the cond probs instead of multiplying
        # first apply the formula to features which are present (value > 0)
        if val > 0 and self.dataCondProb[(feat, label)] > 0:
          logJoint[label] += math.log(self.dataCondProb[(feat, label)])
        # this means that the value of the feature is 0 or no features were detected
        # so take the compliment as the probobality that this feature could be in
        else:
          logJoint[label] += math.log(1-self.dataCondProb[(feat, label)])
        
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
