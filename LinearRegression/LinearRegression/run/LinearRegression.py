import sys
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.graphics.regressionplots import abline_plot
import matplotlib.pyplot as plt
#import MatrixAlgebra as ma
import math

class LinearRegression:
    def __init__(self, inputFileName, outputDirectory, numInstances, numAttributes):
        self.initParameters(inputFileName, outputDirectory, numInstances, numAttributes)
        self.initMatrice()
        self.result = None
        self.coefficient = None

    # Initialize parameters for Linear Regression
    def initParameters(self, inputFileName, outputDirectory, numInstances, numAttributes):
        self.fileName = inputFileName
        self.outputDirectory = outputDirectory
        self.numInstances = numInstances
        self.numAttributes = numAttributes
        self.isExtrOutlierEliminated = False
        self.cookDistanceMax = 1.0
        self.RSS = 0.0
        self.RSE = 0.0

    # Initialize matrice for prediction in Linear Regression
    def initMatrice(self):
        self.responseVector = []  # Response value vector
        self.predictorMatrix = [[] for i in range(self.numAttributes - 1)]  # Predictor's values Matrix
        self.residualList = []
        self.leverage = None
        self.fittedValue = None

    def readFile(self):
        # Read the input file
        inputFile = open(self.fileName, "rt")

        for line in inputFile:
            lineValArr = line.split(',')

            if '?' in lineValArr:
                self.numInstances -= 1
                continue

            # Inputing values into corresponding matrix
            i = 0
            self.responseVector.append(float(lineValArr[i]))

            while (i < (self.numAttributes - 1)):
                self.predictorMatrix[i].append(float(lineValArr[i + 1]))
                i += 1
        inputFile.close()

    # Calculate RSE after regression is fitted
    def calculateRSE(self):
        tempRSS = 0
        for index in range(self.numInstances):
            tempRSS = (self.responseVector[index] - self.fittedValue[index])**2
        self.RSS = tempRSS
        self.RSE = math.sqrt(tempRSS / (self.numInstances - 1 - (self.numAttributes - 1)))
        #print(str(self.RSE)) # Debug use

    def simpleRegression(self, attributeIndex):
        if attributeIndex > self.numAttributes:
            print("simpleRegression: attributeIndex > self.numAttributes")
        ones = np.ones(len(self.predictorMatrix[attributeIndex]))
        X = sm.add_constant(np.column_stack((self.predictorMatrix[0], ones)))
        X = sm.add_constant(np.column_stack((self.predictorMatrix[attributeIndex], X)))
        result = sm.OLS(self.responseVector, X).fit()

        return result

    # Ordinary Least Square Approach for estimating Linear Regression
    def regression(self, isPrintResult):
        # Add the vector of 1s into predictor matrix
        ones = np.ones(len(self.predictorMatrix[0]))
        X = sm.add_constant(np.column_stack((self.predictorMatrix[0], ones)))
        for predVal in self.predictorMatrix[1:]:
            X = sm.add_constant(np.column_stack((predVal, X)))

        # Estimate the Linear Regression and follow statistics
        results = sm.OLS(self.responseVector, X).fit()
        self.coefficient = results.params
        self.leverage = results.get_influence().hat_matrix_diag
        self.fittedValue = results.predict(X)
        self.calculateRSE()

        if isPrintResult:
            print(results.summary())
            #print("P-values: " + str(results.pvalues))
            print("\n")
        return results

    def valueSelection(self):
        """
        NOTES:
        The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. 
        The adjusted R-squared increases only if the new term improves the model more than would be expected by chance. 
        It decreases when a predictor improves the model by less than expected by chance.
        #"""
        print("After variable Selection Model:")
        poppedPredictorMatrix = []
        # Forward selection
        attributeIndex = 0
        current_score, best_score = 0.0, 0.0
        while attributeIndex < self.numAttributes:
            ones = np.ones(len(self.predictorMatrix[0]))
            X = sm.add_constant(np.column_stack((self.predictorMatrix[0], ones)))
            for predVal in self.predictorMatrix[1:(1 + attributeIndex)]:
                X = sm.add_constant(np.column_stack((predVal, X)))

            result = sm.OLS(self.responseVector, X).fit()
            current_score = result.rsquared_adj
            if current_score < best_score:
                poppedPredictorMatrix.append(self.predictorMatrix.pop(attributeIndex))
                print("taken out Attribute number " + str(attributeIndex))
                self.numAttributes -= 1
            else:
                best_score = max(current_score, best_score)
                attributeIndex += 1
        # After value selection, refit the regression
        self.result = self.regression(True)

    # Initial function to start examinating influence point
    def startExamineInfluePoint(self):
        print("After examining Influence point:")
        while (not self.isExtrOutlierEliminated):
            self.isExtrOutlierEliminated = self.examineInfluencePoint()
            if not self.isExtrOutlierEliminated:
                self.result = self.regression(False)
        print(self.result.summary())
        print("\n")

    # Examine Influence point from the result
    def examineInfluencePoint(self):
        (cookDistance, p) = self.result.get_influence().cooks_distance
        #max = 0.0
        for cookDIndex in range(len(cookDistance)):
            cookD = float(cookDistance[cookDIndex])
            #max = cookD if(cookD > max) else max
            # If the cook distance exceed 1.0, eliminate the data point and return False to refit the data
            if (cookD >= float(self.cookDistanceMax)):
                self.responseVector.pop(cookDIndex)
                for attributeIndex in range(self.numAttributes - 1):
                    self.predictorMatrix[attributeIndex].pop(cookDIndex)
                self.numInstances -= 1
                return False
        #print(max)
        return True

    def examineResidual(self):
        # I will use frequency table to plot residuals
        dataIndex = 0
        while dataIndex < self.numInstances:
            epsilon = self.responseVector[dataIndex] - self.fittedValue[dataIndex]
            self.residualList.append(epsilon)
            dataIndex += 1

    def plot(self, predictorIndex):
        fig, ax = plt.subplots()
        fig = sm.graphics.plot_fit(self.simpleRegression(predictorIndex), 0, ax=ax)
        ax.set_ylabel("MPG")
        ax.set_xlabel("Predictor: " + str(predictorIndex))
        ax.set_title("Simple Linear Regression Between MPG and Predictor " + str(predictorIndex + 1))

    def influencePlot(self):
        sm.graphics.influence_plot(self.result)
        #print(self.result.get_influence().summary_table())

    def plotHistogram(self):
        self.examineResidual()
        fig, ax = plt.subplots()
        plt.hist(self.residualList, bins=25)
        plt.show()

    def plotSimpleRegression(self):
        for i in range(0, self.numAttributes - 1):
            self.plot(i)

    def plotMultipleRegression(self):
        fig, ax = plt.subplots()
        fig = sm.graphics.plot_fit(self.result, 0, ax=ax)
        ax.set_ylabel("MPG")
        ax.set_xlabel("Predictors")
        ax.set_title("Multiple Linear Regression Between MDP and Predictors")
        plt.show()

    def process(self):
        self.readFile()

        print("First Full Variables Linear Model:")
        self.result = self.regression(True)

        self.startExamineInfluePoint()

        self.valueSelection()

        # Plot graphs
        self.influencePlot()
        self.plotHistogram()
        self.plotSimpleRegression()
        self.plotMultipleRegression()
