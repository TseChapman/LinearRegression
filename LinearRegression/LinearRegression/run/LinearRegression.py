import sys
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
        self.predictorNames = [
            "Cylinders: 3", "Cylinders: 4", "Cylinders: 5", "Cylinders: 6", "Cylinders: 8", "Displacement",
            "Horsepower", "Weight", "Acceleration", "Model Year: 70", "Model Year: 71", "Model Year: 72",
            "Model Year: 73", "Model Year: 74", "Model Year: 75", "Model Year: 76", "Model Year: 77", "Model Year: 78",
            "Model Year: 79", "Model Year: 80", "Model Year: 81", "Model Year: 82", "Origin: 1", "Origin: 2",
            "Origin: 3"
        ]
        self.numDiscrete = 21  # manually defined after running
        self.numDiscreteAttribute = 3
        self.numAttributes = numAttributes - self.numDiscreteAttribute + self.numDiscrete
        self.isExtrOutlierEliminated = False
        self.cookDistanceMax = 1.0
        self.RSS = 0.0
        self.RSE = 0.0

    # Initialize matrice for prediction in Linear Regression
    def initMatrice(self):
        self.responseVector = []  # Response value vector
        self.predictorMatrix = [[] for i in range(self.numAttributes - 1)]  # Predictor's values Matrix
        self.vif = []
        self.residualList = []
        self.leverage = None
        self.fittedValue = None

    def addCylinders(self, value):
        cylinders = [3, 4, 5, 6, 8]
        listIndex = [0, 1, 2, 3, 4]

        for cylinderInd in range(len(cylinders)):
            if value == cylinders[cylinderInd]:
                self.predictorMatrix[listIndex[cylinderInd]].append(value)
                listIndex.pop(cylinderInd)

        for i in listIndex:
            self.predictorMatrix[i].append(0)

    def addModelYear(self, value):
        year = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
        listIndex = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        for yearInd in range(len(year)):
            if value == year[yearInd]:
                self.predictorMatrix[listIndex[yearInd]].append(value)
                listIndex.pop(yearInd)

        for i in listIndex:
            self.predictorMatrix[i].append(0)

    def addOrigin(self, value):
        origin = [1, 2, 3]
        listIndex = [22, 23, 24]

        for originInd in range(len(origin)):
            if value == origin[originInd]:
                self.predictorMatrix[listIndex[originInd]].append(value)
                listIndex.pop(originInd)

        for i in listIndex:
            self.predictorMatrix[i].append(0)

    def readFile(self):
        # Read the input file
        inputFile = open(self.fileName, "rt")

        for line in inputFile:
            lineValArr = line.split(',')

            if '?' in lineValArr:
                self.numInstances -= 1
                continue

            # Inputing values into corresponding matrix
            readValIndex = 0
            i = 0
            self.responseVector.append(float(lineValArr[readValIndex]))

            while (i < (self.numAttributes - 1) and readValIndex < len(lineValArr) - 2):  # ignore the name and /n
                if (i == 0):  # Cylinders
                    self.addCylinders(float(lineValArr[readValIndex + 1]))
                    i += 5
                elif (i == 9):  # model year
                    self.addModelYear(float(lineValArr[readValIndex + 1]))
                    i += 13
                elif (i == 22):  # Origin
                    self.addOrigin(float(lineValArr[readValIndex + 1]))
                else:
                    self.predictorMatrix[i].append(float(lineValArr[readValIndex + 1]))
                    i += 1

                readValIndex += 1
        inputFile.close()

    # Calculate RSE after regression is fitted
    def calculateRSE(self):
        tempRSS = 0
        for index in range(self.numInstances):
            tempRSS = (self.responseVector[index] - self.fittedValue[index])**2
        self.RSS = tempRSS
        self.RSE = math.sqrt(tempRSS / (self.numInstances - 1 - (self.numAttributes - 1)))
        #print(str(self.RSE)) # Debug use

    def predMatrixConstruct(self):
        if (len(self.predictorMatrix) == 0):
            print("Empty Predictor Matrix")
            return

        ones = np.ones(len(self.predictorMatrix[0]))
        X = sm.add_constant(np.column_stack((self.predictorMatrix[0], ones)))
        for predVal in self.predictorMatrix[1:]:
            X = sm.add_constant(np.column_stack((predVal, X)))
        return X

    def simpleRegression(self, attributeIndex):
        if (len(self.predictorMatrix) == 0):
            print("Empty Predictor Matrix")
            return

        if attributeIndex > self.numAttributes:
            print("simpleRegression: attributeIndex > self.numAttributes")
        result = sm.OLS(self.responseVector, self.predictorMatrix[attributeIndex]).fit()

        return result

    def printPValueSummary(self, result, pValue):
        if (result == None):
            return

        print("\n")
        print("P-Value Summary Table")
        print("==================================================")
        #print(str(pValue))
        pValue = str(pValue).strip("[]")
        pValList = pValue.split()
        for pValIndex in range(0, len(pValList) - 1):
            print("Predictor ", self.predictorNames[pValIndex], ": ", pValList[pValIndex])
        print("Const: ", pValList[-1])
        print("\n")

    # Ordinary Least Square Approach for estimating Linear Regression
    def regression(self, isPrintResult):
        # Add the vector of 1s into predictor matrix
        X = self.predMatrixConstruct()

        # Estimate the Linear Regression and follow statistics
        results = sm.OLS(self.responseVector, X).fit()
        self.coefficient = results.params
        self.leverage = results.get_influence().hat_matrix_diag
        self.fittedValue = results.predict(X)
        self.calculateRSE()

        if isPrintResult is True:
            print(results.summary())
            print("R Squared: ", results.rsquared)
            self.printPValueSummary(results, results.pvalues)

        return results

    def valueSelection(self):
        """
        NOTES:
        The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. 
        The adjusted R-squared increases only if the new term improves the model more than would be expected by chance. 
        It decreases when a predictor improves the model by less than expected by chance.
        #"""
        print("After variable Selection Model:")
        # Forward selection
        attributeIndex = 0
        current_score, best_score = 0.0, 0.0

        # Check the adjusted R squared for each attribute inserted into the regression
        while attributeIndex < self.numAttributes:
            if (len(self.predictorMatrix) == 0):
                print("Empty Predictor Matrix")
                break

            ones = np.ones(len(self.predictorMatrix[0]))
            X = sm.add_constant(np.column_stack((self.predictorMatrix[0], ones)))
            for predVal in self.predictorMatrix[1:(1 + attributeIndex)]:
                X = sm.add_constant(np.column_stack((predVal, X)))

            result = sm.OLS(self.responseVector, X).fit()
            current_score = result.rsquared_adj
            if (attributeIndex < len(self.predictorNames)):
                print("Attribute: ", self.predictorNames[attributeIndex], " adj R^2: ", current_score)
            if current_score < best_score:
                self.predictorMatrix.pop(attributeIndex)
                print("taken out Attribute number ", self.predictorNames[attributeIndex])
                self.predictorNames.pop(attributeIndex)
                self.numAttributes -= 1
            else:
                best_score = max(current_score, best_score)
                attributeIndex += 1

        print("\n")
        # After value selection, refit the regression
        self.result = self.regression(True)

    # Examine Influence point from the result
    def examineInfluencePoint(self):
        (cookDistance, p) = self.result.get_influence().cooks_distance
        self.cookDistanceMax = 4 / self.numInstances
        max = 0.0
        for cookDIndex in range(len(cookDistance)):
            cookD = float(cookDistance[cookDIndex])
            max = cookD if (cookD > max) else max

            # Crude rule of thumb:
            # If the cook distance exceed 1.0, investigate/eliminate the data point and return False to refit the data
            if (cookD > float(self.cookDistanceMax)):
                self.responseVector.pop(cookDIndex)
                for attributeIndex in range(self.numAttributes - 1):
                    self.predictorMatrix[attributeIndex].pop(cookDIndex)
                self.numInstances -= 1
                return False
        print("Max Cook's Distance: ", max)
        return True

    # Initial function to start examinating influence point
    def startExamineInfluePoint(self):
        print("After examining Influence point:")
        while (not self.isExtrOutlierEliminated):
            self.isExtrOutlierEliminated = self.examineInfluencePoint()
            if not self.isExtrOutlierEliminated:
                self.result = self.regression(False)
        print(self.result.summary())
        self.printPValueSummary(self.result, self.result.pvalues)
        print("\n")

    def printVIFSummary(self):
        if (len(self.vif) == 0):
            print("self.vif is empty")
            return
        print("VIF Summary Table")
        print("==================================================")
        print("Const: ", self.vif[0])
        for i in range(1, len(self.vif)):
            #print(f"Predictor {self.predictorNames[i-1]}: {self.vif[i]}")
            print(self.vif[i])
        print("\n")

    def examineVIF(self):
        # Don't take const's vif into account
        for vifIndex in range(1, len(self.vif)):
            # Examine VIF(i) if it is greater than 10
            if self.vif[vifIndex] > 10:
                # taken the predictor out of predictor matrix
                self.predictorMatrix.pop(vifIndex - 1)
                self.numAttributes -= 1
                print("predictor poped: ", vifIndex)
                return False
        return True

    def startExamineMulticollinearity(self):
        """
        isExamineVIFDone = False
        while(not isExamineVIFDone):
            # regression with X_i as response value
            X = self.predMatrixConstruct()
            self.vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            self.printVIFSummary()
            isExamineVIFDone = self.examineVIF()
        """
        X = self.predMatrixConstruct()
        self.vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        self.printVIFSummary()
        """
        self.predictorMatrix.pop(self.vif.index(max(self.vif)) - 1)
        self.numAttributes -= 1
        index = self.vif.index(max(self.vif))
        X = self.predMatrixConstruct()
        self.vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        self.printVIFSummary()
        """
        # print("After Examining Collinearity: ")
        # self.result = self.regression(True)

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
        ax.set_xlabel("Predictor: " + str(self.predictorNames[predictorIndex]))
        ax.set_title("Simple Linear Regression Between MPG and Predictor " + str(self.predictorNames[predictorIndex]))

    def plotSimpleRegression(self):
        for i in range(0, self.numAttributes - 1):
            self.plot(i)

    def influencePlot(self):
        sm.graphics.influence_plot(self.result)
        #print(self.result.get_influence().summary_table())

    def plotHistogram(self):
        self.examineResidual()
        fig, ax = plt.subplots()
        plt.hist(self.residualList, bins=19)
        plt.show()

    def plotMultipleRegression(self):
        fig, ax = plt.subplots()
        fig = sm.graphics.plot_fit(self.result, 0, ax=ax)
        ax.set_ylabel("MPG")
        ax.set_xlabel("Predictors")
        ax.set_title("Multiple Linear Regression Between MDP and Predictors")
        # plt.show()

    def process(self):
        self.readFile()

        print("First Full Variables Linear Model:")
        self.result = self.regression(True)

        self.startExamineInfluePoint()
        self.valueSelection()

        #self.startExamineMulticollinearity()

        print("Final Model:")
        self.result = self.regression(True)

        # Plot graphs
        self.influencePlot()
        self.plotSimpleRegression()
        self.plotHistogram()
        self.plotMultipleRegression()
