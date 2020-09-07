import sys
import LinearRegression
from LinearRegression import LinearRegression

if __name__ == "__main__":
    inputFileName = "Input/auto_mpg.csv"
    outputDirectory = "Output/"
    numInstances = 398
    numAttributes = 8

    linearRegression = LinearRegression(inputFileName, outputDirectory, numInstances, numAttributes)
    linearRegression.process()
