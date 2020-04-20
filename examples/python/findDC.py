from __future__ import print_function, division
import numpy as np

class CFindDC():

    def __init__(self):
        #The following parameters describe the probability to see each of the following tempratures:
        self.valsVec = np.array([35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40 , 40.5, 41  , 41.5]) #temprature
        self.probVec = np.array([1 , 5   , 15, 30  , 25, 15  , 2 , 1.5 , 1 , 0.9 , 0.8, 0.5 , 0.25, 0.1 ]) #probability to see each temprature
        self.probVec = self.probVec / self.probVec.sum() * 0.9  # 0.9 to since we estimate that there are about 10% outliers (temprature was not found correctly)
        self.logProbOutOfRange = np.log(0.1 / 10)  # divided by 10 for range +- 5 deg outside.
        self.logProbVec = np.log(self.probVec)

    def __LogLikePerTemprature(self,measuredTempVec):
        logVals = measuredTempVec*0
        inRange = np.logical_and((measuredTempVec>=self.valsVec[0]) , (measuredTempVec<=self.valsVec[-1]))
        logVals[~inRange] = self.logProbOutOfRange
        logVals[inRange] = np.interp(measuredTempVec[inRange], self.valsVec, self.logProbVec)

        return logVals.mean(), logVals


    def findDC(self,measuredTempVec):
        #gets a vector of tempratures and finds the dc that maximizes the probability of measuredTempVec
        #it returns two estimations of the dc. maxlikelihood and an approximation of the mean likelihood
        dcRange = np.linspace(-5,5,1001)
        dcGrade = dcRange*0
        for k in range(len(dcRange)):
            dc = dcRange[k]
            dcGrade[k],_ = self.__LogLikePerTemprature(measuredTempVec - dc)

        ind = np.argmax(dcGrade)
        maxLikeDc = dcRange[ind]

        maxRange = min( ind, len(dcRange)-1-ind)
        usedRange = min(maxRange,100) #+- one degree
        likeVec = np.exp(dcGrade[ind-usedRange:ind+usedRange])
        meanLikeDc = np.dot(dcRange[ind-usedRange:ind+usedRange], likeVec)/likeVec.sum()

        return maxLikeDc,meanLikeDc



if __name__ == "__main__":
    dcFinder = CFindDC()
    dc = 3 #must be no more than +-5
    tempVector = np.array([36.6,36,37, 36.5, 35.8,37.2, 37.5, 37.2, 40, 42,37,50]) + dc
    dcMaxLike,dcMeanLike = dcFinder.findDC(tempVector)
    simpleMedian = np.median(tempVector) - 37
    print(dcMaxLike,dcMeanLike,simpleMedian)

    dc = -4.5  # must be no more than +-5
    tempVector = np.array([36.6, 36, 37, 36.5, 35.8, 37.2, 37.5, 37.2, 40, 42, 37, 50]) + dc
    dcMaxLike, dcMeanLike = dcFinder.findDC(tempVector)
    simpleMedian = np.median(tempVector) - 37
    print(dcMaxLike, dcMeanLike,simpleMedian)

    dc = -4.5  # must be no more than +-5
    tempVector = np.array([36.6, 36, 37, 36.5, 35.8, 37.2, 40, 50]) + dc
    dcMaxLike, dcMeanLike = dcFinder.findDC(tempVector)
    simpleMedian = np.median(tempVector) - 37
    print(dcMaxLike, dcMeanLike, simpleMedian)

    dc = 3  # must be no more than +-5
    tempVector = np.array([36.6, 37.2, 40, 42, 37, 50]) + dc
    dcMaxLike, dcMeanLike = dcFinder.findDC(tempVector)
    simpleMedian = np.median(tempVector) - 37
    print(dcMaxLike, dcMeanLike, simpleMedian)