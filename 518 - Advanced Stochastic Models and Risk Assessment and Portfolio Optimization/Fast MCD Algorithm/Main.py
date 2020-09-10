import numpy as np
import scipy.stats as sps
import math

# Calculate MCD for univariate data
# Parameters:
#   mnData: numpy 2D matrix of floats, rows are observations, column (1) is the variable (component)
#   iDataLength: integer number of observations
#   iPointsInPartition: integer size of subset of mnData under consideration
# Returns: list of: mean float, variance float
def mcdUnivariate(mnData, iDataLength, iPointsInPartition):
    mnSortData = np.sort(mnData.flatten())
    mnSubsets = np.array(list(map(lambda i: mnSortData[i:(i+iPointsInPartition)],
                                  [*range(iDataLength-iPointsInPartition+1)])))
    vnMeans = np.mean(mnSubsets, axis=1)
    vnVariances = np.var(mnSubsets, axis=1)
    iLowIndex = np.argsort(vnVariances)[0]
    return[vnMeans[iLowIndex], vnVariances[iLowIndex]]

# C Step. Find 'iNumOfComponents' number of rows of mnData closest to current mean and current cov
# Parameters:
#   mnData: numpy 2D matrix of floats, rows are observations, columns are variables (components)
#   iNumOfComponents: integer, size of subset of rows to find
#   vnCurrentMean: numpy 1D vector of floats
#   vnCurrentCov: numpy 2D vector of floats
# Returns: list of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
def oneCStep(mnData, iNumOfComponents, vnCurrentMean, mnCurrentCov):
    mnDistFromMean = np.subtract(mnData, vnCurrentMean)
    vnLastDistance = np.sum(mnDistFromMean@np.linalg.inv(mnCurrentCov)*mnDistFromMean, axis=1)
    viDistanceOrderings = np.argsort(vnLastDistance) #equivalent to Mathematica Ordering[] function
    vnIndex = viDistanceOrderings[0:iNumOfComponents]
    vnMean = np.mean(mnData[vnIndex], axis=0)
    mnCov = np.cov(np.transpose(mnData[vnIndex]))
    return[vnMean, mnCov, vnIndex]

# Repeat C Steps until convergence. Convergence occurs when the determinant of the current covariance estimate equals 0
# or equals the determinant of the previous covariance estimate
# Parameters:
#   lCurrentEstimate: list of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
#   mnData: numpy 2D matrix of floats, rows are observations, columns are variables (components)
#   iPointsInPartition: integer size of subset of mnData being considered
# Returns: list of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
def cStepConvergence(lCurrentEstimate, mnData, iPointsInPartition):
    while True:
        lLastEstimate = lCurrentEstimate
        lCurrentEstimate = oneCStep(mnData, iPointsInPartition, lLastEstimate[0], lLastEstimate[1])
        nCurrentDet = np.linalg.det(lCurrentEstimate[1])
        if nCurrentDet == 0 or nCurrentDet == np.linalg.det(lLastEstimate[1]):
            break
    return lCurrentEstimate

# Calculate candidate estimates
# Parameters:
#   mnData: numpy 2D matrix of floats, rows are observations, columns are variables (components)
#   iNumCandidates: number of candidate estimates to calculate
#   iDataLength: integer number of observations in mnData
#   iInitSubsetSize: initial number of points in partition
# Returns: list of lists of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
def calcCandidateEstimates(mnData,iNumCandidates,iDataLength,iInitSubsetSize):
    lCandidateEstimates = []
    for i in range(iNumCandidates):
        vnIntSubset = np.random.randint(low=iDataLength, size=iDataLength)
        iSubsetSize = iInitSubsetSize

        while True:
            vnElements = vnIntSubset[0:iSubsetSize]
            mnCurrentSubsets = mnData[vnElements]
            mnCurrentCov = np.cov(np.transpose(mnCurrentSubsets))
            iSubsetSize += 1

            if np.linalg.det(mnCurrentCov) > 0:
                break

        lMCI = oneCStep(mnData, iInitSubsetSize, np.mean(mnCurrentSubsets, axis=0), mnCurrentCov)
        lMCI = oneCStep(mnData, iInitSubsetSize, lMCI[0], lMCI[1])
        lCandidateEstimates.append(lMCI)
    return lCandidateEstimates

# Find the 'iLowElements' number of candidates with the smallest determinants
# Parameters:
#   lCandidateEstimates: list of lists of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
#   iLowElements: the number of candidates with the lowest determinants to return
# Returns: list of lists of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
def findLowestDets(lCandidateEstimates, iLowElements):
    vnDets = np.apply_along_axis(lambda x: np.linalg.det(x[1]), axis=1, arr=lCandidateEstimates)
    return lCandidateEstimates[np.argsort(vnDets)[0:iLowElements]]

# Calculate MCD for multivariate data with less than or equal to 600 observations
# Parameters:
#   mnData: numpy 2D matrix of floats, rows are observations, columns are the variables (components)
#   iDataLength: integer number of observations
#   iPointsInPartition: integer size of subset of mnData being considered
# Returns: list of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
def mcdLess600(mnData, iDataLength, iPointsInPartition):
    lCandidateEstimates = np.array(calcCandidateEstimates(mnData, 500, iDataLength, iPointsInPartition))
    lCandidateEstimates = findLowestDets(lCandidateEstimates, 10)
    lCandidateEstimates = np.apply_along_axis(lambda x: cStepConvergence(x, mnData, iPointsInPartition), axis=1,
                                              arr=lCandidateEstimates)
    return findLowestDets(lCandidateEstimates, 1)[0]

# Calculate MCD for multivariate data with greater than 600 observations
# Parameters:
#   mnData: numpy 2D matrix of floats, rows are observations, columns are the variables (components)
#   iDataLength: integer number of observations
#   iPointsInPartition: integer size of subset of mnData being considered
# Returns: list of: numpy 1D mean vector, numpy 2D covariance matrix, numpy 1D index vector
def mcdGreater600(mnData, iDataLength, iPointsInPartition):
    iApproxSubsetLength = math.ceil(iDataLength / 5)
    vnIndexes = np.random.choice(iDataLength, iDataLength, replace=False)
    vnDisjointColumnIndex = []
    lmnDisjointSets = []

    for i in range(4):
        lmnDisjointSets.append(mnData[vnIndexes[i*iApproxSubsetLength:(i+1)*iApproxSubsetLength]])
        vnDisjointColumnIndex.append(iApproxSubsetLength)
    lmnDisjointSets.append(mnData[vnIndexes[4*iApproxSubsetLength:]])
    vnDisjointColumnIndex.append(iDataLength - 4 * iApproxSubsetLength)

    lEstimatesInsideSubsets = []

    for i in range(5):
        lCandidateEstimates = np.array(calcCandidateEstimates(lmnDisjointSets[i], 100, vnDisjointColumnIndex[i],
                                       math.floor(vnDisjointColumnIndex[i] * iPointsInPartition / iDataLength)))
        lCandidateEstimates = findLowestDets(lCandidateEstimates, 10)
        lEstimatesInsideSubsets.extend(lCandidateEstimates)

    lCandidateEstimates = []

    for i in range(50):
        lMCI = oneCStep(mnData, iPointsInPartition, lEstimatesInsideSubsets[i][0], lEstimatesInsideSubsets[i][1])
        lMCI = oneCStep(mnData, iPointsInPartition, lMCI[0], lMCI[1])
        lCandidateEstimates.append(lMCI)

    lCandidateEstimates = findLowestDets(np.array(lCandidateEstimates), 10)
    lCandidateEstimates = np.apply_along_axis(lambda x: cStepConvergence(x, mnData, iPointsInPartition), axis=1,
                                              arr=lCandidateEstimates)
    return findLowestDets(lCandidateEstimates, 1)[0]

# Consistency - to - normal - model factor; Pison, Van Aelst, Willems, 2002
# Parameters:
#   degrees of freedom: integer
#   quantile : float
# Returns: float
def consistToNormal(iDoF, nQuantile):
    nQ = sps.chi2.ppf(nQuantile, iDoF)
    return nQuantile/sps.chi2.cdf(nQ, iDoF+2)

#####################################################Program Start######################################################
#####################################################Program Start######################################################
#####################################################Program Start######################################################
#####################################################Program Start######################################################
#####################################################Program Start######################################################
sDataFile = input("Data file (): ")
iPointsInPartition = int(input("Points ellipse covers (0 for default): "))

mnData = np.loadtxt(sDataFile, delimiter=',', ndmin=2)
iDataLength, iNumOfComponents = mnData.shape

#set nPointsInPartition
if iPointsInPartition == 0:
    iPointsInPartition = math.floor((iDataLength+iNumOfComponents+1)/2)

#program path depending on data and inputs
if not(math.floor((iDataLength+iNumOfComponents+1)/2) <= iPointsInPartition <= iDataLength):
    print("The value: '", iPointsInPartition, "' for h is invalid")
elif iPointsInPartition == iDataLength:
    print("Location: ", np.mean(np.transpose(mnData)))
    print("Spread: ", np.cov(np.transpose(mnData)))
else:
    print("RUNNING MCD... Number of Components = ", iNumOfComponents, ", Data Length = ", iDataLength,
          ", and Points Ellipse Covers = ", iPointsInPartition)
    if iNumOfComponents == 1:
        vResult = mcdUnivariate(mnData, iDataLength, iPointsInPartition)
        print("Location: ", vResult[0])
        print("Spread: ", vResult[1])
    else:
        if iDataLength <= 600:
            vResult = mcdLess600(mnData, iDataLength, iPointsInPartition)
        else:
            vResult = mcdGreater600(mnData, iDataLength, iPointsInPartition)
        nScaling = consistToNormal(iNumOfComponents, iPointsInPartition/iDataLength)
        print("New Mean: ", vResult[0])
        print("Scaled Cov: ", np.multiply(nScaling, vResult[1]))
        print("Full Cov: ", vResult[1])
        print("Subset: ", vResult[2])