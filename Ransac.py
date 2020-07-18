import random
import numpy
import ShapesAlgorithms
# import Plots
import scipy
# import math
# import cv2
# from scipy.misc import imsave

def findShape(P, S, modelFunction, modelDistFunction, minRequiredPointsForModel, maxNumberOfIterations, thresholdToDetermineInlier, minRequiredPointsForFitting):

    iterations = 0
    bestModel = []
    bestInliers = []
    bestError = 0
    firstModel = 1

    while iterations < maxNumberOfIterations:
        maybeInliers = random.sample(S, k=minRequiredPointsForModel)
        if minRequiredPointsForModel == 3:
            maybeModel = modelFunction(maybeInliers[0],maybeInliers[1],maybeInliers[2])
        if minRequiredPointsForModel == 4:
            maybeModel = modelFunction(maybeInliers[0],maybeInliers[1],maybeInliers[2],maybeInliers[3])
        alsoInliers = []

        # for p in P:
        #     if modelDistFunction(maybeModel, p) < thresholdToDetermineInlier:
        #         alsoInliers.append(p)
        #
        # if len(alsoInliers) > minRequiredPointsForFitting:
        #     betterModel = maybeModel #modelFunction(alsoInliers)
        #     minRequiredPointsForFitting = max(0.8*len(alsoInliers), minRequiredPointsForFitting)
        #
        #     fitError = 0
        #     for p in alsoInliers:
        #         fitError += modelDistFunction(betterModel, p)
        #     fitError = fitError/(len(alsoInliers))
        #
        #     if fitError < bestError:
        #         bestModel = betterModel
        #         bestError = fitError
        fitError = 0
        for p in P:
            fitError += modelDistFunction(maybeModel, p)
        #fitError = fitError/(len(alsoInliers))

        if firstModel == 1 or fitError < bestError:
            bestModel = maybeModel
            bestInliers = maybeInliers
            bestError = fitError
            firstModel = 0

        iterations += 1

    return bestModel, bestInliers, bestError
