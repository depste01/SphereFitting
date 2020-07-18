import time
from sys import exit
import matplotlib.pyplot as plt
import numpy as np
#from sympy import S
import CircleShapeFitting
import CoresetsAlgorithms
import RunTests
import ShapesAlgorithms
import TechnionUtil
import circleLeastsquare
import cv2
import Ransac

#c,r = ShapesAlgorithms.getSphereByThreePoints([2,5],[7,4],[8,7])


# Running configuration
runStatistics = 0
runMinimalEnclosingCircle = 0
runKLinesMean = 0
runMedianLines = 0
runCircleComparingFramework = 0
runTechnionFramework = 0
runMedianFramework = 0
plotSyntheticSamples = 0
useBuildSamplesAsL1 = 1
runFrameworkOverImages = 0

# Coreset
runL1Circle = 1
overrideL1Circle = 1
plotSamplesWithL1Circle = 0
runLInfCircle = 1
runLInfCircleOverCoreset = 0
plotCirclesFromSetVxLInfCoreset = 0
plotSetVsL1Coreset = 0
plotSetVsL1RandomSamples = 0
onlyBuildCoreset = 0
runL1CircleOverCoreset = 1
plotL1CircleOverCoreset = 1
runL1CircleOverRandomSampling = 1
plotL1CircleOverRandomSampling = 0
runL1CircleOverRansac = 1
runL1CircleOverRansacUsingCoresetSampling = 1
plotL1CircleOverRansac = 0
runL1CircleAsLeastsqr = 1
plotL1CircleAsLeastsqr = 0
runL1CircleUsingOpenCV = 1
plotL1CircleUsingOpenCV = 0
plotCirclesFromSetVxL1Coreset = 0
plotCirclesFromSetVxL1RandomSamples = 0
useTechnionSamples = 0

# Technion parameters
plotTechnionSamplesWithCircles = 1
technioValidSampleThreshold = 200
technionMinNumberOfSamples = 20
technionMaxNumberOfSamples = 200
maxNumberOfTechnionRecords = 20

# Runnings parameters
roundSamples = 0

# Coreset parameters
estType = 0
usePermutation = 1
streamingCoreset = 0
AppendStatistics = 0
numberOfRansacIterations = 5
#pPer = [(20,1024),(20,2048),(20,4096),(10,8192),(10,16384),(10,32768),(5,65536),(5,131072),(5,262144),(3,524288),(3,1048576)]
#pPer = [(2,12)]
pPer = [(10,512),(10,2048),(10,8192),(10,32768),(5,131072),(5,262144),(5,524288),(3,1048576)]
#pPer = [(1,1024),(1,131072),(1,1048576),(1,1073741824)]
pPerMod = 1
epsPer = [0.1,0.01,0.001,0.0001]
#epsPer = [0.01,0.002]
epsPerMod = len(pPer)
errorRatePer = [40]
errorRatePerMod = len(pPer) * len(epsPer)
dimPer = [2]
dimPerMod = len(pPer) * len(epsPer) * len(errorRatePer)
#dataType = ["records","error rate","dimantion","epsilon","Ransac iterations","P size","Coreset size", "coreset error","random error","L infty error", "ransac error", "ransac coreset error", "least square error","OpenCV error", "Optimal time", "Coreset time", "Random time", "L infty time", "Ransac time", "least square time", "OpenCV time"]
#dataType = ["dimantion","epsilon","Ransac iterations","P size","Coreset size", "Coreset cost", "Ransac cost", "Ransac coreset cost", "Optimal time", "subset time", "Coreset time", "Random time", "L infty time", "Ransac time", "Ransac over coreset", "least square time", "OpenCV time"]
dataType = ["dimantion","epsilon","Ransac iterations","P size","Coreset size", "Coreset cost", "Ransac cost", "Ransac coreset cost", "Coreset time", "Ransac time", "Ransac over coreset", "least square time", "OpenCV time"]
timeStampPer = ['startTime','buildTime','runL1CircleTime','runL1CircleOverCoresetTime','buildCoresetTime','runL1CircleOverRandomSamplingTime','runLInfCircleTime','runL1CircleOverRansacTime', 'runL1CircleOverRansacCoresetTime', 'runL1CircleAsleastsquareTime','runL1CircleUsingOpencvTime','endTime']
timeStemp = np.zeros(len(timeStampPer))

# Files name
technionFilesDirectoryName = r'c://Users/RBD-W540/Documents/PhD/TechnionProject/'
permutationsCirclesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/PermutationsCircles.txt'
mapleCommandFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleCommands.mpl'
mapleResultsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt'
mapleCirclesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/mapleCircles.txt'
permutationsCirclesOverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/PermutationsCirclesOverCoreset.txt'
permutationsCirclesOverRandomSamplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/PermutationsCirclesOverRandomSamples.txt'
mapleCommandOverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleCommandsOverCoreset.mpl'
mapleResultsOverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleResultsOverCoreset.txt'
mapleCirclesOverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/mapleCirclesOverCoreset.txt'
CInfFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/CInfCircles.txt'
CInfOverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/CInfOverCoresetCircles.txt'
coresetSamplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/CoresetSamples.csv'
L1coresetSamplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/L1CoresetSamples.csv'
L1RandomSamplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/L1RandomSamples.csv'
C1OverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/C1OverCoresetCircles.txt'
CEncloseOverCoresetFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/CEncloseOverCoresetCircles.txt'
C1OverRandomSamplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/C1OverRandomSamplesCircles.txt'
statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/dsphere.csv'
technionPath = r'C://Users/RBD-W540/Documents/PhD/TechnionProject/AvivRecords/'

# # plot running time vs. epsilon
# statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/timeforpapervsepsilon.csv'
# dataType = ["dimantion","epsilon","Ransac iterations","P size","Coreset size", "Coreset cost", "Ransac cost", "Ransac coreset cost", "Coreset time", "Ransac time", "Ransac over coreset", "least square time", "OpenCV time"]
# pPer = [(1,1048576)]
# epsPer = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]
# RunTests.plotStatisticsTimevsepsilon(statisticsFileName, dataType, pPer, epsPer)
# exit(0)
# plot running time vs. size
statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/timeforpaper2.csv'
dataType = ["dimantion","epsilon","Ransac iterations","P size","Coreset size", "Coreset cost", "Ransac cost", "Ransac coreset cost", "Coreset time", "Ransac time", "Ransac over coreset", "least square time", "OpenCV time"]
pPer = [(1,32),(1,64),(1,128),(1,256),(1,512),(1,1024),(1,2048),(1,4096),(1,8192),(1,16384),(1,32768),(1,65536),(1,131072),(1,262144),(1,524288),(1,1048576)]
epsPer = [0.001]
RunTests.plotStatisticsTime(statisticsFileName, dataType, pPer, epsPer)

# statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/statisticsWithDualRansac4DForPaper.csv'
# dataType = ["records","error rate","dimantion","epsilon","Ransac iterations","P size","Coreset size", "coreset error","random error","L infty error", "ransac error", "improved ransac error", "least square error","OpenCV error", "Optimal time", "Coreset time", "Random time", "L infty time", "Ransac time", "least square time", "OpenCV time"]
# pPer = [(1,1024),(1,131072),(1,1048576),(1,1073741824)]
# epsPer = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]
# RunTests.plotStatistics(statisticsFileName, dataType, pPer, epsPer, 0)
#
# statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/statisticsWithDualRansac3DForPaper.csv'
# dataType = ["records","error rate","dimantion","epsilon","Ransac iterations","P size","Coreset size", "coreset error","random error","L infty error", "ransac error", "ransac coreset error", "least square error","OpenCV error", "Optimal time", "Coreset time", "Random time", "L infty time", "Ransac time", "least square time", "OpenCV time"]
# pPer = [(1,1024),(1,131072),(1,1048576),(1,1073741824)]
# epsPer = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]
# #RunTests.plotStatistics(statisticsFileName, dataType, pPer, epsPer, 0)
#
# statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/statisticsWithDualRansacForPaper.csv'
# dataType = ["records","error rate","dimantion","epsilon","Ransac iterations","P size","Coreset size", "coreset error","random error","L infty error", "ransac error", "ransac coreset error", "least square error","OpenCV error", "Optimal time", "Coreset time", "Random time", "L infty time", "Ransac time", "least square time", "OpenCV time"]
# pPer = [(1,1024),(1,131072),(1,1048576),(1,1073741824)]
# epsPer = [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]
# RunTests.plotStatistics(statisticsFileName, dataType, pPer, epsPer, 0)
#
exit(0)
#

# Run only statistics
if runStatistics == 1:
    #statisticsFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/Coresetsizevsrequirederrorestimation.csv'

    RunTests.plotStatistics(statisticsFileName, dataType, pPer, epsPer, 0)
    plt.show()
    exit(0)

# Run minimal enclosing circle
if runMinimalEnclosingCircle == 1:

    samplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/SyntheticSamples.csv'
    plotSyntheticSamplesWithCircle = 1
    appendResultsFromMaple = 1
    buildCircleType = 0
    dim = 2

    # Build circle samples
    if buildCircleType == 1:
        # Build circle samples from image
        imagePer = ['paintcorcle.png']
        numberOfRecords = len(imagePer)
        RunTests.defineSamplesFile(samplesFileName)
        for i in range(len(imagePer)):
            RunTests.prepareImage(imagePer[i],'points'+imagePer[i],1)
            RunTests.buildSamplesFileFromImage('points'+imagePer[i],samplesFileName, 1)
    else:
        # Build syntetic noisy circle samples
        numberOfRecords = 2
        numberOfSyntheticSamples = 150
        errorRate = 60
        plotSyntheticSamples = 0
        syntheticCirclesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/SyntheticCircles.txt'
        RunTests.buildNoiseCircleSamples(samplesFileName, numberOfSyntheticSamples, dim, syntheticCirclesFileName, errorRate, numberOfRecords, plotSyntheticSamples)

    # Find minimal enclosing circle using Maple
    successResults = RunTests.runMapleMinimalEnclosingCircle(samplesFileName, mapleCommandOverCoresetFileName, mapleResultsOverCoresetFileName, CEncloseOverCoresetFileName, numberOfRecords, appendResultsFromMaple, plotSyntheticSamplesWithCircle, dim)

    P = RunTests.BuildPFromSamples(samplesFileName, dim, numberOfRecords)

    exit(0)

# Run k-lines mean using coreset
if runKLinesMean == 1:

    samplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/SyntheticSamples.csv'
    syntheticLinesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/SyntheticLines.txt'
    numberOfLines = 1 # number of lines for the syntetic data
    d = 2 # dimantion
    numberOfRecords = 2 # number of records
    eps = 0.1 # error estimation
    k = numberOfLines # number of required lines
    m = 2 # metric space
    errorRate = 20 # noise parameter
    buildCircleType = 1 # image type

    # p = [1,5]
    # l = [[4,6],[7,9]]
    # D = ShapesAlgorithms.squarDistance(p,l)

    # Build circle samples
    if buildCircleType == 1:
        # Build circle samples from image
        imagePer = ['line.png','2lines.png','3lines.png']
        numberOfRecords = len(imagePer)
        RunTests.defineSamplesFile(samplesFileName)
        for i in range(len(imagePer)):
            RunTests.prepareImage(imagePer[i],'points'+imagePer[i],1)
            RunTests.buildSamplesFileFromImage('points'+imagePer[i],samplesFileName, 1)
    else:
        RunTests.buildNoiseLinesSamples(samplesFileName, numberOfSyntheticSamples, d, errorRate, numberOfLines, syntheticLinesFileName, errorRate, numberOfRecords, plotSyntheticSamples)

    P = RunTests.BuildPFromSamples(samplesFileName, d, numberOfRecords)
    S = CoresetsAlgorithms.buildCoresetForLines(P, numberOfRecords, eps, k, m)
    L = ShapesAlgorithms.findKLines(P, k, d, numberOfRecords)
    L = ShapesAlgorithms.meanCircleFitting(P, k)

    exit(0)

### Run circle comparing framework
if runCircleComparingFramework == 1:

    loopIndex = 0
    imageNumber = 1
    images = ['p1.png','p3.png','p4.png','p5.png','p7.png','p8.png','p13.png','p17.png','p18.png','p19.png']
    eps = 0.01
    streamingCoreset = 0
    invertImage = 0
    numberOfRansacIterations = 5
    cdvlsq = []
    cdvocv = []
    cdvrnsc = []
    cdvrnscx = []
    cdvrnd = []
    circles = [(0,0,0)]
    ccc = 0

    for image in images:
        imageNumber = 10 * loopIndex

        P = []
        P.append(RunTests.preparePFromImage(image, invertImage))
        n = len(P[0])
        ccc += 1

        coresetC, S, coresetSize = RunTests.findL1CircleUsingCoreset(P, eps, streamingCoreset, circles)
        RunTests.plot2DSets(P, 1, 'blue', 8, imageNumber)
        RunTests.plot2DCircles(coresetC, 1, 'green', 3, imageNumber)
        RunTests.plot2DSets(S, 1, 'red', 8, imageNumber)
        imageNumber += 1

        linfinityC = (0,0,0)#ShapesAlgorithms.LInfCircle(P)

        leastsquareC = circleLeastsquare.findLeastSquareCircle(P)

        opencvC = ShapesAlgorithms.opencvCircle(image)

        modelPoints = []
        ransacCircles = []
        ransacC, ransacPoints, e = RunTests.runRansacCircleDetection(P, P, numberOfRansacIterations)
        modelPoints.append(ransacPoints)
        RunTests.plot2DSets(P, 1, 'blue', 8, imageNumber)
        ransacCircles.append(ransacC)
        RunTests.plot2DCircles(ransacCircles, 1, 'green', 3, imageNumber)
        RunTests.plot2DSets(modelPoints, 1, 'red', 8, imageNumber)
        imageNumber += 1

        modelPointsx = []
        ransacCirclesx = []
        ransacCx, ransacPointsx, ex = RunTests.runRansacCircleDetection(P, P, numberOfRansacIterations*numberOfRansacIterations*numberOfRansacIterations)
        modelPointsx.append(ransacPointsx)
        RunTests.plot2DSets(P, 1, 'blue', 8, imageNumber)
        ransacCirclesx.append(ransacCx)
        RunTests.plot2DCircles(ransacCirclesx, 1, 'green', 3, imageNumber)
        RunTests.plot2DSets(modelPointsx, 1, 'red', 8, imageNumber)
        imageNumber += 1

        modelPoints = []
        randomC, randomPoints = RunTests.findL1CircleUsingRandomSampling(P, coresetSize)
        modelPoints.append(randomPoints)
        RunTests.plot2DSets(P, 1, 'blue', 8, imageNumber)
        RunTests.plot2DCircles(randomC, 1, 'green', 3, imageNumber)
        RunTests.plot2DSets(randomPoints, 1, 'red', 8, imageNumber)
        loopIndex += 1

        #successResults = RunTests.runMapleMinimalEnclosingCircle(samplesFileName, mapleCommandOverCoresetFileName, mapleResultsOverCoresetFileName, CEncloseOverCoresetFileName, numberOfRecords, appendResultsFromMaple, plotSyntheticSamplesWithCircle, dim)

        ecoreset = 0
        #einfty = 0
        elsq = 0
        eocv = 0
        ernsc = 0
        ernscx = 0
        ernd = 0
        for p in P[0]:
            ecoreset += ShapesAlgorithms.squareDistFromCircle(coresetC[0],p)
            #einfty += ShapesAlgorithms.squareDistFromCircle(linfinityC,p)
            elsq += ShapesAlgorithms.squareDistFromCircle(leastsquareC,p)
            eocv += ShapesAlgorithms.squareDistFromCircle(opencvC,p)
            ernsc += ShapesAlgorithms.squareDistFromCircle(ransacC,p)
            ernscx += ShapesAlgorithms.squareDistFromCircle(ransacCx,p)
            ernd += ShapesAlgorithms.squareDistFromCircle(randomC[0],p)

        cdvlsq.append(ecoreset / elsq)
        cdvocv.append(ecoreset / eocv)
        cdvrnsc.append(ecoreset / ernsc)
        cdvrnscx.append(ecoreset / ernscx)
        cdvrnd.append(ecoreset / ernd)

        #RunTests.printComparedCirclesOverImages(image, coresetC, linfinityC, leastsquareC, opencvC, ransacC, randomC)

    plt.show()

    exit(0)

# Run technion framework
if runTechnionFramework == 1:
    eps = 0.001
    exp = TechnionUtil.searchTechnion(technionPath)
    for e in exp:
        technionSamplesFileName = e[0]+'/'+e[2]
        technionSamplesToFrameFileName = e[0]+'/technionSamplesToFrame.txt'
        samplesFileName = e[0]+'/samples.txt'
        nor = e[0].split('Hight ')
        nnoorr = nor[1].split('\\')
        if nnoorr[0] == 'one':
            maxNumberOfTechnionRecords = 24
        if nnoorr[0] == 'two':
            maxNumberOfTechnionRecords = 20
        if nnoorr[0] == 'three':
            maxNumberOfTechnionRecords = 16
        numberOfRecords = RunTests.buildTechnionSamples(technionSamplesFileName, technionSamplesToFrameFileName, samplesFileName, technioValidSampleThreshold, technionMinNumberOfSamples, technionMaxNumberOfSamples, roundSamples, maxNumberOfTechnionRecords)
        P = RunTests.BuildPFromSamples(samplesFileName, 2, numberOfRecords)
        if len(P) > 0:
            if len(P[0]) > 20:
                S,fdn,whiledo,left,rays,numberOfIterations, totalItr = CoresetsAlgorithms.buildCoresetForL1Circle(P, numberOfRecords, eps, estType, circles)
                L1coresetSamplesFileName = e[0]+'/CoresetSamples.txt'
                RunTests.saveCoresetSamples(L1coresetSamplesFileName, S, numberOfRecords)
                RunTests.runL1Circle(L1coresetSamplesFileName, permutationsCirclesOverCoresetFileName, mapleCommandOverCoresetFileName, mapleResultsOverCoresetFileName, C1OverCoresetFileName, numberOfRecords, 1, 0, usePermutation,1,2)
                pressureTime, pressureSensor, sensor = TechnionUtil.searchPressure(e[0]+'/'+e[1])
                technionPositionResultFileName = e[0]+'/Analyze'
                if plotTechnionSamplesWithCircles == 1:
                    #RunTests.runTechnionSamplesWithCircles(technionSamplesFileName, technionPositionResultFileName, technionSamplesToFrameFileName, C1OverCoresetFileName, 15, technioValidSampleThreshold, numberOfRecords, 'red',eps,1)
                    TechnionUtil.buildResultsFile(technionSamplesFileName, technionPositionResultFileName, technionSamplesToFrameFileName, C1OverCoresetFileName, 15, 200, numberOfRecords, e[0], e[0]+'/'+e[1],pressureTime, pressureSensor, sensor)
    exit(0)

# General median coreset framework. No ready yet
if runMedianFramework == 1:

    images = ['p1.png','p2.png','p3.png','p4.png','p5.png']
    eps = 0.01

    for i in range(len(images)):
        starttime = time.time()
        P = []

        RunTests.prepareImage(images[i],'points'+images[i],0)
        prepareimagetime = time.time()

        P.append(RunTests.readImageToP('points'+images[i]))
        buildptime = time.time()

        S = CoresetsAlgorithms.buildMedianCoreset(P, eps)
        coresettime = time.time()

        RunTests.saveCoresetSamples('coreset' + (images[i].split('.png'))[0]+'.txt', S, 1)
        RunTests.runL1Circle('coreset' + (images[i].split('.png'))[0]+'.txt', permutationsCirclesOverCoresetFileName, mapleCommandOverCoresetFileName, mapleResultsOverCoresetFileName, 'L1' + (images[i].split('.png'))[0]+'.txt', 1, 1, 0, 1,1,2)
        #RunTests.printCoresetOverImage(imagePer[i], 'coreset' + (imagePer[i].split('.png'))[0]+'.txt', 'L1' + (imagePer[i].split('.png'))[0]+'.txt')
        #RunTests.printCoresetOverImagesec(imagePer[i], 'coreset' + (imagePer[i].split('.png'))[0]+'.txt', 'L1' + (imagePer[i].split('.png'))[0]+'.txt')
        x,y,r = circleLeastsquare.findLeastSquareCircle(P)
        inf = 0
        if len(P[0]) < 5000:
            inf = 1
        if inf == 1:
                ShapesAlgorithms.LInfCircle(P, 1, 'lint' + (images[i].split('.png'))[0]+'.txt')


        RunTests.printComparedCirclesOverImages(images[i], 'coreset' + (images[i].split('.png'))[0] + '.txt', 'L1' + (images[i].split('.png'))[0] + '.txt', 'lint' + (images[i].split('.png'))[0] + '.txt', 0, inf, x, y, r)
        #RunTests.prinfImageWithCircle(imagePer[i], CircleInitial+imagePer[i], i)

        #RunTests.opencvCircle(imagePer[i])
        #b= i

    plt.show()

    exit(0)

runningLoops = len(dimPer)*len(errorRatePer)*len(epsPer)*len(pPer)

#RunTests.plotStatisticsTime(statisticsFileName, dataType, pPer, epsPer)
#exit(1)

# Build statistics file
if AppendStatistics == 0:
    statisticsFile = open(statisticsFileName,'w')
    statisticsFile.write("%s\n"%','.join(dataType))
    statisticsFile.close()

# Run L1 circle detection framework
for rnnglp in range(runningLoops):

    imageNumber = 100*rnnglp
    optimalCost = []
    coresetCost = []
    randomSamplingCost = []
    linfinityCost = []
    ransacCost = []
    ransacCoresetCost = []
    lsqCost = []
    opencvCost = []

    if runFrameworkOverImages == 1:
        numberOfRecords = pPer[(rnnglp/pPerMod)%len(pPer)][0]
    else:
        # Set iteration parameters
        numberOfRecords = pPer[(rnnglp/pPerMod)%len(pPer)][0]
        numberOfSyntheticSamples = pPer[(rnnglp/pPerMod)%len(pPer)][1]
        eps  = epsPer[(rnnglp/epsPerMod)%len(epsPer)]
        errorRate = errorRatePer[(rnnglp/errorRatePerMod)%len(errorRatePer)]
        dim = dimPer[(rnnglp/dimPerMod)%len(dimPer)]

        # Files name
        prerecordedSamplesFileExt = '_'+str(errorRate)+'_'+str(numberOfRecords)+'_'+str(numberOfSyntheticSamples)#+'_'+str(dim)
        samplesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/SyntheticSamples'+prerecordedSamplesFileExt+'.csv'
        syntheticCirclesFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/SyntheticCircles'+prerecordedSamplesFileExt+'.txt'
        C1FileName = syntheticCirclesFileName
        if useBuildSamplesAsL1 == 0:
            C1FileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/C1Circles'+prerecordedSamplesFileExt+'.txt'

        # Build samples file
        tmp, circles = RunTests.buildNoiseCircleSamples(samplesFileName, numberOfSyntheticSamples, dim, syntheticCirclesFileName, errorRate, numberOfRecords, plotSyntheticSamples)

    # Build set P from samples file
    P = RunTests.BuildPFromSamples(samplesFileName, dim, numberOfRecords)

    timeStemp[timeStampPer.index('startTime')] = time.time()

    # Find L1 circle over samples
    if runL1Circle == 1:
        a = 9
        #RunTests.runL1Circle(samplesFileName, permutationsCirclesFileName, mapleCommandFileName, mapleResultsFileName, C1FileName, numberOfRecords, 1, plotSamplesWithL1Circle, usePermutation,overrideL1Circle,dim)
        #optimalCost.append(CircleShapeFitting.calculateFittingCost(samplesFileName, dim, C1FileName, numberOfRecords))
    timeStemp[timeStampPer.index('runL1CircleTime')] = time.time()

    # Find L1 circle over coreset
    if runL1CircleOverCoreset == 1:
        #RunTests.dsphereCoreset(P, eps)
        coresetC, S, coresetSize, buildTime = RunTests.findL1CircleUsingCoreset(P, eps, streamingCoreset, circles)
        timeStemp[timeStampPer.index('buildCoresetTime')] = buildTime
        if plotL1CircleOverCoreset == 1:
            RunTests.plot2DCircles(coresetC, len(P), 'green', 4, imageNumber)
            RunTests.plot2DSets(P, len(P), 'blue', 8, imageNumber)
            RunTests.plot2DSets(S, len(P), 'red', 8, imageNumber)
            imageNumber += 10
        coresetCost.append(CircleShapeFitting.fittingCost(P, coresetC, numberOfRecords))
    else:
        coresetCost.append(0)

        #S,fdn,whiledo,left,rays,numberOfIterations, totalItr = CoresetsAlgorithms.buildStreamingCoresetForL1Circle(P, numberOfRecords, eps, estType)
        #S,fdn,whiledo,left,rays,numberOfIterations, totalItr = CoresetsAlgorithms.buildCoresetForL1Circle(P, numberOfRecords, eps, estType)
        #RunTests.saveCoresetSamples(L1coresetSamplesFileName, S, numberOfRecords)
        #if onlyBuildCoreset == 0:
        #    RunTests.runL1Circle(L1coresetSamplesFileName, permutationsCirclesOverCoresetFileName, mapleCommandOverCoresetFileName, mapleResultsOverCoresetFileName, C1OverCoresetFileName, numberOfRecords, 1, 0, usePermutation,1,dim)
        #    coresetCost = CircleShapeFitting.calculateFittingCost(samplesFileName, dim, C1OverCoresetFileName, numberOfRecords)
    timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')] = time.time()

    # Find L1 circle over random samples
    if runL1CircleOverRandomSampling == 1:
        randomC, randomPoints = RunTests.findL1CircleUsingRandomSampling(P, coresetSize)
        if plotL1CircleOverRandomSampling == 1:
            RunTests.plot2DCircles(randomC, len(P), 'green', 4, imageNumber)
            RunTests.plot2DSets(P, len(P), 'blue', 8, imageNumber)
            modelPoints = [randomPoints]
            RunTests.plot2DSets(randomPoints, len(P), 'red', 8, imageNumber)
            imageNumber += 10
        randomSamplingCost.append(CircleShapeFitting.fittingCost(P, randomC, numberOfRecords))
    else:
        randomSamplingCost.append(0)

        # R = CoresetsAlgorithms.buildRandomSamplesForL1Circle(P, numberOfRecords, S)
        # RunTests.saveCoresetSamples(L1RandomSamplesFileName, R, numberOfRecords)
        # RunTests.runL1Circle(L1RandomSamplesFileName, permutationsCirclesOverRandomSamplesFileName, mapleCommandOverCoresetFileName, mapleResultsOverCoresetFileName, C1OverRandomSamplesFileName, numberOfRecords, 1, 0, usePermutation,1,dim)
    timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')] = time.time()

    # Find L infinity circle
    # if runLInfCircle == 1:
    #     for qqq in range(numberOfRecords):
    #         CInf = ShapesAlgorithms.LInfCircle(P)
        #linfCost = CircleShapeFitting.calculateFittingCost(samplesFileName, CInfFileName, numberOfRecords)
    timeStemp[timeStampPer.index('runLInfCircleTime')] = time.time()
    #
    # if runLInfCircleOverCoreset == 1:
    #     S = CoresetsAlgorithms.buildCoresetForLInfCircleUsingRays(P, CInf, eps, 9, 9, numberOfRecords)
    #     RunTests.saveCoresetSamples(coresetSamplesFileName, S, numberOfRecords)
    #     CInfOverCoreset = ShapesAlgorithms.LInfCircle(S, numberOfRecords, CInfOverCoresetFileName)
    # timeStemp[timeStampPer.index('runLInfCircleOverCoresetTime')] = time.time()
    #
    # # Plot Set P and coreset S circles over samples
    # if plotCirclesFromSetVxLInfCoreset == 1:
    #     RunTests.plotResults(samplesFileName, CInfFileName, CInfOverCoresetFileName, CInfOverCoresetFileName, numberOfRecords)
    #     RunTests.plotResults(coresetSamplesFileName, CInfFileName, CInfOverCoresetFileName, CInfOverCoresetFileName, numberOfRecords)

    # Find L1 circle over RANSAC algorithm
    if runL1CircleOverRansac == 1:
        numberOfRansacIterations = int(((np.log(len(P[0])))**2))
        ransacC, ransacPoints, e = RunTests.runRansacCircleDetection(P, P, numberOfRansacIterations)
        if plotL1CircleOverRansac == 1:
            ransacCircles = [ransacC]
            RunTests.plot2DCircles(ransacC, len(P), 'green', 4, imageNumber)
            RunTests.plot2DSets(P, len(P), 'blue', 8, imageNumber)
            modelPoints = [ransacPoints]
            RunTests.plot2DSets(ransacPoints, len(P), 'red', 8, imageNumber)
            imageNumber += 10
        ransacCost.append(CircleShapeFitting.fittingCost(P, ransacC, numberOfRecords))
    timeStemp[timeStampPer.index('runL1CircleOverRansacTime')] = time.time()

    # Find L1 circle over RANSAC algorithm using coreset sampling
    if runL1CircleOverRansacUsingCoresetSampling == 1:
        numberOfRansacIterations = int(((np.log(len(S[0])))**2))
        ransacC, ransacPoints, e = RunTests.runRansacCircleDetection(P, S, numberOfRansacIterations)
        if plotL1CircleOverRansac == 1:
            ransacCircles = [ransacC]
            RunTests.plot2DCircles(ransacC, len(P), 'green', 4, imageNumber)
            RunTests.plot2DSets(P, len(P), 'blue', 8, imageNumber)
            modelPoints = [ransacPoints]
            RunTests.plot2DSets(ransacPoints, len(P), 'red', 8, imageNumber)
            imageNumber += 10
        ransacCoresetCost.append(CircleShapeFitting.fittingCost(P, ransacC, numberOfRecords))
    timeStemp[timeStampPer.index('runL1CircleOverRansacCoresetTime')] = time.time()

    # Find L1 circle as least square method
    if runL1CircleAsLeastsqr == 1:
        leastsquareC = circleLeastsquare.findLeastSquareCircle(P)
        if plotL1CircleAsLeastsqr == 1:
            RunTests.plot2DCircles(leastsquareC, len(P), 'green', 4, imageNumber)
            RunTests.plot2DSets(P, len(P), 'blue', 8, imageNumber)
            imageNumber += 10
        #lsqCost.append(CircleShapeFitting.fittingCost(P, leastsquareC, numberOfRecords))
    #else:
        #lsqCost.append(0)
    timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')] = time.time()

    # Find L1 circle using OpenCV library (only for images)
    if runL1CircleUsingOpenCV == 1 and runFrameworkOverImages == 1:
        opencvC = ShapesAlgorithms.opencvCircle(image)
        if plotL1CircleUsingOpenCV == 1:
            RunTests.plot2DCircles(opencvC, 1, 'green', 4, imageNumber)
            RunTests.plot2DSets(P, 1, 'blue', 8, imageNumber)
            imageNumber += 10
        #opencvCost.append(CircleShapeFitting.fittingCost(P, opencvC, numberOfRecords))
    #else:
        #opencvCost.append(0)
    timeStemp[timeStampPer.index('runL1CircleUsingOpencvTime')] = time.time()

    # Plot Set P and coreset S circles over samples
    # if plotCirclesFromSetVxL1Coreset == 1:
    #     RunTests.plotSamples(samplesFileName,L1coresetSamplesFileName,numberOfRecords,'blue','red',8,3,rnnglp)
    #
    # if plotCirclesFromSetVxL1RandomSamples == 1:
    #     RunTests.plotCircles(C1OverRandomSamplesFileName, numberOfRecords, 'yellow',rnnglp)

    #
    # Statistics
    statisticsFile = open(statisticsFileName,'a')
    for r in range(numberOfRecords):
        relCoresetCost = (coresetCost[0][r])# - optimalCost[0][r])/float(optimalCost[0][r])
        relRandomCost = (randomSamplingCost[0][r])# - optimalCost[0][r])/float(optimalCost[0][r])
        #relLinfinityCost = (linfinityCost[0][r] - optimalCost[0][r])/float(optimalCost[0][r])
        relRansacCost = (ransacCost[0][r])# - optimalCost[0][r])/float(optimalCost[0][r])
        relRansacCoresetCost = (ransacCoresetCost[0][r])# - optimalCost[0][r])/float(optimalCost[0][r])
        #relLeastsqrCost = (lsqCost[0][r] - optimalCost[0][r])/float(optimalCost[0][r])
        #relOpencvCost = (opencvCost[0][r] - optimalCost[0][r])/float(optimalCost[0][r])

        # statisticsFile.write("%d,%d,%d,%f,%d,%d,%d"%(1,errorRate,dim,eps, numberOfRansacIterations, len(P[r]),len(S[r])))
        # statisticsFile.write(",%f,%f,%f,%f,%f,%f,%f"%(relCoresetCost,relRandomCost,relLinfinityCost,relRansacCost,relRansacCoresetCost,relLeastsqrCost,relOpencvCost))
        # statisticsFile.write(",%f,%f,%f,%f,%f,%f,%f\n"%(timeStemp[timeStampPer.index('runL1CircleTime')]-timeStemp[timeStampPer.index('startTime')],timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')]-timeStemp[timeStampPer.index('runL1CircleTime')],timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')]-timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')],timeStemp[timeStampPer.index('runLInfCircleTime')]-timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')],timeStemp[timeStampPer.index('runL1CircleOverRansacTime')]-timeStemp[timeStampPer.index('runLInfCircleTime')],timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')]-timeStemp[timeStampPer.index('runL1CircleOverRansacTime')],timeStemp[timeStampPer.index('runL1CircleUsingOpencvTime')]-timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')]))

        statisticsFile.write("%d,%f,%d,%d,%d"%(dim,eps, numberOfRansacIterations, len(P[r]),len(S[r])))
        statisticsFile.write(",%f,%f,%f"%(coresetCost[0][r],ransacCost[0][r],ransacCoresetCost[0][r]))
        statisticsFile.write(",%f,%f,%f,%f,%f,%f,%f,%f,%f\n"%(timeStemp[timeStampPer.index('runL1CircleTime')]-timeStemp[timeStampPer.index('startTime')],timeStemp[timeStampPer.index('buildCoresetTime')]-timeStemp[timeStampPer.index('runL1CircleTime')],timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')]-timeStemp[timeStampPer.index('runL1CircleTime')],timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')]-timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')],timeStemp[timeStampPer.index('runLInfCircleTime')]-timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')],timeStemp[timeStampPer.index('runL1CircleOverRansacTime')]-timeStemp[timeStampPer.index('runLInfCircleTime')],timeStemp[timeStampPer.index('runL1CircleOverRansacCoresetTime')]-timeStemp[timeStampPer.index('runL1CircleOverRansacTime')],timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')]-timeStemp[timeStampPer.index('runL1CircleOverRansacCoresetTime')],timeStemp[timeStampPer.index('runL1CircleUsingOpencvTime')]-timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')]))

    statisticsFile.close()

    # Plot technion results
    if useTechnionSamples == 1:
        if plotTechnionSamplesWithCircles == 1:
            RunTests.runTechnionSamplesWithCircles(technionSamplesFileName, technionPositionResultFileName, technionSamplesToFrameFileName, C1OverCoresetFileName, 15, technioValidSampleThreshold, numberOfRecords, 'red',eps,1)

# Plot statistics
RunTests.plotStatisticsTime(statisticsFileName, dataType, pPer, epsPer, runningLoops+1)

plt.show()

print ('end')

