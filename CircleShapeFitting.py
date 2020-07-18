import random
import numpy as np
import matplotlib.pyplot as plt
import CoresetsAlgorithms
import ShapesAlgorithms
import Plots
import scipy
import MapleUtils
from PIL import Image
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# def runCircleFitting(samplesFileName, circlesFileName):
#     srcFile = open(srcFileName, "r")
#     dataFile = srcFile.readlines(0)
#     srcFile.close()
#
#     mapleSrcFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleSrc.txt'
#     mapleSrcFile = open(mapleSrcFileName, "r")
#     MapleDataFile = mapleSrcFile.readlines(0)
#     mapleSrcFile.close()
#
#     minimumNumberOfSamples = 15
#     validSampleThreshold = 200
#
#     centers = []
#     centersnumber = 0
#     norm = []
#
#     todoitr = 28
#     for itrIndex in range(todoitr):
#         data = dataFile[6215+itrIndex].split(',')#6215
#         dataSize = np.size(data)
#         P = []
#         domain = []
#         samples = []
#         fullSamples = []
#         offset = 0
#         for i in range(dataSize):
#             f = float(data[i])
#             fullSamples.append(f)
#             if f > validSampleThreshold:
#                 samples.append((i,f))
#                 domain.append(f)
#                 if offset == 0:
#                     offset = i
#
#         numberOfSamples = np.size(samples)/2
#         if numberOfSamples > minimumNumberOfSamples:
#             tnorm = (0.55 * numberOfSamples) / (np.max(domain)-np.min(domain))
#             b = np.min(domain)
#             for fixIndex in range (numberOfSamples):
#                 #norm.append(tnorm*0.5*(1+(float(numberOfSamples)-float(fixIndex))/float(numberOfSamples)))
#                 norm.append(tnorm)
#             for fixIndex in range (numberOfSamples):
#                 P.append((fixIndex, b + (samples[fixIndex][1]-b) * norm[fixIndex]))
#
#             circleX, circleY, circlesr, circlesR = ShapesAlgorithms.l_inf_annulus(P)
#             circleR = 0.5*(circlesr+circlesR)
#
#             # Plot samples
#             fig = plt.figure(itrIndex)
#             plt.plot(fullSamples, color='blue')
#
#             # Plot L-infinity circle from voronoi
#             vX = []
#             vY = []
#             for i in range(360):
#                 vX.append((circleR * np.sin(i)) + circleX + offset)
#                 vY.append((((circleR * np.cos(i)) + circleY) - np.min(samples))/norm[fixIndex] + np.min(samples))
#             plt.scatter(vX[:],vY[:], color='orange',s=0.2)
#
#             #Plot L-1 circle from Maple
#             if withMaple == 1:
#                 mapleData = MapleDataFile[itrIndex].split(',')
#                 circleX = float(mapleData[0])
#                 circleY = float(mapleData[1])
#                 circleR = float(mapleData[2])
#
#                 mX = []
#                 mY = []
#                 for i in range(360):
#                     mX.append((circleR * np.sin(i)) + circleX + offset)
#                     mY.append((((circleR * np.cos(i)) + circleY) - np.min(samples))/norm[fixIndex] + np.min(samples))
#                 plt.scatter(mX[:],mY[:], color='black',s=0.2)
#
#             centers.append((circleX + offset, (circleY - np.min(samples))/norm[fixIndex] + np.min(samples), circleR))
#
#             centersnumber += 1
#
#             fig.savefig('sample%s.png' % itrIndex)
#
#     fig = plt.figure(itrIndex+1)
#     runningX = []
#     runningy = []
#     for i in range(centersnumber):
#         runningX.append(centers[i][0])
#         runningy.append(i)
#     plt.scatter(runningy[:], runningX[:],color='green',s=5)
#     fig.savefig('X%s.png' % (itrIndex+1))
#
#     runningdX = []
#     runningdy = []
#     fig = plt.figure(itrIndex+2)
#     for i in range(centersnumber-1):
#         runningdX.append(centers[i+1][0]-centers[i][0])
#         runningdy.append(i)
#     plt.scatter(runningdy[:], runningdX[:],color='green',s=5)
#     fig.savefig('dX%s.png' % (itrIndex+2))
#
#     runningr = []
#     fig = plt.figure(itrIndex+3)
#     for i in range(centersnumber):
#         runningr.append(centers[i][2])
#     plt.scatter(runningy[:], runningr[:],color='green',s=5)
#     fig.savefig('r%s.png' % (itrIndex+3))
#
#     plt.show()
#
#     return

def calculateFittingCost(samplesFileName, dim, circlesFileName, numberOfRecords):
    samplesFile = open(samplesFileName,"r")
    dataFile = samplesFile.readlines(0)
    samplesFile.close()

    circlesFile = open(circlesFileName,"r")
    circlesDataFile = circlesFile.readlines(0)
    circlesFile.close()

    cost = []

    for itrIndex in range(numberOfRecords):
        data = dataFile[itrIndex].split(',')
        dataSize = int(np.size(data)/dim)
        samples = []
        for i in range(int(dataSize)):
            f = []
            for d in range(dim):
                f.append(float(data[dim*i+d]))
            samples.append(f)

        circleData = circlesDataFile[itrIndex].split(',')
        c = []
        for d in range(dim+1):
            c.append(float(circleData[d]))

        itrCost = 0
        for i in range(int(dataSize)):
            pCost = 0
            for d in range(dim):
                pCost += (c[d]-samples[i][d])**2
            itrCost += abs(np.sqrt(pCost)-c[dim])

        cost.append(itrCost)

    return cost

def fittingCost(P, circles, numberOfRecords):

    cost = []
    d = len(P[0][0])
    for i in range(numberOfRecords):

        totalCost = 0
        for p in P[i]:
            pCost = 0
            for dd in range(d):
                pCost += (circles[i][dd]-p[dd])**2
            totalCost += abs(np.sqrt(pCost)-circles[i][d])

        cost.append(totalCost)

    return cost

def runTechnionSamples(srcFileName, withMaple):

    srcFile = open(srcFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    mapleSrcFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleSrc.txt'
    mapleSrcFile = open(mapleSrcFileName, "r")
    MapleDataFile = mapleSrcFile.readlines(0)
    mapleSrcFile.close()

    minimumNumberOfSamples = 15
    validSampleThreshold = 200

    centers = []
    centersnumber = 0
    norm = []

    todoitr = 28
    for itrIndex in range(todoitr):
        data = dataFile[6215+itrIndex].split(',')#6215
        dataSize = np.size(data)
        P = []
        domain = []
        samples = []
        fullSamples = []
        offset = 0
        for i in range(dataSize):
            f = float(data[i])
            fullSamples.append(f)
            if f > validSampleThreshold:
                samples.append((i,f))
                domain.append(f)
                if offset == 0:
                    offset = i

        numberOfSamples = np.size(samples)/2
        if numberOfSamples > minimumNumberOfSamples:
            tnorm = (0.55 * numberOfSamples) / (np.max(domain)-np.min(domain))
            b = np.min(domain)
            for fixIndex in range (numberOfSamples):
                #norm.append(tnorm*0.5*(1+(float(numberOfSamples)-float(fixIndex))/float(numberOfSamples)))
                norm.append(tnorm)
            for fixIndex in range (numberOfSamples):
                P.append((fixIndex, b + (samples[fixIndex][1]-b) * norm[fixIndex]))

            circleX, circleY, circlesr, circlesR = ShapesAlgorithms.l_inf_annulus(P)
            circleR = 0.5*(circlesr+circlesR)

            # Plot samples
            fig = plt.figure(itrIndex)
            plt.plot(fullSamples, color='blue')

            # Plot L-infinity circle from voronoi
            vX = []
            vY = []
            for i in range(360):
                vX.append((circleR * np.sin(i)) + circleX + offset)
                vY.append((((circleR * np.cos(i)) + circleY) - np.min(samples))/norm[fixIndex] + np.min(samples))
            plt.scatter(vX[:],vY[:], color='orange',s=0.2)

            #Plot L-1 circle from Maple
            if withMaple == 1:
                mapleData = MapleDataFile[itrIndex].split(',')
                circleX = float(mapleData[0])
                circleY = float(mapleData[1])
                circleR = float(mapleData[2])

                mX = []
                mY = []
                for i in range(360):
                    mX.append((circleR * np.sin(i)) + circleX + offset)
                    mY.append((((circleR * np.cos(i)) + circleY) - np.min(samples))/norm[fixIndex] + np.min(samples))
                plt.scatter(mX[:],mY[:], color='black',s=0.2)

            centers.append((circleX + offset, (circleY - np.min(samples))/norm[fixIndex] + np.min(samples), circleR))

            centersnumber += 1

            fig.savefig('sample%s.png' % itrIndex)

    fig = plt.figure(itrIndex+1)
    runningX = []
    runningy = []
    for i in range(centersnumber):
        runningX.append(centers[i][0])
        runningy.append(i)
    plt.scatter(runningy[:], runningX[:],color='green',s=5)
    fig.savefig('X%s.png' % (itrIndex+1))

    runningdX = []
    runningdy = []
    fig = plt.figure(itrIndex+2)
    for i in range(centersnumber-1):
        runningdX.append(centers[i+1][0]-centers[i][0])
        runningdy.append(i)
    plt.scatter(runningdy[:], runningdX[:],color='green',s=5)
    fig.savefig('dX%s.png' % (itrIndex+2))

    runningr = []
    fig = plt.figure(itrIndex+3)
    for i in range(centersnumber):
        runningr.append(centers[i][2])
    plt.scatter(runningy[:], runningr[:],color='green',s=5)
    fig.savefig('r%s.png' % (itrIndex+3))

    plt.show()

    return
#########################       circle fitting                           ###############################

def generate2DNoisyCircle(n,c,e):
    P = []
    for i in range(n):
        r = 40*np.random.random_sample()
        rx = e*c[2]*(0.5-np.random.random_sample())
        ry = e*c[2]*(0.5-np.random.random_sample())
        x = c[2] * np.sin(r) + c[0] + rx
        y = c[2] * np.cos(r) + c[1] + ry
        P.append((x,y))

    printNoisyCircle = 0
    if printNoisyCircle == 1:
        for i in range (n):
            plt.scatter(P[i][0], P[i][1], color='orange')

    return P


def voronoi(points,shape=(500,500)):
    depthmap = np.ones(shape,np.float)*1e308
    colormap = np.zeros(shape,np.int)

    def hypot(X,Y):
        return (X-x)**2 + (Y-y)**2

    for i,(x,y) in enumerate(points):
        paraboloid = np.fromfunction(hypot,shape)
        colormap = np.where(paraboloid < depthmap,i+1,colormap)
        depthmap = np.where(paraboloid <
depthmap,paraboloid,depthmap)

    for (x,y) in points:
        colormap[x-1:x+2,y-1:y+2] = 0

    return colormap


print ('end')
