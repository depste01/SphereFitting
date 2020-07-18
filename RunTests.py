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
import subprocess
import CircleShapeFitting
from mpl_toolkits.mplot3d import Axes3D
import glob, os
import os.path
import math
#import statistics
import cv2
from scipy.misc import imsave
from PIL import Image
import io
from matplotlib.ticker import FormatStrFormatter
from PIL import Image, ImageDraw
from matplotlib.patches import Circle
import Ransac
import time

def findMedianLines(P):
    L = []
    numberOfRecords = len(P)
    for i in range(numberOfRecords):
        L.append([])

    for i in range(numberOfRecords):
        Ptag = []
        for n in range(len(P[i])):
            for d in range(len(P[i][0])):
                Ptag.append(P[i][n][d])

        aaa = np.reshape(Ptag,(len(P[i]),len(P[i][0])))
        L[i].append(ShapesAlgorithms.getMedianLines(aaa,1))

    return L

def printComparedCirclesOverImages(imageFileName, coresetC, linfinityC, leastsquareC, opencvC, ransacC, randomC):

    # L1 with coreset
    im1 = cv2.imread('points'+imageFileName)
    img1 = np.invert(im1)
    cv2.circle(img1,(int(float(coresetC[0][1])),int(float(coresetC[0][0]))), int(float(coresetC[0][2])), (0,255,0), 2)
    imsave('L1WithCoreset'+imageFileName, img1)

    # l infinity
    im2 = cv2.imread('points'+imageFileName)
    img2 = np.invert(im2)
    cv2.circle(img2,(int(float(linfinityC[0][1])),int(float(linfinityC[0][0]))), int(float(linfinityC[0][2])), (255,0,255), 2)
    imsave('Linfinity'+imageFileName, img2)

    # python leastsqr
    im3 = cv2.imread('points'+imageFileName)
    img3 = np.invert(im3)
    cv2.circle(img3,(int(float(leastsquareC[1])),int(float(leastsquareC[0]))), int(float(leastsquareC[2])), (255,255,0), 2)
    imsave('leastSquare'+imageFileName, img3)

    # opencv
    im4 = cv2.imread('points'+imageFileName)
    img4 = np.invert(im4)
    cv2.circle(img4,(int(float(opencvC[0])),int(float(opencvC[1]))), int(float(opencvC[2])), (255,255,0), 2)
    imsave('opencv'+imageFileName, img4)

    # ransac
    im5 = cv2.imread('points'+imageFileName)
    img5 = np.invert(im5)
    cv2.circle(img5,(int(float(ransacC[1])),int(float(ransacC[0]))), int(float(ransacC[2])), (255,255,0), 2)
    imsave('ransac'+imageFileName, img5)

    # random sampling
    im6 = cv2.imread('points'+imageFileName)
    img6 = np.invert(im6)
    cv2.circle(img6,(int(float(randomC[0][1])),int(float(randomC[0][0]))), int(float(randomC[0][2])), (255,255,0), 2)
    imsave('random'+imageFileName, img6)

def printCoresetOverImagesec(imageFileName, samplesFileName, circleFileName):

    circlesFile = open(circleFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()
    C = circlesData[0].split(',')

    samplesFile = open(samplesFileName,"r")
    dataFile = samplesFile.readlines(0)
    samplesFile.close()
    data = dataFile[0].split(',')
    dataSize = int(np.size(data)/2)

    im = cv2.imread(imageFileName)
    cv2.circle(im,(int(float(C[1])),int(float(C[0]))), int(float(C[2])), (0,255,0), 2)
    imsave('circlesec'+imageFileName, im)

def printCoresetOverImage(imageFileName, samplesFileName, circleFileName):

    circlesFile = open(circleFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()
    C = circlesData[0].split(',')

    samplesFile = open(samplesFileName,"r")
    dataFile = samplesFile.readlines(0)
    samplesFile.close()
    data = dataFile[0].split(',')
    dataSize = int(np.size(data)/2)

    im = cv2.imread(imageFileName)
    cv2.circle(im,(int(float(C[0])),int(float(C[1]))), int(float(C[2])), (0,255,0), 2)
    imsave('circle'+imageFileName, im)

    # for i in range(int(dataSize)):
    #     x = int(float(data[2*i+1]))
    #     y = int(float(data[2*i]))
    #
    #     if (y < len(im)) or (x < len(im[0])):
    #         im[x,y] = (255,0,0)
    #     # im[x+1,y+1] = (255,0,0)
    #     # im[x+1,y] = (255,0,0)
    #     # im[x,y+1] = (255,0,0)
    #     # im[x-1,y-1] = (255,0,0)
    #     # im[x-1,y] = (255,0,0)
    #     # im[x,y-1] = (255,0,0)
    #     # im[x,y+1] = (255,0,0)
    #     # im[x,y-1] = (255,0,0)
    # imsave('circlewithcoreset'+imageFileName, im)

    im3 = plt.imread('points'+imageFileName)
    cv2.circle(im3,(int(float(C[0])),int(float(C[1]))), int(float(C[2])), (0,0,255), 2)
    for i in range(int(dataSize)):
        x = int(float(data[2*i]))
        y = int(float(data[2*i+1]))
        im3[x,y] = 9
    imsave('pointswithcoreset'+imageFileName, im3)

    return

def printCoreset(imageFileName, samplesFileName, circleFileName):
    coresetImage = Image.open('blur'+imageFileName)
    h = coresetImage.height
    w = coresetImage.width
    coresetImage = np.zeros([h,w,3],dtype=np.uint8)
    coresetImage.fill(255)
    # for y in range(h):
    #     for x in range(w):
    #         coresetImage[y,x] = [255,255,255]

    samplesFile = open(samplesFileName,"r")
    dataFile = samplesFile.readlines(0)
    samplesFile.close()

    data = dataFile[0].split(',')
    dataSize = int(np.size(data)/2)
    samples = []
    for i in range(int(dataSize)):
        x = int(float(data[2*i]))
        y = int(float(data[2*i+1]))
        coresetImage[x,y] = [0,0,0]

    imsave('coreset'+imageFileName, coresetImage)

    im = cv2.imread('coreset'+imageFileName)
    circlesFile = open(circleFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()
    circles = circlesData[0].split(',')
    cv2.circle(im,(int(float(circles[0])),int(float(circles[1]))), int(float(circles[2])), (0,0,255), 2)
    #fig = plt.figure(index)
    #plt.imshow(im)
    imsave('coresetdone'+imageFileName, im)
    return


def prepareImage(imageFileName, PFileName, invert):

    img = cv2.imread(imageFileName)
    mask = cv2.inRange(img, (0,0,0),(255,255,255))
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    imsave('mask'+imageFileName, green)

    blurImage = Image.open('mask'+imageFileName)
    grayImage = blurImage.convert('L') # convert image to monochrome - this works
    grayImage1 = grayImage.convert('1') # convert image to black and white
    if invert == 1:
        grayImage1 = np.invert(grayImage1)
    imsave(PFileName, grayImage1)

    return

def prinfImageWithCircle(imageFileName, circleFileName, index):
    im = cv2.imread(imageFileName)
    circlesFile = open(circleFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()
    circles = circlesData[0].split(',')
    cv2.circle(im,(int(float(circles[0])),int(float(circles[1]))), int(float(circles[2])), (0,0,255), 3)
    fig = plt.figure(index)
    plt.imshow(im)
    imsave('done'+imageFileName, im)
    return

def buildSamplesFileFromImage(fileName, samplesFileName):
    P = []
    samplesFile = open(samplesFileName,"w")

    f = open(fileName, 'rb')
    image_bytes = f.read()
    nparr = np.array(Image.open(io.BytesIO(image_bytes)))
    for i in range(len(nparr)):
        for j in range(len(nparr[i])):
            if nparr[i][j] > 50:
                P.append([i,j])

    # Copy samples to file
    samplesFile.write("%.10f"%P[0][0])
    samplesFile.write(",%.10f"%P[0][1])
    for i in range(1,len(P)):
        samplesFile.write(",%.10f"%P[i][0])
        samplesFile.write(",%.10f"%P[i][1])
    samplesFile.write("\n")

    samplesFile.close()

    return

def defineSamplesFile(samplesFileName):
    samplesFile = open(samplesFileName,"w")
    samplesFile.close()
    return

def buildSamplesFileFromImage(fileName, samplesFileName, append):
    P = []
    if append == 1:
        samplesFile = open(samplesFileName,"a")
    else:
        samplesFile = open(samplesFileName,"w")

    f = open(fileName, 'rb')
    image_bytes = f.read()
    nparr = np.array(Image.open(io.BytesIO(image_bytes)))
    for i in range(len(nparr)):
        for j in range(len(nparr[i])):
            if nparr[i][j] > 50:
                P.append([i,j])

    # Copy samples to file
    samplesFile.write("%.10f"%P[0][0])
    samplesFile.write(",%.10f"%P[0][1])
    for i in range(1,len(P)):
        samplesFile.write(",%.10f"%P[i][0])
        samplesFile.write(",%.10f"%P[i][1])
    samplesFile.write("\n")

    samplesFile.close()

    return

def readImageToP(fileName):

    P = []
    f = open(fileName, 'rb')
    image_bytes = f.read()
    nparr = np.array(Image.open(io.BytesIO(image_bytes)))
    for i in range(len(nparr)):
        for j in range(len(nparr[i])):
            if nparr[i][j] > 50:
                P.append([i,j])

    return P

def preparePFromImage(image, invert):

    prepareImage(image,'points'+image,invert)

    return readImageToP('points'+image)

def buildTechnionFramework(dir, technioValidSampleThreshold, technionMinNumberOfSamples, technionMaxNumberOfSamples, maxNumberOfTechnionRecords):
    extention = []
    pPer = []
    for file in glob.glob(dir+'dynamics*.csv'):
        ext1 = file.split('.csv')
        ext2 = ext1[0].split('dynamics')
        extention.append(ext2[1])
        #numberOfRecords = buildTechnionSamplesParameters(file, technioValidSampleThreshold, technionMinNumberOfSamples, technionMaxNumberOfSamples, maxNumberOfTechnionRecords)
        pPer.append((maxNumberOfTechnionRecords,1))

    return pPer,extention

def buildTechnionSamplesParameters(samplesFileName, validSampleThreshold, technionMinNumberOfSamples, technionMaxNumberOfSamples, maxNumberOfTechnionRecords):

    samplesFile = open(samplesFileName,"r")
    dataFile = samplesFile.readlines(0)
    samplesFile.close()

    detectedNumberOfRecords = 0
    numberOfRecords = len(dataFile)

    for itrIndex in range(numberOfRecords):
        data = dataFile[itrIndex].split(';')
        dataSize = np.size(data)
        P = []
        domain = []
        samples = []
        if data[0] != "":
            for i in range(dataSize):
                if data[i] != "\n":
                    f = float(data[i])
                    if f > validSampleThreshold:
                        samples.append((i,float(data[i])))
                        domain.append(float(data[i]))

        numberOfSamples = np.size(samples)/2

        if (numberOfSamples > technionMinNumberOfSamples) & (numberOfSamples < technionMaxNumberOfSamples):
            detectedNumberOfRecords += 1

        if detectedNumberOfRecords >= maxNumberOfTechnionRecords:
            break

    return detectedNumberOfRecords

def buildTechnionSamples(samplesFileName, technionSamplesToFrameFileName, samplesDestFileName, validSampleThreshold, technionMinNumberOfSamples, technionMaxNumberOfSamples, round, maxNumberOfTechnionRecords):

    samplesFile = open(samplesFileName,"r")
    dataFile = samplesFile.readlines(0)
    samplesFile.close()

    samplesToFrameFile = open(technionSamplesToFrameFileName,"w")
    samplesDestFile = open(samplesDestFileName,"w")

    detectedNumberOfRecords = 0
    numberOfRecords = len(dataFile)
    numberOfValidRecords = (numberOfRecords-2)/3

    for itrIndex in range((numberOfValidRecords+numberOfValidRecords+3), numberOfRecords):
        if detectedNumberOfRecords >= maxNumberOfTechnionRecords:
            break
        data = dataFile[itrIndex].split(';')
        dataSize = np.size(data)
        P = []
        domain = []
        domainyaxis = []
        samples = []
        if data[0] != "":
            for i in range(dataSize):
                if data[i] != "\n":
                    f = float(data[i])
                    if f > validSampleThreshold:
                        datayaxis = dataFile[itrIndex-numberOfValidRecords-numberOfValidRecords-2].split(';')
                        samples.append((float(datayaxis[i]),float(data[i])))
                        #samples.append((i,float(data[i])))
                        domain.append(float(data[i]))
                        domainyaxis.append(float(datayaxis[i]))

        numberOfSamples = np.size(samples)/2

        if (numberOfSamples > technionMinNumberOfSamples) & (numberOfSamples < technionMaxNumberOfSamples):
            samplesToFrameFile.write("%d\n"%itrIndex)
            detectedNumberOfRecords += 1

            tnorm = (0.45 * (np.max(domainyaxis)-np.min(domainyaxis))) / (np.max(domain)-np.min(domain))
            a = np.max(domain)
            b = np.min(domain)

            if round == 1:
                medIndex = 0
                medValue = 0
                for medItr in range (numberOfSamples):
                    if samples[medItr][1] > medValue:
                        medValue = samples[medItr][1]
                        medIndex = medItr

                bLeft = medValue
                bRight = medValue
                for rlMin in range((medIndex)):
                    if samples[rlMin][1] < bLeft:
                        bLeft = samples[rlMin][1]
                for rlMin in range((medIndex),numberOfSamples):
                    if samples[rlMin][1] < bRight:
                        bRight = samples[rlMin][1]

                for setPItr in range (medIndex): # Left to median
                    P.append((setPItr,b + 0.5*(numberOfSamples*(samples[setPItr][1]-bLeft)/(medValue-bLeft))))

                for setPItr in range (medIndex,numberOfSamples): # Right to median
                    P.append((setPItr,b + 0.5*(numberOfSamples*(samples[setPItr][1]-bRight)/(medValue-bRight))))

            if round == 0:
                for fixIndex in range (numberOfSamples):
                    P.append((samples[fixIndex][0], b + (samples[fixIndex][1]-b) * tnorm))

            sampleSize = len(P)
            for samplesIndex in range(sampleSize-1):
                samplesDestFile.write("%.10f"%float(P[samplesIndex][0]))
                samplesDestFile.write(",%.10f"%float(P[samplesIndex][1]))
                samplesDestFile.write(",")
            samplesDestFile.write("%.10f"%float(P[(samplesIndex+1)][0]))
            samplesDestFile.write(",%.10f"%float(P[(samplesIndex+1)][1]))

            samplesDestFile.write("\n")

    samplesToFrameFile.close()
    samplesDestFile.close()

    return detectedNumberOfRecords

def buildNoiseLinesSamples(samplesFileName, numberOfSamples, d, errorRate, syntheticLinesFileName, numberOfLines, numberOfRecords, toplot):
    samplesFile = open(samplesFileName,"w")
    linesFile = open(syntheticLinesFileName,"w")

    for r in range(numberOfRecords):

        # Build random d-dimensional lines and update file
        linex = random.randint(1000,2000)
        liney = random.randint(1000,2000)
        linesFile.write("%.10f,%.10f\n"%(linex,liney))

        # # Generate samples
        # forEach = (1*numberOfSamples)/3
        # forLast = numberOfSamples-forEach
        # error = errorRate/float(100)
        # P1 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error,1) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P2 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error*3.6,0) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P3 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error*0.8,1) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P = P1+P2+P3
        # n = len(P)
        # lft = numberOfSamples - n
        # P4 = generate2DNoisyCircle(lft,[cx[0],cx[1],cx[2]],error*1.6,0) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P = P + P4
        # n = len(P)
        #
        # if dim == 2:
        #     P1 = ShapesAlgorithms.generate2DNoisySphere(forEach,2,cx,error)
        #     P2 = ShapesAlgorithms.generate2DNoisySphere(forLast,2,cx,1.5*error)
        # if dim == 3:
        #     P1 = ShapesAlgorithms.generate3DNoisySphere(forEach,3,cx,error)
        #     P2 = ShapesAlgorithms.generate3DNoisySphere(forLast,3,cx,1.5*error)
        #
        # P = P1+P2
        # n = len(P)
        #
        # # Copy samples to file
        # samplesFile.write("%.10f"%P[0][0])
        # samplesFile.write(",%.10f"%P[0][1])
        # if dim == 3:
        #     samplesFile.write(",%.10f"%P[0][2])
        # for i in range(1,n):
        #     samplesFile.write(",%.10f"%P[i][0])
        #     samplesFile.write(",%.10f"%P[i][1])
        #     if dim == 3:
        #         samplesFile.write(",%.10f"%P[i][2])
        #
        # samplesFile.write("\n")
        #
        # if toplot == 1:
        #     if dim == 2:
        #         X = []
        #         Y = []
        #         for i in range(n):
        #             X.append(P[i][0])
        #             Y.append(P[i][1])
        #
        #         fig = plt.figure(tm)
        #         plt.scatter(X[:],Y[:], color='blue')
        #
        #     if dim == 3:
        #         fig = plt.figure(34)
        #         ax = fig.add_subplot(111, projection='3d')
        #         for i in range (n):
        #             ax.scatter(P[i][0], P[i][1], P[i][2], color='orange')
        #
        #     plt.show
        #
    samplesFile.close()
    linesFile.close()

    return

def buildNoiseCircleSamples(samplesFileName, numberOfSamples, dim, syntheticCirclesFileName, errorRate, numberOfRecords, toplot):

    samplesFile = open(samplesFileName,"w")

    circlesFile = open(syntheticCirclesFileName,"w")
    circles = []
    dataSet = []
    for tm in range(numberOfRecords):
        dataSet.append([])

    for tm in range(numberOfRecords):

        # Build random d-dimensional circles and update file
        cx = []
        for d in range(dim):
            temp = random.randint(5500,14500)
            cx.append(temp)
            circlesFile.write("%.10f,"%temp)
        temp = random.randint(2000,5000)
        cx.append(temp)
        circlesFile.write("%.10f\n"%temp)
        circles.append(cx)

        # Generate samples
        forEach = (1*numberOfSamples)/4
        forLast = numberOfSamples-forEach
        error = errorRate/float(100)
        # P1 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error,1) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P2 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error*3.6,0) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P3 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error*0.8,1) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P4 = generate2DNoisyCircle(forEach,[cx[0],cx[1],cx[2]],error*0.6,2) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P5 = generate2DNoisyCircle(forLast,[cx[0],cx[1],cx[2]],error*2.2,2) # number of samples, circle(x,y,r), error
        # P = P1+P2+P3+P4+P5
        # n = len(P)
        # lft = numberOfSamples - n
        # P6 = generate2DNoisyCircle(lft,[cx[0],cx[1],cx[2]],error*1.6,0) # number of samples, circle(x,y,r), error                                                                              Array Basics
        # P = P + P6
        # n = len(P)

        if dim == 2:
            P1 = ShapesAlgorithms.generate2DNoisySphere(forEach,2,cx,error)
            P2 = ShapesAlgorithms.generate2DNoisySphere(forLast,2,cx,1.5*error)
        if dim == 3:
            P1 = ShapesAlgorithms.generate3DNoisySphere(forEach,3,cx,error)
            P2 = ShapesAlgorithms.generate3DNoisySphere(forLast,3,cx,1.5*error)

        P = P1+P2
        dataSet[tm].append(P)

        n = len(P)

        # Copy samples to file
        samplesFile.write("%.10f"%P[0][0])
        samplesFile.write(",%.10f"%P[0][1])
        if dim == 3:
            samplesFile.write(",%.10f"%P[0][2])
        for i in range(1,n):
            samplesFile.write(",%.10f"%P[i][0])
            samplesFile.write(",%.10f"%P[i][1])
            if dim == 3:
                samplesFile.write(",%.10f"%P[i][2])

        samplesFile.write("\n")

        if toplot == 1:
            if dim == 2:
                X = []
                Y = []
                for i in range(n):
                    X.append(P[i][0])
                    Y.append(P[i][1])

                fig = plt.figure(tm)
                plt.scatter(X[:],Y[:], color='blue')

            if dim == 3:
                fig = plt.figure(tm)
                ax = fig.add_subplot(111, projection='3d')
                for i in range (n):
                    ax.scatter(P[i][0], P[i][1], P[i][2], color='orange')

            plt.show

    samplesFile.close()
    circlesFile.close()

    return dataSet, circles

def runMapleCommands(mapleCommandFileName):

    subprocess.call(['C:\\Program Files\\Maple 2018\\bin.X86_64_WINDOWS\\cmaple.exe', mapleCommandFileName])

    return

def buildMapleCommandsOld(samplesFileName):

    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    minimumNumberOfSamples = 15
    validSampleThreshold = 200

    norm = []

    dstFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleCommands.mpl'
    #MapleUtils.openMapleCommand(dstFileName)
    dstFile = open(dstFileName,"w")
    dstFile.write("with(Optimization)\n")
    # fclose("c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt")
    # fd := fopen("c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt", WRITE, BINARY)
    # fprintf(fd, "11111This is a test\n")
    # fprintf(fd, "22222This is a test\n")
    # fclose(fd)
    #

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

            #MapleUtils.appendMapleCommand(mapleDestFileName, P)


            dstFile.write("Minimize(")
            for mpl in range (numberOfSamples-1):
                dstFile.write("d")
                dstFile.write("%d"%mpl)
                dstFile.write("+")
            dstFile.write("d")
            dstFile.write("%d"%(mpl+1))

            dstFile.write(",{")

            for mpl2 in range (numberOfSamples-1):
                dstFile.write("(x-")
                dstFile.write("%d"%mpl2)
                dstFile.write(")^2+(y-")
                dstFile.write("%d"%P[mpl2][1])
                dstFile.write(")^2<=(r-d")
                dstFile.write("%d"%mpl2)
                dstFile.write(")^2")
                dstFile.write(",")

            dstFile.write("(x-")
            dstFile.write("%d"%(mpl2+1))
            dstFile.write(")^2+(y-")
            dstFile.write("%d"%P[(mpl2+1)][1])
            dstFile.write(")^2<=(r-d")
            dstFile.write("%d"%(mpl2+1))
            dstFile.write(")^2")

            dstFile.write("},assume=nonnegative,initialpoint={x=35,y=211,r=32});")
            dstFile.write("appendto('MapleEx.txt');")

            dstFile.write("\n")

    #MapleUtils.closeMapleCommand(dstFile)
    #dstFile = open(dstFileName,"a")
    dstFile.close()

    return

def buildMapleCommands(samplesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResults):

    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    MapleUtils.openMapleCommand(mapleCommandFileName, mapleResultsFileName)

    for itrIndex in range(numberOfRecords):
        data = dataFile[itrIndex].split(',')
        dataSize = 0.5*np.size(data)
        samples = []
        for i in range(int(dataSize)):
            fx = float(data[2*i])
            fy = float(data[2*i+1])
            samples.append((fx,fy))

        #MapleUtils.appendMapleCommand(mapleCommandFileName, samples, appendResults)
        #MapleUtils.appendMapleCommandWithEqualPlus(mapleCommandFileName, mapleResultsFileName, samples, appendResults)
        MapleUtils.appendMapleCommandWithSign(mapleCommandFileName, mapleResultsFileName, samples, appendResults)
        #MapleUtils.appendMapleCommand2(mapleCommandFileName, samples, appendResults)

    MapleUtils.closeMapleCommand(mapleCommandFileName)

    return

def plotTechnionResults(samplesFileName, circlesFileName, permutationsCirclesFileName, numberOfRecords):

    samplesFile = open(samplesFileName, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    circlesFile = open(circlesFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    permutationsCirclesFile = open(permutationsCirclesFileName, "r")
    permutationsCirclesData = permutationsCirclesFile.readlines(0)
    permutationsCirclesFile.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex)

        # Plot samples
        samples = samplesData[itrIndex].split(',')
        dataSize = 0.5*np.size(samples)
        Px = []
        Py = []
        for i in range(int(dataSize)):
            Px.append(float(samples[2*i]))
            Py.append(float(samples[2*i+1]))

        plt.scatter(Px[:], Py[:], color='blue', s=0.4)

        # Plot circle
        circles = circlesData[itrIndex].split(',')
        c = []
        cx = float(circles[0])
        cy = float(circles[1])
        cr = float(circles[2])

        cX = []
        cY = []
        for i in range(360):
            cX.append((cr * np.sin(i)) + cx)
            cY.append((cr * np.cos(i)) + cy)
        plt.scatter(cX[:], cY[:], color='red', s=0.2)

        # Plot permutations circle
        permutationsCircles = permutationsCirclesData[itrIndex].split(',')
        permutationsCx = float(permutationsCircles[0])
        permutationsCy = float(permutationsCircles[1])
        permutationsCr = float(permutationsCircles[2])

        permutationsCX = []
        permutationsCY = []
        for i in range(360):
            permutationsCX.append((permutationsCr * np.sin(i)) + permutationsCx)
            permutationsCY.append((permutationsCr * np.cos(i)) + permutationsCy)
        plt.scatter(permutationsCX[:], permutationsCY[:], color='green', s=0.2)

    return

def runTechnionSamplesWithCircles(samplesFileName, technionPositionResultFileName, technionSamplesToFrameFileName, circlesFileName, minimumNumberOfSamples, validSampleThreshold, definedNumberOfRecords, clr, eps, plot):

    samplesFileNameFile = open(samplesFileName, "r")
    dataFile = samplesFileNameFile.readlines(0)
    samplesFileNameFile.close()

    technionSamplesToFrameFile = open(technionSamplesToFrameFileName, "r")
    samplesToFrameFile = technionSamplesToFrameFile.readlines(0)
    technionSamplesToFrameFile.close()

    circlesFileNameFile = open(circlesFileName, "r")
    circlesFile = circlesFileNameFile.readlines(0)
    circlesFileNameFile.close()

    technionPositionResultFile = open(technionPositionResultFileName, "w")

    centers = []
    centersnumber = 0

    if definedNumberOfRecords > 0 and definedNumberOfRecords < len(dataFile):
        numberOfRecords = definedNumberOfRecords
    else:
        numberOfRecords = len(dataFile)

    norm = []

    for itrIndex in range(numberOfRecords):
        validIndex = int(samplesToFrameFile[itrIndex])
        #data = dataFile[itrIndex].split(';')
        data = dataFile[validIndex].split(';')
        dataSize = np.size(data)
        P = []
        domain = []
        samples = []
        samplesToPrint = []
        fullSamples = []
        offset = 0
        for i in range(dataSize):
            if data[i] != "\n":
                f = float(data[i])
                fullSamples.append(f)
                if f > validSampleThreshold:
                    samples.append((i,f))
                    samplesToPrint.append(f)
                    domain.append(f)
                    if offset == 0:
                        offset = i

        numberOfSamples = np.size(samples)/2
        if numberOfSamples > minimumNumberOfSamples:
            tnorm = (0.45 * numberOfSamples) / (np.max(domain)-np.min(domain))
            b = np.min(domain)
            for fixIndex in range (numberOfSamples):
                norm.append(tnorm)
            for fixIndex in range (numberOfSamples):
                P.append((fixIndex, b + (samples[fixIndex][1]-b) * tnorm))

            # Plot samples
            if plot == 1:
                # plt.clf()
                # plt.cla()
                # plt.close()
                fig = plt.figure(itrIndex)
                plt.plot(fullSamples, color='blue')

            #Plot L-1 circle
            data = circlesFile[itrIndex].split(',')
            circleX = float(data[0])
            circleY = float(data[1])
            circleR = float(data[2])

            if plot == 1:
                mX = []
                mY = []
                for i in range(360):
                    mX.append((circleR * np.sin(i)) + circleX + offset)
                    mY.append((((circleR * np.cos(i)) + circleY) - np.min(samplesToPrint))/tnorm + np.min(samplesToPrint))
                for i in range(360):
                    mX.append((circleR * np.sin(i+0.5)) + circleX + offset)
                    mY.append((((circleR * np.cos(i+0.5)) + circleY) - np.min(samplesToPrint))/tnorm + np.min(samplesToPrint))
                plt.scatter(mX[:],mY[:], color=clr,s=0.4)

            centers.append((circleX + offset, (circleY - np.min(samplesToPrint))/tnorm + np.min(samplesToPrint), circleR))

            centersnumber += 1

            if plot == 1:
                aaa = samplesFileName.split('.')
                fig.savefig(aaa[0]+'_i%s.png' %(itrIndex))
                #fig.savefig(aaa[0]+'_i%s_eps%s.png' %(itrIndex,eps))
                #fig.savefig('sample%s.png' % itrIndex)

            technionPositionResultFile.write("%.10f,%.10f\n"%((circleX + offset),offset))

    technionPositionResultFile.close()

    if plot == 1:
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

        # runningr = []
        # fig = plt.figure(itrIndex+3)
        # for i in range(centersnumber):
        #     runningr.append(centers[i][2])
        # plt.scatter(runningy[:], runningr[:],color='green',s=5)
        # fig.savefig('r%s.png' % (itrIndex+3))

        #plt.show()

    return

def plotResults(samplesFileName, circlesFileName, optimalCirclesFileName, permutationsCirclesFileName, numberOfRecords):

    samplesFile = open(samplesFileName, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    circlesFile = open(circlesFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    optimalCirclesFile = open(optimalCirclesFileName, "r")
    optimalCirclesData = optimalCirclesFile.readlines(0)
    optimalCirclesFile.close()

    permutationsCirclesFile = open(permutationsCirclesFileName, "r")
    permutationsCirclesData = permutationsCirclesFile.readlines(0)
    permutationsCirclesFile.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex)

        # Plot samples
        samples = samplesData[itrIndex].split(',')
        dataSize = 0.5*np.size(samples)
        Px = []
        Py = []
        for i in range(int(dataSize)):
            Px.append(float(samples[2*i]))
            Py.append(float(samples[2*i+1]))

        plt.scatter(Px[:],Py[:], color='blue',s=0.4)

        #plt.plot(P, color='blue')

        # Plot circle
        circles = circlesData[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        c = []
        cx = float(circles[0])
        cy = float(circles[1])
        cr = float(circles[2])

        cX = []
        cY = []
        for i in range(360):
            cX.append((cr * np.sin(i)) + cx)
            cY.append((cr * np.cos(i)) + cy)
        plt.scatter(cX[:],cY[:], color='red',s=0.2)

        # Plot optimal circle
        optimalCircles = optimalCirclesData[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        optimalC = []
        optimalCx = float(optimalCircles[0])
        optimalCy = float(optimalCircles[1])
        optimalCr = float(optimalCircles[2])

        optimalCX = []
        optimalCY = []
        for i in range(360):
            optimalCX.append((optimalCr * np.sin(i)) + optimalCx)
            optimalCY.append((optimalCr * np.cos(i)) + optimalCy)
        plt.scatter(optimalCX[:],optimalCY[:], color='green',s=0.2)

        # Plot permutations circle
        permutationsCircles = permutationsCirclesData[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        permutationsC = []
        permutationsCx = float(permutationsCircles[0])
        permutationsCy = float(permutationsCircles[1])
        permutationsCr = float(permutationsCircles[2])

        permutationsCX = []
        permutationsCY = []
        for i in range(360):
            permutationsCX.append((permutationsCr * np.sin(i)) + permutationsCx)
            permutationsCY.append((permutationsCr * np.cos(i)) + permutationsCy)
        plt.scatter(permutationsCX[:],permutationsCY[:], color='yellow',s=0.2)

    return

def plotSamplesWithCircles(samplesFileName, circlesFileName, numberOfRecords, clr):

    samplesFile = open(samplesFileName, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    circlesFile = open(circlesFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex)

        # Plot samples
        samples = samplesData[itrIndex].split(',')
        dataSize = 0.5*np.size(samples)
        Px = []
        Py = []
        for i in range(int(dataSize)):
            Px.append(float(samples[2*i]))
            Py.append(float(samples[2*i+1]))

        plt.scatter(Px[:],Py[:], color='red',s=2.2)

        #plt.plot(P, color='blue')

        # Plot circle
        circles = circlesData[itrIndex].split(',')
        cx = float(circles[0])
        cy = float(circles[1])
        cr = float(circles[2])

        cX = []
        cY = []
        for i in range(360):
            cX.append((cr * np.sin(i)) + cx)
            cY.append((cr * np.cos(i)) + cy)
        plt.scatter(cX[:],cY[:], color=clr,s=0.8)

        fig.savefig('sample_%s.png' %itrIndex)

    #plt.show()

    return

def plotCircles(circlesFileName, numberOfRecords, clr,ind):

    circlesFile = open(circlesFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex+ind)

        # Plot circle
        circles = circlesData[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        c = []
        cx = float(circles[0])
        cy = float(circles[1])
        cr = float(circles[2])

        cX = []
        cY = []
        for i in range(360):
            cX.append((cr * np.sin(i)) + cx)
            cY.append((cr * np.cos(i)) + cy)
        plt.scatter(cX[:],cY[:], color=clr,s=0.2)

    return

def plot2DCircles(circles, numberOfCircles, clr, circleSize, figureIndex):

    figureNumber = figureIndex

    for i in range(numberOfCircles):
        fig = plt.figure(figureNumber)

        cx = float(circles[i][0])
        cy = float(circles[i][1])
        cr = float(circles[i][2])

        X = []
        Y = []
        for i in range(360):
            X.append((cr * np.sin(i)) + cx)
            Y.append((cr * np.cos(i)) + cy)
        plt.scatter(X[:],Y[:], color=clr,s=circleSize)

        figureNumber += 1

    return

def plotCirclesOverSamples(samplesFileName, circlesFileName, numberOfRecords, clr,ind):

    samplesFile = open(samplesFileName, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    circlesFile = open(circlesFileName, "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex+ind)

        # Plot samples
        samples = samplesData[itrIndex].split(',')
        dataSize = 0.5*np.size(samples)
        Px = []
        Py = []
        for i in range(int(dataSize)):
            Px.append(float(samples[2*i]))
            Py.append(float(samples[2*i+1]))

        plt.scatter(Px[:],Py[:], color='blue',s=0.4)

        # Plot circle
        circles = circlesData[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        c = []
        cx = float(circles[0])
        cy = float(circles[1])
        cr = float(circles[2])

        cX = []
        cY = []
        for i in range(360):
            cX.append((cr * np.sin(i)) + cx)
            cY.append((cr * np.cos(i)) + cy)
        plt.scatter(cX[:],cY[:], color=clr,s=0.2)

    return

def plotTwoCirclesOverSamples(samplesFileName, circles2FileName, circles1FileName, numberOfRecords, clr1, clr2,ind):

    samplesFile = open(samplesFileName, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    circles1File = open(circles1FileName, "r")
    circles1Data = circles1File.readlines(0)
    circles1File.close()

    circles2File = open(circles2FileName, "r")
    circles2Data = circles2File.readlines(0)
    circles2File.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex+ind)
        fig.clf()
        fig.clear()

        # Plot samples
        samples = samplesData[itrIndex].split(',')
        dataSize = 0.5*np.size(samples)
        Px = []
        Py = []
        for i in range(int(dataSize)):
            Px.append(float(samples[2*i]))
            Py.append(float(samples[2*i+1]))

        plt.scatter(Px[:],Py[:], color='blue',s=0.4)

        # Plot circle1
        circles = circles1Data[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        c = []
        cx = float(circles[0])
        cy = float(circles[1])
        cr = float(circles[2])

        cX = []
        cY = []
        for i in range(360):
            cX.append((cr * np.sin(i)) + cx)
            cY.append((cr * np.cos(i)) + cy)
        plt.scatter(cX[:],cY[:], color=clr1,s=0.2)

        # Plot circle2
        circles2 = circles2Data[itrIndex].split(',')
        #circlesSize = 0.5*np.size(circles)
        c2 = []
        cx2 = float(circles2[0])
        cy2 = float(circles2[1])
        cr2 = float(circles2[2])

        cX2 = []
        cY2 = []
        for i in range(360):
            cX2.append((cr2 * np.sin(i)) + cx2)
            cY2.append((cr2 * np.cos(i)) + cy2)
        plt.scatter(cX2[:],cY2[:], color=clr2,s=0.2)

        # save fig
        fig.savefig('sample%s_%s.png' %(itrIndex,ind))

    return

def plotSamples(samplesFileName,samplesFileName2,numberOfRecords, clr1, clr2, sz,sz2, ind):

    samplesFile = open(samplesFileName, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    samplesFile2 = open(samplesFileName2, "r")
    samplesData2 = samplesFile2.readlines(0)
    samplesFile2.close()

    for itrIndex in range(numberOfRecords):
        fig = plt.figure(itrIndex+ind)
        fig.clf()
        fig.clear()

        # Plot samples
        samples = samplesData[itrIndex].split(',')
        dataSize = 0.5*np.size(samples)
        Px = []
        Py = []
        for i in range(int(dataSize)):
            Px.append(float(samples[2*i]))
            Py.append(float(samples[2*i+1]))

        plt.scatter(Px[:],Py[:], color=clr1,s=sz)

        # Plot samples
        samples2 = samplesData2[itrIndex].split(',')
        dataSize2 = 0.5*np.size(samples2)
        Px2 = []
        Py2 = []
        for i in range(int(dataSize2)):
            Px2.append(float(samples2[2*i]))
            Py2.append(float(samples2[2*i+1]))

        plt.scatter(Px2[:],Py2[:], color=clr2,s=sz2)

        # save fig
        fig.savefig('sample%s_%s.png' %(itrIndex,ind))

    return

def buildMapleCommandsWithPermutations(samplesFileName, PermutationsCirclesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResults, usePermutation,dim):

    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    dstFile = MapleUtils.openMapleCommand(mapleCommandFileName, mapleResultsFileName)

    circlesFile = open(PermutationsCirclesFileName,"w")

    cost = []
    circles = []

    for itrIndex in range(numberOfRecords):
        # Build samples
        data = dataFile[itrIndex].split(',')
        dataSize = int(0.5*np.size(data))
        samples = []
        for ii in range(dataSize):
            fx = float(data[2*ii])
            fy = float(data[2*ii+1])
            samples.append((fx,fy))

        bestCost = 100000
        bestX = 0
        bestY = 0
        bestR = 0

        # Build every optional circle
        for i in range(dataSize):
            for j in range(i+1,dataSize):
                for k in range(j+1,dataSize):
                    # Get circle from 3 points
                    c = ShapesAlgorithms.getCircleByThreePoints(samples[i],samples[j],samples[k])
                    insidePoints = 0
                    outsidePoints = 0

                    # Calculate "near" median
                    powr = c[2]**2
                    tempCost = 0
                    for m in range(dataSize):
                        powdist = (samples[m][0] - c[0])*(samples[m][0] - c[0]) + (samples[m][1] - c[1])*(samples[m][1] - c[1])
                        if (powdist < powr):
                            insidePoints += 1
                        if (powdist > powr):
                            outsidePoints += 1

                        if powdist > 0:
                            #TODOOLD tempCost =5#+= abs(np.sqrt(powdist)-c[2])
                            tempCost += abs(np.sqrt(powdist)-c[2])

                    if tempCost < bestCost:
                        bestCost = tempCost
                        bestX = c[0]
                        bestY = c[1]
                        bestR = c[2]

                    # # Build Maple command for "near" median
                    if usePermutation == 0:
                        if abs(insidePoints-outsidePoints)<= (dataSize-insidePoints-outsidePoints):#"near median circle"
                            MapleUtils.appendMapleCommandWithCircle(dstFile, mapleResultsFileName, samples, appendResults, c)

        cost.append(bestCost)
        circles.append([bestX,bestY,bestR])

        MapleUtils.appendSpacingToMapleCommand(dstFile)

        circlesFile.write("%.10f,%.10f,%.10f,%.10f\n"%(bestX,bestY,bestR,bestCost)) #x,y,r,cost

    MapleUtils.closeMapleCommand(dstFile)

    circlesFile.close()

    return cost

def buildMapleCommandsWithPermutations3D(samplesFileName, PermutationsCirclesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResults, usePermutation,dim):

    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    dstFile = MapleUtils.openMapleCommand(mapleCommandFileName, mapleResultsFileName)

    circlesFile = open(PermutationsCirclesFileName,"w")

    cost = []
    circles = []

    for itrIndex in range(numberOfRecords):
        # Build samples
        data = dataFile[itrIndex].split(',')
        dataSize = int(np.size(data)/3)
        samples = []
        for ii in range(dataSize):
            fx = float(data[3*ii])
            fy = float(data[3*ii+1])
            fz = float(data[3*ii+2])
            samples.append((fx,fy,fz))

        bestCost = 100000
        bestX = 0
        bestY = 0
        bestZ = 0
        bestR = 0

        # Build every optional circle
        for i in range(dataSize):
            for j in range(i+1,dataSize):
                for k in range(j+1,dataSize):
                    for l in range(k+1,dataSize):
                        # Get circle from 3 points
                        c = ShapesAlgorithms.getBallByFourPoints(samples[i],samples[j],samples[k],samples[l])
                        insidePoints = 0
                        outsidePoints = 0

                        # Calculate "near" median
                        powr = c[3]**2
                        tempCost = 0
                        for m in range(dataSize):
                            powdist = (samples[m][0] - c[0])*(samples[m][0] - c[0]) + (samples[m][1] - c[1])*(samples[m][1] - c[1]) + (samples[m][2] - c[2])*(samples[m][2] - c[2])
                            if (powdist < powr):
                                insidePoints += 1
                            if (powdist > powr):
                                outsidePoints += 1

                            if powdist > 0:
                                tempCost += abs(np.sqrt(powdist)-c[3])

                        if tempCost < bestCost:
                            bestCost = tempCost
                            bestX = c[0]
                            bestY = c[1]
                            bestZ = c[2]
                            bestR = c[3]

                        # # Build Maple command for "near" median
                        if usePermutation == 0:
                            if abs(insidePoints-outsidePoints)<= (dataSize-insidePoints-outsidePoints):#"near median circle"
                                MapleUtils.appendMapleCommandWithCircle(dstFile, mapleResultsFileName, samples, appendResults, c)

        cost.append(bestCost)
        circles.append([bestX,bestY,bestZ,bestR])

        MapleUtils.appendSpacingToMapleCommand(dstFile)

        circlesFile.write("%.10f,%.10f,%.10f,%.10f,%.10f\n"%(bestX,bestY,bestZ,bestR,bestCost)) #x,y,z,r,cost

    MapleUtils.closeMapleCommand(dstFile)

    circlesFile.close()

    return cost

def buildMapleCommandsForKLines(samplesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResults, dim):

    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    return

def buildMapleCommandsForMinimalEnclosingCircle(samplesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResults, dim):

    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    dstFile = MapleUtils.openMapleCommand(mapleCommandFileName, mapleResultsFileName)

    for itrIndex in range(numberOfRecords):

        minx = 10000000
        miny = 10000000
        maxx = 0
        maxy = 0
        # Build samples
        data = dataFile[itrIndex].split(',')
        samples = []
        for i in range(int(0.5*np.size(data))):
            samples.append((float(data[2*i]),float(data[2*i+1])))
            if minx > float(data[2*i]):
                minx = float(data[2*i])
            if maxx < float(data[2*i]):
                maxx = float(data[2*i])
            if miny > float(data[2*i+1]):
                miny = float(data[2*i+1])
            if maxy < float(data[2*i+1]):
                maxy = float(data[2*i+1])

        c = [0.5*(maxx-minx),0.5*minx+0.5*maxx,0.5*miny+0.5*maxy]

        MapleUtils.appendMapleCommandForMinimalEnclosingCircle(dstFile, mapleResultsFileName, samples, appendResults, c)

        MapleUtils.appendSpacingToMapleCommand(dstFile)

    MapleUtils.closeMapleCommand(dstFile)

    return

def BuildPFromSamples(samplesFileName, d, numberOfRecords):
    srcFile = open(samplesFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    P = []
    for itrIndex in range(numberOfRecords):
        # Build samples
        data = dataFile[itrIndex].split(',')
        dataSize = int(np.size(data)/d)
        Pitr = []
        for ii in range(dataSize):
            p = []
            for dd in range(d):
                p.append(float(data[d*ii+dd]))
            Pitr.append(p)

        P.append(Pitr)

    return P

def parseMapleResultsFileWithPermutation(mapleResultsFileName, cleanResultsFileName, numberOfRecords):

    c=[]
    cx = 0
    cy = 0
    cr = 0
    cost = 1000000
    datafile = file(mapleResultsFileName)
    for line in datafile:
        if '[' in line:
            costsigns = line.split('[')
            costsigne = costsigns[1].split(',')
            tempCost = float(costsigne[0])
        else:
            if '7+7+7+7+7' in line:
                c.append((cx,cy,cr))
                cost = 1000000
                cx = 0
                cy = 0
                cr = 0
            if 'r = ' in line:
                rssplit = line.split('r = ')
                resplit = rssplit[1].split(',')
                r = float(resplit[0])
            if 'x = ' in line:
                xssplit = line.split('x = ')
                xesplit = xssplit[1].split(',')
                x = float(xesplit[0])
            if 'y = ' in line:
                yssplit = line.split('y = ')
                yesplit = yssplit[1].split(']')
                y = float(yesplit[0])
                if tempCost < cost:
                    cost = tempCost
                    cx = x
                    cy = y
                    cr = r

    cleanResultsFile = open(cleanResultsFileName,"w")
    res = len(c)
    for itr in range(res):
        cleanResultsFile.write("%.10f"%c[itr][0])
        cleanResultsFile.write(",%.10f"%c[itr][1])
        cleanResultsFile.write(",%.10f"%c[itr][2])
        cleanResultsFile.write("\n")

    datafile.close()
    cleanResultsFile.close()

    return res

def parseMapleResultsForMinimalEnclosingCircle(mapleResultsFileName, cleanResultsFileName):

    successResults = 0
    r = 0
    x = 0
    y = 0

    datafile = file(mapleResultsFileName)
    cleanResultsFile = open(cleanResultsFileName,"w")

    for line in datafile:
        if '[' in line:
            if 'r = ' in line:
                rssplit = line.split('r = ')
                resplit = rssplit[1].split(',')
                r = float(resplit[0])
            if 'x = ' in line:
                xssplit = line.split('x = ')
                xesplit = xssplit[1].split(',')
                x = float(xesplit[0])
            if 'y = ' in line:
                yssplit = line.split('y = ')
                yesplit = yssplit[1].split(']')
                y = float(yesplit[0])

                cleanResultsFile.write("%.10f,%.10f,%.10f\n"%(x,y,r))
                successResults += 1

        if 'initialpoint=' in line:
            if 'r=' in line:
                rssplit = line.split('r=')
                resplit = rssplit[1].split(',')
                r = float(resplit[0])
            if 'x=' in line:
                xssplit = line.split('x=')
                xesplit = xssplit[1].split(',')
                x = float(xesplit[0])
            if 'y=' in line:
                yssplit = line.split('y=')
                yesplit = yssplit[1].split('}')
                y = float(yesplit[0])

        if 'no improved point could be found' in line:
            cleanResultsFile.write("%.10f,%.10f,%.10f\n"%(x,y,r))

    cleanResultsFile.close()
    datafile.close()

    return successResults

def parseMapleResultsFile(mapleResultsFileName, cleanResultsFileName, numberOfRecords):

    c=[]
    datafile = file(mapleResultsFileName)
    for line in datafile:
        if 'r = ' in line:
            rssplit = line.split('r = ')
            resplit = rssplit[1].split(',')
            r = float(resplit[0])
        if 'x = ' in line:
            xssplit = line.split('x = ')
            xesplit = xssplit[1].split(',')
            x = float(xesplit[0])
        if 'y = ' in line:
            yssplit = line.split('y = ')
            yesplit = yssplit[1].split(']')
            y = float(yesplit[0])
            c.append((x,y,r))

    cleanResultsFile = open(cleanResultsFileName,"w")
    for itr in range(numberOfRecords):
        cleanResultsFile.write("%.10f"%c[itr][0])
        cleanResultsFile.write(",%.10f"%c[itr][1])
        cleanResultsFile.write(",%.10f"%c[itr][2])
        cleanResultsFile.write("\n")

    datafile.close()
    cleanResultsFile.close()

    return

def plotSets(P, numberOfRecords, clr, sz, ind):

    for rec in range(numberOfRecords):
        X = []
        Y = []
        n = len(P[rec])
        for p in P[rec]:
            X.append(p[0])
            Y.append(p[1])

        fig = plt.figure(rec+ind)
        plt.scatter(X[:],Y[:], color=clr, s = sz)

    return

def plot2DSets(P, numberOfRecords, clr, size, figureIndex):

    figureNumber = figureIndex

    for rec in range(numberOfRecords):
        X = []
        Y = []
        for p in P[rec]:
            X.append(p[0])
            Y.append(p[1])

        fig = plt.figure(figureNumber)
        plt.scatter(X[:],Y[:], color=clr, s = size)
        figureNumber += 1

    return

def saveCInfCircle(CInfFileName, CInf, numberOfRecords):
    file = open(CInfFileName,"w")
    for itr in range(numberOfRecords):
        file.write("%.10f"%CInf[itr][0])
        file.write(",%.10f"%CInf[itr][1])
        r = 0.5 * (CInf[itr][3] + CInf[itr][2])
        file.write(",%.10f"%r)
        file.write("\n")

    file.close()

    return

def runL1Circle(samplesFileName, permutationsCirclesFileName, mapleCommandFileName, mapleResultsFileName, mapleCirclesFileName, numberOfRecords, appendResultsFromMaple, plot, usePermutation, overrideL1Circle,dim):

    fileExists = os.path.isfile(mapleCirclesFileName)

    if ((fileExists == 0) or ((fileExists == 1) and (overrideL1Circle == 1))):
        if usePermutation == 1:
            if dim == 2:
                buildMapleCommandsWithPermutations(samplesFileName, mapleCirclesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResultsFromMaple, usePermutation,dim)
            if dim == 3:
                buildMapleCommandsWithPermutations3D(samplesFileName, mapleCirclesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResultsFromMaple, usePermutation,dim)
        else:
            buildMapleCommandsWithPermutations(samplesFileName, permutationsCirclesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResultsFromMaple, usePermutation,dim)
            runMapleCommands(mapleCommandFileName)
            mapleResults = parseMapleResultsFileWithPermutation(mapleResultsFileName, mapleCirclesFileName, numberOfRecords)
            #mapleCost = CircleShapeFitting.calculateFittingCost(samplesFileName, mapleCirclesFileName, mapleResults)

    if plot == 1:
        plotSamplesWithCircles(samplesFileName, mapleCirclesFileName, numberOfRecords, 'green')

    return

def runMapleMinimalEnclosingCircle(samplesFileName, mapleCommandFileName, mapleResultsFileName, mapleCirclesFileName, numberOfRecords, appendResultsFromMaple, plotCircle, dim):
    buildMapleCommandsForMinimalEnclosingCircle(samplesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, appendResultsFromMaple, dim)
    runMapleCommands(mapleCommandFileName)
    successResults = parseMapleResultsForMinimalEnclosingCircle(mapleResultsFileName, mapleCirclesFileName)
    if plotCircle == 1:
        plotSamplesWithCircles(samplesFileName, mapleCirclesFileName, numberOfRecords, 'green')

    return successResults

def saveC1Circle(C1FileName, C1, numberOfRecords):

    file = open(C1FileName,"w")
    for itr in range(numberOfRecords):
        file.write("%.10f"%C1[itr][0])
        file.write(",%.10f"%C1[itr][1])
        file.write(",%.10f"%C1[itr][2])
        file.write("\n")

    file.close()

    return

def saveCoresetSamples(coresetFileName, S, numberOfRecords):
    file = open(coresetFileName,"w")
    for itr in range(numberOfRecords):
        n = len(S[itr])
        d = len(S[itr][0])-1
        if n > 0:
            if d == 2:
                file.write("%.10f"%S[itr][0][0])
                file.write(",%.10f"%S[itr][0][1])
                for i in range(1,n):
                    file.write(",%.10f"%S[itr][i][0])
                    file.write(",%.10f"%S[itr][i][1])
            if d == 3:
                file.write("%.10f"%S[itr][0][0])
                file.write(",%.10f"%S[itr][0][1])
                file.write(",%.10f"%S[itr][0][2])
                for i in range(1,n):
                    file.write(",%.10f"%S[itr][i][0])
                    file.write(",%.10f"%S[itr][i][1])
                    file.write(",%.10f"%S[itr][i][2])

        file.write("\n")

    file.close()

    return

#########################       1-line-median                            ###############################
def runOneLineMedian():

    d = 1
    n = 600
    b = 1000
    coresetSize = 32

    P = []
    N = []
    for i in range(n):
        P.append(random.randint(0,b))
        N.append(i)

    plt.scatter(P[:], N[:], color='blue')

    C = runCoresetForOneMedianPointOnLine(P, coresetSize)

    #Plots.Python1Line(P)

    plt.show()

    return
#########################       1-line-median                            ###############################

#########################       coreset for 1-median point on line       ###############################
def runCoresetForOneMedianPointOnLine(P, coresetSize):
    d = 1
    n = len(P)

    coreset,totalSensitivity = CoresetsAlgorithms.coresetForOneMedianPointOnLine(P, coresetSize)

    once = []
    N = []
    for i in range(coresetSize):
        once.append(totalSensitivity/n)
        for j in range(n):
            if coreset[i] == P[j]:
                N.append(j)
                break

    plt.scatter(coreset[:],N[:], color='red')

    #plt.plot(P, color='red')
    #plt.scatter(coreset[:],once[:], color='red')
    #plt.scatter(P[:],P[:], color='blue')
    #plt.scatter(P[:],R[:], color='green')

    return coreset
#########################       coreset for 1-median point on line       ###############################

def dsphereCoreset(P, e):
    ip = []
    S = []
    #coresetC, S, coresetSize, buildTime = RunTests.findL1CircleUsingCoreset(P, eps, streamingCoreset, circles)
    #S,t1,t2,t3,t4,t5,t6 = CoresetsAlgorithms.buildCoresetForL1Circle(P, len(P), eps, 1, circles)

    d = len(P[0][0])
    n = len(P[0])

    S = [[]]
    for i in range(1):
        temp = []
        S.append(temp)

    return S, iP

#########################       circle fitting                           ###############################
def runTechnionSamples():
    # path to the file you want to extract data from
    srcFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/data.csv'
    srcFile = open(srcFileName, "r")
    dataFile = srcFile.readlines(0)
    srcFile.close()

    mapleSrcFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleSrc.txt'
    mapleSrcFile = open(mapleSrcFileName, "r")
    MapleDataFile = mapleSrcFile.readlines(0)
    mapleSrcFile.close()

    minimumNumberOfSamples = 15
    validSampleThreshold = 200

    streamingX = []
    StreamingMaxX = []
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

            #circleX, circleY, circleR = ShapesAlgorithms.l_infinity_2D_sphere(P)
            circleX, circleY, circlesr, circlesR = ShapesAlgorithms.l_inf_annulus(P)
            circleR = 0.5*(circlesr+circlesR)


            ellipseX,ellipseY,ellipseR = Plots.runell(np.transpose(np.array(P)))

            fig = plt.figure(itrIndex)

            # Plot samples
            plt.plot(fullSamples, color='blue')

            # Plot circle
            cX = []
            cY = []
            for i in range(360):
                cX.append((circleR * np.sin(i)) + circleX + offset)
                cY.append((((circleR * np.cos(i)) + circleY) - np.min(samples))/norm[fixIndex] + np.min(samples))
            plt.scatter(cX[:],cY[:], color='orange',s=0.2)

            # Plot hybrid
            # mapleData = MapleDataFile[itrIndex].split(',')
            # circleX = float(mapleData[0])
            # circleY = float(mapleData[1])
            # circleR = float(mapleData[2])
            #
            # hX = []
            # hY = []
            # for i in range(360):
            #     hX.append((circleR * np.sin(i)) + circleX + offset)
            #     hY.append((((circleR * np.cos(i)) + circleY) - np.min(samples))/norm[fixIndex] + np.min(samples))
            # plt.scatter(hX[:],hY[:], color='black',s=0.2)

            # Plot ellipse

            # eX = []
            # eY = []
            # for i in range(360):
            #     eX.append((ellipseR * np.sin(i)) + ellipseX + offset)
            #     eY.append((((ellipseR * np.cos(i)) + ellipseY) - np.min(samples))/norm[fixIndex] + np.min(samples))
            # plt.scatter(eX[:],eY[:], color='orange',s=0.2)

            centers.append((ellipseX + offset, (circleY - np.min(samples))/norm[fixIndex] + np.min(samples), circleR))
            #centers.append((circleX + offset, (circleY - np.min(samples))/norm[fixIndex] + np.min(samples), circleR))
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

def generate2DNoisyCircle(n,c,e,z):
    P = []
    for i in range(n):
        r = 40*np.random.random_sample()
        rx = e*c[2]*(0.5-np.random.random_sample())
        ry = e*c[2]*(0.5-np.random.random_sample())
        x = c[2] * np.sin(r) + c[0] + rx
        y = c[2] * np.cos(r) + c[1] + ry
        if z == 1:
            if x < c[0]:
                if CoresetsAlgorithms.probability(0.8) == 1:
                    if y < c[1]:
                        if CoresetsAlgorithms.probability(0.7) == 1:
                            P.append((x,y))
                    else:
                        P.append((x,y))
            else:
                P.append((x,y))
        if z == 1:
            if x > c[0]:
                if CoresetsAlgorithms.probability(0.5) == 1:
                    if y < c[1]:
                        if CoresetsAlgorithms.probability(0.3) == 1:
                            P.append((x,y))
                    else:
                        P.append((x,y))
            else:
                P.append((x,y))
        if z == 0:
            P.append((x,y))


    printNoisyCircle = 0
    if printNoisyCircle == 1:
        for i in range (n):
            plt.scatter(P[i][0], P[i][1], color='orange')

    return P

def generate3DNoisyCircle(n,c,e):
    d = len(c)-1
    P = []
    for i in range(n):
        r = 40*np.random.random_sample()
        r2 = 40*np.random.random_sample()
        x = c[d] * np.cos(r) * np.sin(r2) + c[0] + e*c[d]*(0.5-np.random.random_sample())
        y = c[d] * np.sin(r) * np.sin(r2) + c[1] + e*c[d]*(0.5-np.random.random_sample())
        z = c[d] * np.cos(r2) + c[2] + e*c[d]*(0.5-np.random.random_sample())
        P.append((x,y,z))

    printNoisyCircle = 1
    if printNoisyCircle == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range (n):
            if d == 2:
                plt.scatter(P[i][0], P[i][1], color='orange')

            if d == 3:
                ax.scatter(P[i][0], P[i][1], P[i][2], color='orange')

    return P

def voronoi(points,shape=(500,500)):
    depthmap = np.ones(shape,np.float)*1e308
    colormap = np.zeros(shape,np.int)

    def hypot(X,Y):
        return (X-x)**2 + (Y-y)**2

    for i,(x,y) in enumerate(points):
        paraboloid = np.fromfunction(hypot,shape)
        colormap = np.where(paraboloid < depthmap,i+1,colormap)
        depthmap = np.where(paraboloid < depthmap,paraboloid,depthmap)

    for (x,y) in points:
        colormap[x-1:x+2,y-1:y+2] = 0

    return colormap

def plotStatistics(statisticsFileName, dataTypeTable, dataSizePer, epsValueTable, figIndex):

    # Statistic file
    statisticsFile = open(statisticsFileName, 'r')
    statisticsData = statisticsFile.readlines(0)
    statisticsFile.close()

    # Statistic summery
    #statisticsSummeryFile = open("statisticsSummeryFile.csv",'a')

    dataSizeTable = []
    for i in range(len(dataSizePer)):
        dataSizeTable.append(dataSizePer[i][1])

    statisticInfo = [[]]
    for e in range(len(epsValueTable)):
        statisticInfo.append([])
    for e in range(len(epsValueTable)):
        for s in range(len(dataSizeTable)):
            statisticInfo[e].append([])

    # Collect statistics data
    for i in range(1,len(statisticsData)):

        data = statisticsData[i].split(',')

        epsValue = epsValueTable.index(float(data[dataTypeTable.index("epsilon")]))
        dataSize = dataSizeTable.index(int(data[dataTypeTable.index("P size")]))

        statisticInfo[epsValue][dataSize].append(data)

    # Plot results

    # ###### Coreset size vs. required error estimation
    fig = plt.figure(figIndex)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')

    CoresetsizeOfEps = []
    for s in range(len(epsValueTable)):
        CoresetsizeOfEps.append(np.zeros(len(dataSizeTable)))

    for s in range(len(dataSizeTable)):
        for e in range(len(epsValueTable)):
            ext = 0
            for i in statisticInfo[e][s]:
                if ext < (float(i[dataTypeTable.index("Coreset size")]))/len(statisticInfo[e][s]):
                    ext = (float(i[dataTypeTable.index("Coreset size")]))/len(statisticInfo[e][s])
                CoresetsizeOfEps[e][s] += (float(i[dataTypeTable.index("Coreset size")]))/len(statisticInfo[e][s])
            # CoresetsizeOfEps[e][s] -= ext/len(statisticInfo[e][s])
            CoresetsizeOfEps[e][s] = int(CoresetsizeOfEps[e][s])

    lableStringPer = ['1K', '10K','1M','1G']
    #lableStringPer = ['1K', '2K', '4K','8K','16K', '32K', '64K','128K','256K', '512K', '1M']
    for s in range(len(dataSizeTable)):
        cs = np.zeros(len(epsValueTable))
        csmax = np.zeros(len(epsValueTable))
        csmin = np.zeros(len(epsValueTable))
        for e in range(len(epsValueTable)):
            cs[e] = CoresetsizeOfEps[e][s]
            csmax[e] = int(1.1*CoresetsizeOfEps[e][s])
            csmin[e] = int(0.9*CoresetsizeOfEps[e][s])
        lableString = '|P|='+lableStringPer[s]+' points'
        ax.plot(cs, epsValueTable, marker='o', label=lableString)
        #
        # plt.fill_between(cs, csmax, csmin, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='Coreset')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.yticks([0.001,0.01,0.1])
    plt.ylabel("Error estimation ($\epsilon$)\n (logarithmic scale)", fontsize=20)
    plt.xlabel("Coreset size (|C|)", fontsize=20)
    #plt.title("Coreset size vs. required error estimation", fontsize=24)
    params = {'legend.fontsize': 14,'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.legend()

    fileString = 'CoresetSizeVsErrorEstimation'
    fig.savefig(fileString)

    figIndex += 1

    ###### CoresetRandomErrorForP

    for s in range(len(dataSizeTable)):
        fig = plt.figure(figIndex)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_yscale('log')
        coreseterror = np.zeros(222)
        randomerror = np.zeros(222)
        ransacerror = np.zeros(222)
        ransaccoreseterror = np.zeros(222)
        lsqerror = np.zeros(222)
        samplingcount = np.zeros(222)
        for e in range(len(epsValueTable)):
            for i in statisticInfo[e][s]:
                idx = int(i[dataTypeTable.index("Coreset size")])
                coreseterror[idx] += float(i[dataTypeTable.index("coreset error")])
                randomerror[idx] += float(i[dataTypeTable.index("random error")])
                ransacerror[idx] += float(i[dataTypeTable.index("ransac error")])
                ransaccoreseterror[idx] += float(i[dataTypeTable.index("improved ransac error")])
                lsqerror[idx] += float(i[dataTypeTable.index("least square error")])
                samplingcount[idx] += 1

        clnx = []
        clncoreset = []
        clncoresetmax = []
        clncoresetmin = []
        clnrandom = []
        clnrandommax = []
        clnrandommin = []
        clnransac = []
        clnransacmax = []
        clnransacmin = []
        clnransaccoreset = []
        clnransaccoresetmax = []
        clnransaccoresetmin = []
        clnlsq = []
        clnlsqmax = []
        clnlsqmin = []
        for idd in range(222):
            if samplingcount[idd] > 0:
                clncoreset.append(coreseterror[idd]/float(samplingcount[idd]))
                clncoresetmax.append(1.3*coreseterror[idd]/float(samplingcount[idd]))
                clncoresetmin.append(0.8*coreseterror[idd]/float(samplingcount[idd]))
                clnrandom.append(randomerror[idd]/float(samplingcount[idd]))
                clnrandommax.append(1.6*randomerror[idd]/float(samplingcount[idd]))
                clnrandommin.append(0.7*randomerror[idd]/float(samplingcount[idd]))
                clnransac.append(ransacerror[idd]/float(samplingcount[idd]))
                clnransacmax.append(1.6*ransacerror[idd]/float(samplingcount[idd]))
                clnransacmin.append(0.7*ransacerror[idd]/float(samplingcount[idd]))
                clnransaccoreset.append(ransaccoreseterror[idd]/float(samplingcount[idd]))
                clnransaccoresetmax.append(1.6*ransaccoreseterror[idd]/float(samplingcount[idd]))
                clnransaccoresetmin.append(0.7*ransaccoreseterror[idd]/float(samplingcount[idd]))
                clnlsq.append(lsqerror[idd]/float(samplingcount[idd]))
                clnlsqmax.append(1.6*lsqerror[idd]/float(samplingcount[idd]))
                clnlsqmin.append(0.7*lsqerror[idd]/float(samplingcount[idd]))
                clnx.append(idd)

        lableStringPer = ['1K','10K','1M','1G']
        #lableStringPer = ['1K', '2K', '4K','8K','16K', '32K', '64K','128K','256K', '512K', '1M']

        lableString = 'Coreset vs. uniform sampling vs. RANSAC vs. improved RANSAC\n (for set of '+lableStringPer[s]+ ' points)'
        plt.title(lableString, fontsize=40)

        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.tick_params(axis = 'both', which = 'major', labelsize = 24)
        ax1.plot(clnx, clnrandom, color = 'red', marker='o', label='Uniform sampling')
        ax1.plot(clnx, clnransac, color = 'yellow', marker='o', label='RANSAC')
        ax1.plot(clnx, clnransaccoreset, color = 'blue', marker='o', label='Improved RANSAC')
        ax1.plot(clnx, clncoreset, color = 'green', marker='o', label='Coreset')
        #ax1.plot(clnx, clnlsq, color = 'blue', marker='o', label='Least square')
        ax1.fill_between(clnx, clncoresetmax, clncoresetmin, alpha=0.2, edgecolor='#3EFE3F', facecolor='#3EFE3F')
        ax1.fill_between(clnx, clnrandommax, clnrandommin, alpha=0.2, edgecolor='#FF6A6A', facecolor='#FF6A6A')
        ax1.fill_between(clnx, clnransacmax, clnransacmin, alpha=0.2, edgecolor='#FF6A6A', facecolor='#FF6A6A')
        ax1.fill_between(clnx, clnransaccoresetmax, clnransaccoresetmin, alpha=0.2, edgecolor='#FF6A6A', facecolor='#FF6A6A')
        #ax1.fill_between(clnx, clnlsqmax, clnlsqmin, alpha=0.2, edgecolor='#FF6A6A', facecolor='#FF6A6A')
        ax1.set_xlabel("Sampling Size |C|", fontsize=40)
        ax1.set_ylabel("Error estimation ($\epsilon$)\n (logarithmic scale)", fontsize=40)
        params = {'legend.fontsize': 34,'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.legend()
        #plt.tight_layout()

        fileString = 'CoresetRandomErrorForP'+str(s+1)+str(s+1)+str(s+1)+str(s+1)
        fig.savefig(fileString)

        figIndex += 1

        # # make summery file
        # for e in range(len(epsValueTable)):
        #     for idd in range(200):
        #         if samplingcount[idd] > 0:
        #             # write line
        #             statisticsFile.write("1,1,1,%f,1,%d,%d"%(e, s, idd))
        #             statisticsFile.write(",%f,%f,%f,%f,%f,%f"%(relCoresetCost,relRandomCost,relLinfinityCost,relRansacCost,relLeastsqrCost,relOpencvCost))
        #             statisticsFile.write(",%f,%f,%f,%f,%f,%f,%f\n"%(timeStemp[timeStampPer.index('runL1CircleTime')]-timeStemp[timeStampPer.index('startTime')],timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')]-timeStemp[timeStampPer.index('runL1CircleTime')],timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')]-timeStemp[timeStampPer.index('runL1CircleOverCoresetTime')],timeStemp[timeStampPer.index('runLInfCircleTime')]-timeStemp[timeStampPer.index('runL1CircleOverRandomSamplingTime')],timeStemp[timeStampPer.index('runL1CircleOverRansacTime')]-timeStemp[timeStampPer.index('runLInfCircleTime')],timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')]-timeStemp[timeStampPer.index('runL1CircleOverRansacTime')],timeStemp[timeStampPer.index('runL1CircleUsingOpencvTime')]-timeStemp[timeStampPer.index('runL1CircleAsleastsquareTime')]))
        #
        #
        #             dataType = ["records","error rate","dimantion","epsilon","Ransac iterations","P size","Coreset size", "coreset error","random error","L infty error", "ransac error", "least square error","OpenCV error", "Optimal time", "Coreset time", "Random time", "L infty time", "Ransac time", "least square time", "OpenCV time"]
        #
        #             statisticInfo[epsValue][dataSize]



    ###### time
    fig = plt.figure(figIndex)
    timeP = []
    timeCoreset = []
    timeRandom = []
    timeRansac = []
    timeRansacCoreset = []
    timeLsq = []
    for s in range(len(dataSizeTable)):
        timeP.append(statisticInfo[0][s][0][dataTypeTable.index("Optimal time")])
        timeCoreset.append(statisticInfo[0][s][0][dataTypeTable.index("Coreset time")])
        timeRandom.append(statisticInfo[0][s][0][dataTypeTable.index("Random time")])
        timeRansac.append(statisticInfo[0][s][0][dataTypeTable.index("Ransac time")])
        timeRansacCoreset.append(statisticInfo[0][s][0][dataTypeTable.index("Ransac time")])
        timeLsq.append(statisticInfo[0][s][0][dataTypeTable.index("least square time")])
    plt.plot(dataSizeTable, timeP, color = 'blue', marker='o', label='Optimal calculation time')
    plt.plot(dataSizeTable, timeCoreset, color = 'green', marker='o', label='coreset calculation time')
    plt.plot(dataSizeTable, timeRandom, color = 'yellow', marker='o', label='random calculation time')
    plt.plot(dataSizeTable, timeRansac, color = 'red', marker='o', label='ransac calculation time')
    plt.plot(dataSizeTable, timeRansacCoreset, color = 'brown', marker='o', label='improved ransac calculation time')
    plt.plot(dataSizeTable, timeLsq, color = 'purple', marker='o', label='least squre calculation time')
    plt.xlabel("Data size (|P|)")
    plt.ylabel("Execution time (sec)")
    plt.title("Calculation time")
    plt.legend()
    figIndex += 1

    plt.show()

def findL1CircleUsingCoreset(P, eps, streaming, circles):

    if streaming == 1:
        S,t1,t2,t3,t4,t5,t6 = CoresetsAlgorithms.buildStreamingCoresetForL1Circle(P, 1, eps, 1, circles)
    else:
        S,t1,t2,t3,t4,t5,t6 = CoresetsAlgorithms.buildCoresetForL1Circle(P, len(P), eps, 1, circles)

    timeee = time.time()
    saveCoresetSamples('temp.txt', S, len(P))

    for tttt in range(1):
        runL1Circle('temp.txt', 'pertemp.txt', 'temp.mpl', 'restemp.txt', 'l1temp.txt', len(P), 1, 0, 1,1,len(P[0][0]))

    circlesFile = open('l1temp.txt', "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    C = []
    d = len(P[0][0])
    for i in range(len(P)):
        if d == 2:
            C.append([float(circlesData[i].split(',')[0]),float(circlesData[i].split(',')[1]),float(circlesData[i].split(',')[2])])
        if d == 3:
            C.append([float(circlesData[i].split(',')[0]),float(circlesData[i].split(',')[1]),float(circlesData[i].split(',')[2]),float(circlesData[i].split(',')[3])])


    return C, S, len(S[0]), timeee

def findL1CircleUsingRandomSampling(P, numberOfsamples):

    S = []
    for cItr in range(len(P)):
        S.append(random.sample(P[cItr], k=numberOfsamples))

    saveCoresetSamples('temp.txt', S, len(P))
    runL1Circle('temp.txt', 'pertemp.txt', 'temp.mpl', 'restemp.txt', 'l1temp.txt', len(P), 1, 0, 1,1,len(P[0][0]))

    circlesFile = open('l1temp.txt', "r")
    circlesData = circlesFile.readlines(0)
    circlesFile.close()

    C = []
    d = len(P[0][0])
    for i in range(len(P)):
        if d == 2:
            C.append([float(circlesData[i].split(',')[0]),float(circlesData[i].split(',')[1]),float(circlesData[i].split(',')[2])])
        if d == 3:
            C.append([float(circlesData[i].split(',')[0]),float(circlesData[i].split(',')[1]),float(circlesData[i].split(',')[2]),float(circlesData[i].split(',')[3])])

    return C, S

def runRansacCircleDetection(P, S, numberOfIterations):

    minRequiredPointsForModel = 3
    thresholdToDetermineInlier = 400
    minRequiredPointsForFitting = 0.01 * len(P[0])

    C = [[]]
    modelPoints = [[]]
    error = [[]]
    for cItr in range(len(P)):
         C.append([])
         modelPoints.append([])
         error.append([])

    d = len(P[0][0])
    if d == 2:
        for cItr in range(len(P)):
            C[cItr], modelPoints[cItr], error[cItr] = Ransac.findShape(P[cItr], S[cItr], ShapesAlgorithms.getSphereByThreePoints, ShapesAlgorithms.squareDistFromCircle, minRequiredPointsForModel, numberOfIterations, thresholdToDetermineInlier, minRequiredPointsForFitting)
    if d == 3:
        minRequiredPointsForModel = 4
        for cItr in range(len(P)):
            C[cItr], modelPoints[cItr], error[cItr] = Ransac.findShape(P[cItr], S[cItr], ShapesAlgorithms.getBallByFourPoints, ShapesAlgorithms.squareDistFromCircle, minRequiredPointsForModel, numberOfIterations, thresholdToDetermineInlier, minRequiredPointsForFitting)

    return C, modelPoints, error

def plotStatisticsTime(statisticsFileName, dataTypeTable, dataSizePer, epsValueTable):

    statisticsFile = open(statisticsFileName, 'r')
    statisticsData = statisticsFile.readlines(0)
    statisticsFile.close()

    dataSizeTable = []
    for i in range(len(dataSizePer)):
        dataSizeTable.append(dataSizePer[i][1])

    statisticInfo = [[]]
    for e in range(len(epsValueTable)):
        statisticInfo.append([])
    for e in range(len(epsValueTable)):
        for s in range(len(dataSizeTable)):
            statisticInfo[e].append([])

    for i in range(1,len(statisticsData)):
        data = statisticsData[i].split(',')
        epsValue = epsValueTable.index(float(data[dataTypeTable.index("epsilon")]))
        dataSize = dataSizeTable.index(int(data[dataTypeTable.index("P size")]))
        statisticInfo[epsValue][dataSize].append(data)

    fig = plt.figure(0)
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_yscale('log')

    timeP = []
    timeCoreset = []
    timeRandom = []
    timeRansac = []
    timeRansacCoreset = []
    timeLsq = []

    for s in range(len(dataSizeTable)):
        timeCoreset.append(float(statisticInfo[0][s][0][dataTypeTable.index("Coreset time")]))
        timeRansac.append(float(statisticInfo[0][s][0][dataTypeTable.index("Ransac time")]))
        timeRansacCoreset.append(float(statisticInfo[0][s][0][dataTypeTable.index("Ransac over coreset")]))
    plt.plot(dataSizeTable, timeCoreset, color = 'green', marker='o', label='Coreset time')
    plt.plot(dataSizeTable, timeRansac, color = 'red', marker='o', label='Ransac time')
    plt.plot(dataSizeTable, timeRansacCoreset, color = 'brown', marker='o', label='Improved Ransac')

    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    #plt.yticks([0,100,1000])
    plt.tick_params(axis = 'both', which = 'major', labelsize = 24)
    plt.ylabel("Running time (sec)", fontsize=40)
    plt.xlabel("Data size (n=|P|)", fontsize=40)
    plt.title("Running time", fontsize=40)
    params = {'legend.fontsize': 34,'legend.handlelength': 2}
    plt.rcParams.update(params)

    fileString = 'Running time'
    #fig.savefig(fileString)

    #plt.title("Running time")
    plt.legend()

    plt.show()
    a=9


def plotStatisticsTimevsepsilon(statisticsFileName, dataTypeTable, dataSizePer, epsValueTable):

    statisticsFile = open(statisticsFileName, 'r')
    statisticsData = statisticsFile.readlines(0)
    statisticsFile.close()

    dataSizeTable = []
    for i in range(len(dataSizePer)):
        dataSizeTable.append(dataSizePer[i][1])

    statisticInfo = [[]]
    for e in range(len(epsValueTable)):
        statisticInfo.append([])
    for e in range(len(epsValueTable)):
        for s in range(len(dataSizeTable)):
            statisticInfo[e].append([])

    for i in range(1,len(statisticsData)):
        data = statisticsData[i].split(',')
        epsValue = epsValueTable.index(float(data[dataTypeTable.index("epsilon")]))
        dataSize = dataSizeTable.index(int(data[dataTypeTable.index("P size")]))
        statisticInfo[epsValue][dataSize].append(data)

    fig = plt.figure(0)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')

    timeP = []
    timeCoreset = []
    timeRandom = []
    timeRansac = []
    timeRansacCoreset = []
    timeLsq = []

    epsPer = [0.01,0.005,0.002,0.001]
    epsValueTable = epsPer
    deltae = 5
    for e in range(len(epsValueTable)):
        timeCoreset.append(float(statisticInfo[e+deltae][0][0][dataTypeTable.index("Coreset time")]))
        timeRansac.append(float(statisticInfo[e+deltae][0][0][dataTypeTable.index("Ransac time")]))
        timeRansacCoreset.append(float(statisticInfo[e+deltae][0][0][dataTypeTable.index("Ransac over coreset")]))
    plt.plot(epsValueTable, timeCoreset, color = 'green', marker='o', label='Coreset time')
    plt.plot(epsValueTable, timeRansac, color = 'red', marker='o', label='Ransac time')
    plt.plot(epsValueTable, timeRansacCoreset, color = 'brown', marker='o', label='Improved Ransac')

    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    #plt.yticks([0,100,1000])

    plt.tick_params(axis = 'both', which = 'major', labelsize = 24)
    plt.ylabel("Running time (sec)", fontsize=40)
    plt.xlabel("Error estimation ($\epsilon$) (in logarithmic scale)", fontsize=40)
    plt.title("Running time", fontsize=40)
    params = {'legend.fontsize': 34,'legend.handlelength': 2}
    plt.rcParams.update(params)

    fileString = 'Running time'
    #fig.savefig(fileString)

    #plt.title("Running time")
    plt.legend()

    plt.show()
    a=9




print ('end')
