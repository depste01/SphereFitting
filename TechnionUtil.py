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

def searchTechnion(path):

    exp = []
    for root, dirs, files in os.walk(path):
        lvmfile = ""
        csvfile = ""
        for file in files:
            if file.endswith((".lvm")):
                lvmfile = file
            if file.endswith((".csv")):
                csvfile = file
        if (lvmfile != "") and (csvfile != ""):
            eee = []
            eee.append(root)
            eee.append(lvmfile)
            eee.append(csvfile)
            exp.append(eee)

    return exp

def searchPressure(path):

    samplesFile = open(path, "r")
    samplesData = samplesFile.readlines(0)
    samplesFile.close()

    start = 0
    time = 0
    st = 1
    sens = 0
    found = 0
    for line in samplesData:
        data = line.split('\t')
        if data[0] == "0.000000":
            start = 1

        if (start > 10) and (found == 0):
            for i in range(1,8):
                if float(data[i]) < -0.6:
                    time = data[0]
                    sens = i
                    found = 1
        if start > 0:
            start += 1
        st += 1

    #a, b = np.polyfit(x, y, 1)

    return time, st, sens

def buildResultsFile(samplesFileName, technionPositionResultFileName, technionSamplesToFrameFileName, circlesFileName, minimumNumberOfSamples, validSampleThreshold, numberOfRecords, rootPass, pressureFileName,pressureTime, pressureSensor, sensor):

    samplesFileNameFile = open(samplesFileName, "r")
    dataFile = samplesFileNameFile.readlines(0)
    samplesFileNameFile.close()

    pressureFileNameFile = open(pressureFileName, "r")
    pressureDataFile = pressureFileNameFile.readlines(0)
    pressureFileNameFile.close()

    technionSamplesToFrameFile = open(technionSamplesToFrameFileName, "r")
    samplesToFrameFile = technionSamplesToFrameFile.readlines(0)
    technionSamplesToFrameFile.close()

    circlesFileNameFile = open(circlesFileName, "r")
    circlesFile = circlesFileNameFile.readlines(0)
    circlesFileNameFile.close()

    aa = samplesFileName.split('.csv')
    bb = aa[0].split('mics')
    technionPositionResultFile = open(technionPositionResultFileName+bb[1]+".csv", "w")
    technionPositionResultFile.write("Dynamics frame number,    Circle height,  Circle velocity,    Circle accelaration,    Circle hit indication,  Pressure hit time\n")

    numberOfValidRecords = (len(dataFile)-2)/3

    xprev = 0
    xdprev = 0
    vprev = 0

    hitIndication = 0
    hitIndicationUp = 1
    for itrIndex in range(numberOfRecords):
        validIndex = int(samplesToFrameFile[itrIndex])
        data = dataFile[validIndex].split(';')
        samples = []
        offset = 0
        for i in range(np.size(data)):
            if data[i] != "\n":
                if float(data[i]) > validSampleThreshold:
                    datayaxis = dataFile[validIndex-numberOfValidRecords-numberOfValidRecords-2].split(';')
                    samples.append((float(datayaxis[i]),float(data[i])))
                    #samples.append((i,float(data[i])))
                    if offset == 0:
                        offset = float(datayaxis[i])

        c = circlesFile[itrIndex].split(',')
        x = 0.84*(float(c[0]) + offset)
        printv = 0
        printa = 0.0
        if itrIndex > 1:

            printx = (x+xprev+xdprev)/3
            if itrIndex > 2:
                printv = ((x-printx)*240)/1000
            if itrIndex > 3:
                printa = ((printv-vprev)*240)/1000

            if x > xprev:
                hitIndication = 1*hitIndicationUp
                hitIndicationUp = 0

            technionPositionResultFile.write("%d,   %.10f,  %.10f,  %.10f,  %d, %.4f\n"%(printValidIndex,printx,printv,printa,hitIndication,float(pressureTime)))
            hitIndication = 0

        printValidIndex = validIndex
        xdprev = xprev
        xprev = x
        vprev = printv

    technionPositionResultFile.close()

    os.remove(rootPass+"/samples.txt")
    os.remove(rootPass+"/technionSamplesToFrame.txt")
    os.remove(rootPass+"/CoresetSamples.txt")

    return
