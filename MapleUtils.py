import random
import numpy as np
import matplotlib.pyplot as plt
import CoresetsAlgorithms
import ShapesAlgorithms
import Plots
import scipy
from PIL import Image
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

def writeMapleCommand(dstFileName, P):
    #dstFileName = r'c://Users/RBD-W540/PycharmProjects/Coresets/MapleCommands.mpl'
    dstFile = open(dstFileName,"w")
    dstFile.write("with(Optimization)\n")

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%d"%mpl2)
        dstFile.write(")^2+(y-")
        dstFile.write("%d"%P[mpl2][1])
        dstFile.write(")^2=(r-d")
        dstFile.write("%d"%mpl2)
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%d"%(mpl2+1))
    dstFile.write(")^2+(y-")
    dstFile.write("%d"%P[(mpl2+1)][1])
    dstFile.write(")^2=(r-d")
    dstFile.write("%d"%(mpl2+1))
    dstFile.write(")^2")

    gessX = 0.5*len(P)
    aa = []
    for pindex in range(len(P)):
        aa.append(P[pindex][1])
    gessY = 0.5*(np.max(aa)+np.min(aa))
    gessR = 0.5*len(P)
    dstFile.write("},assume=nonnegative,initialpoint={x=%d"%gessX)
    dstFile.write(",y=%d"%gessY)
    dstFile.write(",r=%d"%gessR)
    dstFile.write("});")
    #dstFile.write("appendto('MapleEx.txt');")

    dstFile.write("\n")

    dstFile.close()

    return

def closeMapleCommand(dstFile):
    #dstFile = open(dstFileName,"a")
    dstFile.write("writeto(\"c://Users/RBD-W540/PycharmProjects/Coresets/temp.txt\");\n")
    dstFile.close()

    return

def openMapleCommand(dstFileName, mapleResultsFileName):
    dstFile = open(dstFileName,"w")
    dstFile.write("with(Optimization);\nwith(LinearAlgebra);\n")
    dstFile.write("writeto(\"")
    dstFile.write(mapleResultsFileName)
    dstFile.write("\");\n")

    return dstFile

def appendSpacingToMapleCommand(dstFile):
    #dstFile = open(dstFileName,"a")
    dstFile.write("7+7+7+7+7;\n")

    return

def appendMapleCommandWithCircle(dstFile, mapleResultsFileName, P, appendResultsFromMaple, c):
    powr = c[2]**2
    itr = len(P)-1

    dstFile.write("Minimize(")
    for mpl in range (itr):
        dstFile.write("d%d+"%mpl)
    dstFile.write("d%d,{"%itr)

    for mpl2 in range (itr):
        dstFile.write("(x-%.10f)^2+(y-%.10f"%(P[mpl2][0],P[mpl2][1]))
        if ((P[mpl2][0] - c[0])**2 + (P[mpl2][1] - c[1])**2) > powr:# outside
            dstFile.write(")^2=(r+d%d)^2,"%mpl2) # outside
        else:# inside
            dstFile.write(")^2=(r-d%d)^2,"%mpl2) # inside

    dstFile.write("(x-%.10f)^2+(y-%.10f"%(P[mpl2+1][0],P[mpl2+1][1]))
    if ((P[mpl2+1][0] - c[0])**2 + (P[mpl2+1][1] - c[1])**2) > powr:# outside
        dstFile.write(")^2=(r+d%d)^2"%(mpl2+1)) # outside
    else:# inside
        dstFile.write(")^2=(r-d%d)^2"%(mpl2+1)) # inside

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f,y=%.10f,r=%.10f});"%(c[0],c[1],c[2]))
    if appendResultsFromMaple == 1:
        dstFile.write("appendto(\"")
        dstFile.write(mapleResultsFileName)
        dstFile.write("\");")

    dstFile.write("\n")

    return

def appendMapleCommandForMinimalEnclosingCircle(dstFile, mapleResultsFileName, P, appendResultsFromMaple, initial):
    #Minimize(r, {(x-p1x)^2+(y-p2y)^2 <= r^2, ... , (x+pnx)^2+(y-pny)^2 <= r^2}, initialpoint = {r = 1.3, x = 1, y = 1.1})

    dstFile.write("Minimize(r,{")
    for i in range (len(P)-1):
        dstFile.write("(x-%.10f)^2+(y-%.10f)^2<=r^2,"%(P[i][0],P[i][1]))
    dstFile.write("(x-%.10f)^2+(y-%.10f)^2<=r^2"%(P[len(P)-1][0],P[len(P)-1][1]))
    dstFile.write("},assume=nonnegative,initialpoint={r=%.10f,x=%.10f,y=%.10f},iterationlimit = 3);"%(initial[0],initial[1],initial[2]))
    #dstFile.write("},assume=nonnegative,initialpoint={x=%.10f,y=%.10f});"%(initial[1],initial[2]))

    if appendResultsFromMaple == 1:
        dstFile.write("appendto(\"")
        dstFile.write(mapleResultsFileName)
        dstFile.write("\");")

    dstFile.write("\n")

    return

def appendMapleCommandWithCircleaaaaaaaa(dstFile, mapleResultsFileName, P, appendResultsFromMaple, c):
#dstFile = open(dstFileName,"a")
    powr = c[2]**2

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        if ((P[mpl2][0] - c[0])**2 + (P[mpl2][1] - c[1])**2) > powr:# outside
            dstFile.write(")^2=(r+d") # outside
        else:# inside
            dstFile.write(")^2=(r-d") # inside
        dstFile.write("%d"%mpl2)
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%.10f"%P[mpl2+1][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2+1][1])
    if ((P[mpl2+1][0] - c[0])**2 + (P[mpl2+1][1] - c[1])**2) > powr:# outside
        dstFile.write(")^2=(r+d") # outside
    else:# inside
        dstFile.write(")^2=(r-d") # inside
    dstFile.write("%d"%(mpl2+1))
    dstFile.write(")^2")

    yy = []
    xx = []
    for pindex in range(len(P)):
        xx.append(P[pindex][0])
        yy.append(P[pindex][1])
    gessY = 0.5*(np.max(yy)+np.min(yy))
    gessX = 0.5*(np.max(xx)+np.min(xx))
    gessR = 0.5*max((np.max(xx)-np.min(xx)),(np.max(yy)-np.min(yy)))

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f"%gessX)
    dstFile.write(",y=%.10f"%gessY)
    dstFile.write(",r=%.10f"%gessR)
    dstFile.write("});")
    if appendResultsFromMaple == 1:
        dstFile.write("appendto(\"")
        dstFile.write(mapleResultsFileName)
        dstFile.write("\");")
        #dstFile.write("appendto(\"c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt\");")

    dstFile.write("\n")

    return

def appendMapleCommand(dstFileName, P, appendResults):
    dstFile = open(dstFileName,"a")

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        dstFile.write(")^2=(r-d")
        dstFile.write("%d"%P[mpl2][0])
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%.10f"%P[mpl2+1][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2+1][1])
    dstFile.write(")^2=(r-d")
    dstFile.write("%d"%P[mpl2+1][0])
    dstFile.write(")^2")

    yy = []
    xx = []
    for pindex in range(len(P)):
        xx.append(P[pindex][0])
        yy.append(P[pindex][1])
    gessY = 0.5*(np.max(yy)+np.min(yy))
    gessX = 0.5*(np.max(xx)+np.min(xx))
    gessR = 0.5*max((np.max(xx)-np.min(xx)),(np.max(yy)-np.min(yy)))

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f"%gessX)
    dstFile.write(",y=%.10f"%gessY)
    dstFile.write(",r=%.10f"%gessR)
    dstFile.write("});")
    if appendResults == 1:
        dstFile.write("appendto('MapleResults.txt');")

    dstFile.write("\n")

    return

def appendMapleCommandWithSign(dstFileName, mapleResultsFileName, P, appendResults):
    dstFile = open(dstFileName,"a")

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        dstFile.write(")^2=(r+sign((x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        dstFile.write(")^2-r^2")
        dstFile.write(")*d")
        dstFile.write("%d"%mpl2)
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%.10f"%P[mpl2+1][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2+1][1])
    dstFile.write(")^2=(r+sign((x-")
    dstFile.write("%.10f"%P[mpl2][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2][1])
    dstFile.write(")^2-r^2")
    dstFile.write(")*d")
    dstFile.write("%d"%(mpl2+1))
    dstFile.write(")^2")

    yy = []
    xx = []
    for pindex in range(len(P)):
        xx.append(P[pindex][0])
        yy.append(P[pindex][1])
    gessY = 0.5*(np.max(yy)+np.min(yy))
    gessX = 0.5*(np.max(xx)+np.min(xx))
    gessR = 0.5*max((np.max(xx)-np.min(xx)),(np.max(yy)-np.min(yy)))

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f"%gessX)
    dstFile.write(",y=%.10f"%gessY)
    dstFile.write(",r=%.10f"%gessR)
    dstFile.write("});")
    if appendResults == 1:
        dstFile.write("appendto(\"")
        dstFile.write(mapleResultsFileName)
        dstFile.write("\");")
        #dstFile.write("appendto(\"c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt\");")

    dstFile.write("\n")

    return

def appendMapleCommandWithEqualMinus(dstFileName, mapleResultsFileName, P, appendResults):
    dstFile = open(dstFileName,"a")

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        dstFile.write(")^2=(r-d")
        dstFile.write("%d"%mpl2)
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%.10f"%P[mpl2+1][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2+1][1])
    dstFile.write(")^2=(r-d")
    dstFile.write("%d"%(mpl2+1))
    dstFile.write(")^2")

    yy = []
    xx = []
    for pindex in range(len(P)):
        xx.append(P[pindex][0])
        yy.append(P[pindex][1])
    gessY = 0.5*(np.max(yy)+np.min(yy))
    gessX = 0.5*(np.max(xx)+np.min(xx))
    gessR = 0.5*max((np.max(xx)-np.min(xx)),(np.max(yy)-np.min(yy)))

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f"%gessX)
    dstFile.write(",y=%.10f"%gessY)
    dstFile.write(",r=%.10f"%gessR)
    dstFile.write("});")
    if appendResults == 1:
        dstFile.write("appendto(\"")
        dstFile.write(mapleResultsFileName)
        dstFile.write("\");")
        #dstFile.write("appendto(\"c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt\");")

    dstFile.write("\n")

    return

def appendMapleCommandWithEqualPlus(dstFileName, mapleResultsFileName, P, appendResults):
    dstFile = open(dstFileName,"a")

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        dstFile.write(")^2=(r+d")
        dstFile.write("%d"%mpl2)
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%.10f"%P[mpl2+1][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2+1][1])
    dstFile.write(")^2=(r+d")
    dstFile.write("%d"%(mpl2+1))
    dstFile.write(")^2")

    yy = []
    xx = []
    for pindex in range(len(P)):
        xx.append(P[pindex][0])
        yy.append(P[pindex][1])
    gessY = 0.5*(np.max(yy)+np.min(yy))
    gessX = 0.5*(np.max(xx)+np.min(xx))
    gessR = 0.5*max((np.max(xx)-np.min(xx)),(np.max(yy)-np.min(yy)))

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f"%gessX)
    dstFile.write(",y=%.10f"%gessY)
    dstFile.write(",r=%.10f"%gessR)
    dstFile.write("});")
    if appendResults == 1:
        dstFile.write("appendto(\"")
        dstFile.write(mapleResultsFileName)
        dstFile.write("\");")
        #dstFile.write("appendto(\"c://Users/RBD-W540/PycharmProjects/Coresets/MapleResults.txt\");")

    dstFile.write("\n")

    return

def appendMapleCommandLeq(dstFileName, P, appendResults):
    dstFile = open(dstFileName,"a")

    dstFile.write("Minimize(")
    for mpl in range (len(P)-1):
        dstFile.write("d")
        dstFile.write("%d"%mpl)
        dstFile.write("+")
    dstFile.write("d")
    dstFile.write("%d"%(mpl+1))

    dstFile.write(",{")

    for mpl2 in range (len(P)-1):
        dstFile.write("(x-")
        dstFile.write("%.10f"%P[mpl2][0])
        dstFile.write(")^2+(y-")
        dstFile.write("%.10f"%P[mpl2][1])
        dstFile.write(")^2<=(r-d")
        dstFile.write("%d"%P[mpl2][0])
        dstFile.write(")^2")
        dstFile.write(",")

    dstFile.write("(x-")
    dstFile.write("%.10f"%P[mpl2+1][0])
    dstFile.write(")^2+(y-")
    dstFile.write("%.10f"%P[mpl2+1][1])
    dstFile.write(")^2<=(r-d")
    dstFile.write("%d"%P[mpl2+1][0])
    dstFile.write(")^2")

    yy = []
    xx = []
    for pindex in range(len(P)):
        xx.append(P[pindex][0])
        yy.append(P[pindex][1])
    gessY = 0.5*(np.max(yy)+np.min(yy))
    gessX = 0.5*(np.max(xx)+np.min(xx))
    gessR = 0.5*max((np.max(xx)-np.min(xx)),(np.max(yy)-np.min(yy)))

    dstFile.write("},assume=nonnegative,initialpoint={x=%.10f"%gessX)
    dstFile.write(",y=%.10f"%gessY)
    dstFile.write(",r=%.10f"%gessR)
    dstFile.write("});")
    if appendResults == 1:
        dstFile.write("appendto('MapleResults.txt');")

    dstFile.write("\n")

    return
