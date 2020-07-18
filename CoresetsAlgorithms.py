# import sys
import random
# import math
# import array
# import numpy
# import sys
# import random
import math
# import array
# import os
# from mpl_toolkits.mplot3d import Axes3D
# import skipy
# from skipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
# import scipy
# from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
import scipy
from scipy.signal import medfilt2d
from scipy.spatial import ConvexHull
import ShapesAlgorithms
import RunTests
from operator import itemgetter, attrgetter

def buildCoresetForLInfCircleUsingCircleGrid(P, CInf, eps):

    # 1. Find annulus width
    n = len(P)
    cx = CInf[0]
    cy = CInf[1]
    cr = 0.5*(CInf[2]+CInf[3])

    w = CInf[3] - CInf[2]

    # 2. Build projected set
    numberOfCircles = 0.5*w/eps
    numberOfCircles = 50 ###################!!!!!!!!!!!!!!!!!!!!!!!!
    tagP = [[]]
    for t in range(numberOfCircles):
        temp = []
        tagP.append(temp)

    for i in range(n):

        # Find circle index
        ind = int(numberOfCircles*((scipy.sqrt((cx-P[i][0])**2 + (cy-P[i][1])**2) - CInf[2])/(CInf[3]-CInf[2])))

        # Project point on circle - NOT IMPORT TO PROJECT, JUST TO ASSOCIATE TO CIRCLE !!!
        tagp = P[i]

        # Append P' with projected point in the correct index
        tagP[ind].append(tagp)

    # 3. Build coreset S
    S = []
    for i in range(numberOfCircles):
        cn = len(tagP[i])
        prob = float(1.5*3/(1+float(cn))) # 3 - for circle. maybe need more for L1...

        for j in range(cn):
            if probability(prob) == 1:
                S.append(tagP[i][j])

    return S

def buildRandomSamplesForL1Circle(P, numberOfRecords, S):
    d = len(P[0][0])
    n = len(P[0])

    R = [[]]
    for t in range(numberOfRecords):
        temp = []
        R.append(temp)

    for cItr in range(numberOfRecords):
        prob = float(len(S[cItr]))/float(n)
        for i in range(n):
            if probability(prob) == 1:
                R[cItr].append(P[cItr][i])
                if len(R[cItr]) == len(S[cItr]):
                    break

        prob2 = float(len(S[cItr])-len(R[cItr]))/float(n)
        if len(R[cItr]) < len(S[cItr]):
            for j in range(n):
                if probability(prob2) == 1:
                    R[cItr].append(P[cItr][j])
                    if len(R[cItr]) == len(S[cItr]):
                        break

        if len(R[cItr]) < len(S[cItr]):
            for j in range(n):
                if probability(prob2) == 1:
                    R[cItr].append(P[cItr][j])
                    if len(R[cItr]) == len(S[cItr]):
                        break

    return R

def buildStreamingCoresetForL1Circle(P, numberOfRecords, eps, type, circles):

    d = len(P[0][0])
    n = len(P[0])

    S = [[]]
    streamS = [[]]
    for i in range(numberOfRecords):
        S.append([])
        streamS.append([])

    for cItr in range(numberOfRecords):

        n = len(P[cItr])
        numberOfStreams = pow(10,(int(math.log10(n)/2)))
        tagn = int(n/numberOfStreams)
        PP = np.array_split(P[cItr], numberOfStreams)
        PPP = [[]]
        for nos in range(numberOfStreams):
            PPP.append([])
        for nos in range(numberOfStreams):
            PPP[nos].append(PP[nos].tolist())
        for nos in range(numberOfStreams):
            fin = int(math.ceil(((math.log10(1/(eps/10.0)))**d)))
            lft = max(d+2,int(fin*math.log10(tagn)))
            rays = int(fin*math.log10(tagn))
            tagS, itrIndex = buildCoresetForLInfCircleUsingRays(PPP[nos][0], lft, eps, rays, fin, 1, circles[cItr])

            basicProb = 1
            exp = max(fin*math.log10(tagn),d+2)
            flag = 1
            tt = 0
            while (tt < exp) and (flag == 1):
                for seti in range(itrIndex):
                    if tt < ((1.1*exp)+1):
                        prob = basicProb/float(seti+1)
                        if len(tagS[0][int(itrIndex)]) == 0:
                            if itrIndex == 0:
                                flag = 0
                            else:
                                itrIndex = itrIndex - 1

                        for p in range(len(tagS[0][int(itrIndex-seti)])):
                            if probability(prob) == 1:
                                tt += 1
                                streamS[cItr].append(tagS[0][int(itrIndex-seti)][0])
                                tagS[0][int(itrIndex-seti)].remove(tagS[0][int(itrIndex-seti)][0])

        fin = int(math.ceil(((math.log10(1/eps))**d)))
        lft = max(d+2,int(fin*math.log10(n)))
        rays = int(fin*math.log10(n))
        tagS, itrIndex = buildCoresetForLInfCircleUsingRays(streamS[cItr], lft, eps, rays, fin, 1, circles[cItr])

        basicProb = 1
        exp = max(fin*math.log10(n),d+2)
        flag = 1
        tt = 0
        while (tt < exp) and (flag == 1):
            for seti in range(itrIndex):
                if tt < ((1.1*exp)+1):
                    prob = basicProb/float(seti+1)
                    if len(tagS[0][int(itrIndex)]) == 0:
                        if itrIndex == 0:
                            flag = 0
                        else:
                            itrIndex = itrIndex - 1

                    for p in range(len(tagS[0][int(itrIndex-seti)])):
                        if probability(prob) == 1:
                            tt += 1
                            S[cItr].append(tagS[0][int(itrIndex-seti)][0])
                            tagS[0][int(itrIndex-seti)].remove(tagS[0][int(itrIndex-seti)][0])

    return S, fin, 0, 0, fin, itrIndex, exp

def samplePoints(P, numberToSample):

    E = []
    n = len(P)
    prob = float(numberToSample/n)

    for p in P:
        if probability(prob) == 1:
            E.append(p)

    return E

def buildCoresetForLines(P, numberOfRecords, eps, k, m):

    for r in range(numberOfRecords):
        bestCost = 100000000

        bestLines = []
        for kk in range(k):
            bestLines.append([0,0])

        n = len(P[r])

        # Sample epsilon-coreset
        E = samplePoints(P[r], float(n/2))

        # for k = 4!!!
        #for i1 in range(n):
            #for i2 in range(i1, n):


    S = [[]]

    return S

def buildCoresetForL1Circle(P, numberOfRecords, eps, type, circles):

    d = len(P[0][0])
    n = len(P[0])

    S = [[]]
    for i in range(numberOfRecords):
        temp = []
        S.append(temp)

    for cItr in range(numberOfRecords):

        if circles == (0,0,0):
        #CInf = ShapesAlgorithms.LInfCircle(Pi, 1, "")
            CInf = ShapesAlgorithms.ApproximationLInfCircleCenter(P[cItr])
            if d == 3:
                circles[cItr] = (CInf[0],CInf[1],CInf[2],CInf[3])
            if d == 2:
                circles[cItr] = (CInf[0],CInf[1],CInf[2])

        fin = int(math.ceil((d/2.0)*((math.log10(1/eps))**2)))
        lft = max(d+2,int(fin*math.log10(n)))
        rays = int((d-1)*fin*math.log10(n))
        if rays == 0:
            rays = int((d)*fin*math.log10(n))
        tagS, itrIndex = buildCoresetForLInfCircleUsingRays(P[cItr], lft, eps, rays, fin, 1, circles[cItr])

        basicProb = 1
        exp = max(fin*math.log10(n),d+2)
        flag = 1
        while (len(S[cItr]) < exp) and (flag == 1):
            for seti in range(itrIndex):
                if len(S[cItr]) < ((1.1*exp)+1):
                    prob = basicProb/float(seti+1)
                    if len(tagS[0][int(itrIndex)]) == 0:
                        if itrIndex == 0:
                            flag = 0
                        else:
                            itrIndex = itrIndex - 1

                    for p in range(len(tagS[0][int(itrIndex-seti)])):
                        if probability(prob) == 1:
                            S[cItr].append(tagS[0][int(itrIndex-seti)][0])
                            tagS[0][int(itrIndex-seti)].remove(tagS[0][int(itrIndex-seti)][0])

    return S, fin, 0, 0, fin, itrIndex, exp

def buildMedianCoreset(P, eps):

    S = []

    tagS = buildBoundingLayers(P[0], eps)

    for i in range(len(tagS)):
        for j in range(len(tagS[i])):
            prob = 1.0/float(i+1)
            for p in tagS[i]:
                if probability(prob) == 1:
                    S.append((p,1/prob))

    return S

def buildBoundingLayers(P, eps):

    n = len(P)
    d = len(P[0])

    L = [[]]
    for i in range(len(P)):
        L.append([])

    layerIndex = 0
    while len(P) > 50:
        q = ShapesAlgorithms.findBoundingShape(P, d)

        layerP = np.zeros(d)
        layerD = np.zeros(d)
        for p in P:
            for i in range(d):
                Di = (p[i]-q[i])**2
                if Di > layerD[i]:
                    layerD[i] = Di
                    layerD[i] = p
                if Di < layerD[i+d]:
                    layerD[i+d] = Di
                    layerD[i+d] = p

        L[layerIndex].append(layerP)
        P.remove(layerP)

        layerIndex += 1

    return L

def buildCoresetForLInfCircleUsingRays(P, last, eps, numberOfRays, numberOfSamples, numberOfRecords, circle):

    n = len(P)
    d = len(P[0])

    S = [[[]]]
    for r in range(numberOfRecords):
        S.append([[]])

    for r in range(numberOfRecords):
        for i in range(len(P)):
            S[r].append([])

    itr = np.zeros(numberOfRecords)
    for r in range(numberOfRecords):
        n = len(P)

        # Find C infinity
        cf = circle #ShapesAlgorithms.ApproximationLInfCircleCenter(P) # change to lsq
        c = []
        for ddd in range(d+1):
            tt = np.random.random_sample()-0.5
            c.append(circle[ddd] + 0.3*circle[d]*tt)

        # Build projected set
        rayAngle = 360.0/numberOfRays
        tagP = [[]]
        for y in range(numberOfRays):
            tagP.append([])

        for i in range(n):

            # Find ray index
            if d == 3:
                ind = 0
                dx = P[i][0]-c[0]
                dy = P[i][1]-c[1]
                dz = P[i][2]-c[2]
                tagP[ind].append((P[i][0],P[i][1],P[i][2],(dx**2+dy**2+dz**2)))
            else:
                dx = P[i][0]-c[0]
                dy = P[i][1]-c[1]

                if dy < 0:
                    ind = int((360 + math.degrees(math.atan2(dy, dx)))/rayAngle)
                else:
                    ind = int((math.degrees(math.atan2(dy, dx)))/rayAngle)

                # Append P' with associate projected point in the correct index with distance (no sqrt)
                tagP[ind].append((P[i][0],P[i][1],(dx**2+dy**2)))

        for y in range(numberOfRays):
            sortedList = sorted(tagP[y], key=itemgetter(d))
            tagP[y] = sortedList

        # Build coreset S with convexes
        left = n
        while left > last:
            if left == 40:
                d = 2
            while len(S[r][int(itr[r])]) < numberOfSamples:
                selectedRay = 0
                selectedDist = -1
                for i in range(numberOfRays):
                    lst = len(tagP[i])
                    if lst > 1:
                        if (tagP[i][lst-1][d]-tagP[i][0][d]) > selectedDist:
                            selectedRay = i
                            selectedDist = tagP[i][lst-1][d]-tagP[i][0][d]
                    if lst == 1:
                        if (tagP[i][0][d]-c[d]) > selectedDist:
                            selectedRay = i
                            selectedDist = 1000000#tagP[i][0][2]-c[d]

                if len(tagP[selectedRay]) == 1:
                    S[r][int(itr[r])].append(tagP[selectedRay][0])
                    tagP[selectedRay].remove(tagP[selectedRay][0])
                    left -= 1
                if len(tagP[selectedRay]) > 1:
                    S[r][int(itr[r])].append(tagP[selectedRay][0])
                    S[r][int(itr[r])].append(tagP[selectedRay][len(tagP[selectedRay])-1])
                    tagP[selectedRay].remove(tagP[selectedRay][0])
                    tagP[selectedRay].remove(tagP[selectedRay][len(tagP[selectedRay])-1])
                    left -= 2

            itr[r] += 1
        while left > 0:
            for i in range(numberOfRays):
                for j in range(len(tagP[i])):
                    S[r][int(itr[r])].append(tagP[i][j])
                    #tagP[i].remove(tagP[i][j])
                    left -= 1

    return S, itr

def buildCoresetForLInfCircleBounded(P, CInf, coresetSize, numberOfRecords):

    S = [[]]
    for rec in range(numberOfRecords):
        temp = []
        S.append(temp)

    for rec in range(numberOfRecords):
        D = []
        for i in range(len(P[rec])):
            D.append((P[rec][i][0]-CInf[rec][0])**2 + (P[rec][i][1]-CInf[rec][1])**2)

        Pfar = ShapesAlgorithms.getFarest(P[rec], D, int(0.5*coresetSize))
        Pclose = ShapesAlgorithms.getClosest(P[rec], D, int(0.5*coresetSize))
        for f in range(int(0.5*coresetSize)):
            S[rec].append(Pfar[f])
            S[rec].append(Pclose[f])
        # S[rec].append(ShapesAlgorithms.getFarest(P[rec], D, int(0.5*coresetSize)))
        # S[rec].append(ShapesAlgorithms.getClosest(P[rec], D, int(0.5*coresetSize)))

    return S

def probability(p):

    randRange = 10000
    x = random.randint(0,randRange)

    d = 0
    if x < (p*randRange):
        d = 1

    return d

def l_infinity_2D_sphere_coreset(P,eps):
    # Optimal algorithm O(p^3*h^2)

    # Get points parameters
    #d = np.ndim(P)
    n = len(P)

    # Build external sphere
    H = []
    hull = ConvexHull(P)
    for v in hull.vertices:
        H.append(P[v])
    h = len(H)

    # Find minimal enclosing circle - O(p*h^3)


    C = []


    return C

def coresetForOneMedianPointOnLine(P, coresetSize):
    n = len(P)

    P.sort()

    statistics = [[0 for x in range(5)] for y in range(n)] # number, position, maxindex, sensitivity, totalsum

    for i in range(n):
        statistics[i][0] = P[i]
        statistics[i][1] = 0
        statistics[i][2] = 0
        statistics[i][3] = 0
        statistics[i][4] = 0.0

        for j in range(n):
            statistics[i][4] += abs(P[i]-P[j])
            if P[i] > P[j]:
                statistics[i][1] += 1

    median = np.average(P)# can be O(n) together with i: calc median -> for each point calc until getting down. it is O(2*n)
    for i in range(n):
        for j in range(n):
            sensitivity = abs(P[i]-P[j])/statistics[j][4]
            if sensitivity > statistics[i][3]:
                statistics[i][3] = sensitivity
                statistics[i][2] = statistics[j][1]


    totalSensitivity = 0
    sortedSelectedIndexex = []
    for i in range(n):
        totalSensitivity += statistics[i][3]
        sortedSelectedIndexex.append(statistics[i][2])

    sortedSelectedIndexex.sort()
    #plt.plot(sortedSelectedIndexex, color='green')

    scatter0 = []
    scatter1 = []
    scatter2 = []
    scatter3 = []
    for i in range(n):
        scatter0.append(statistics[i][0])
        scatter1.append(statistics[i][1])
        scatter2.append(statistics[i][2])
        scatter3.append(statistics[i][3])

    #plt.scatter(scatter0[:], scatter3[:], color='green')

    coreset = np.random.choice(P, coresetSize, scatter3)

    return coreset, totalSensitivity

def meanCircleFitting(samples,norm):
    circleX = 0
    circleY = 0
    circleR = 0

    numberOfSamples = np.size(samples)

    b = np.min(samples)
    for i in range(numberOfSamples):
        samples[i] = b + (samples[i]-b) * norm

    # Iterate geometric circle calculation
    bestError = 100000
    iMin = 0
    iMax = 5
    jMin = numberOfSamples/2 - 5
    jMax = numberOfSamples/2 + 5
    kMin = numberOfSamples - 5
    kMax = numberOfSamples
    for i in range(iMin,iMax):
        for j in range(jMin,jMax):
            for k in range(kMin,kMax):
                # Calculate circle properties
                Cx = ((samples[i]*samples[i]+i*i)*(samples[j]-samples[k])+(samples[j]*samples[j]+j*j)*(samples[k]-samples[i])+(samples[k]*samples[k]+k*k)*(samples[i]-samples[j]))/(2*(i*(samples[j]-samples[k])-samples[i]*(j-k)+j*samples[k]-k*samples[j]))
                Cy = ((samples[i]*samples[i]+i*i)*(k-j)+(samples[j]*samples[j]+j*j)*(i-k)+(samples[k]*samples[k]+k*k)*(j-i))/(2*(i*(samples[j]-samples[k])-samples[i]*(j-k)+j*samples[k]-k*samples[j]))

                Cr = 0.0
                error = 0
                for r in range(numberOfSamples):
                    Cr += np.sqrt((Cx-r)*(Cx-r)+(samples[r]-Cy)*(samples[r]-Cy))
                    error += ((Cx-r) * (Cx-r) + (Cy-samples[r]) * (Cy-samples[r]))
                Cr = Cr / numberOfSamples
                error = error / numberOfSamples

                if bestError > error:
                    bestError = error
                    circleX = Cx
                    circleY = Cy
                    circleR = Cr

    return circleX,circleY,circleR

def getCircleByThreePoints(i,j,k):

    c = []
    c.append(((i[1]*i[1]+i[0]*i[0])*(j[1]-k[1])+(j[1]*j[1]+j[0]*j[0])*(k[1]-i[1])+(k[1]*k[1]+k[0]*k[0])*(i[1]-j[1]))/(2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1])))
    c.append(((i[1]*i[1]+i[0]*i[0])*(k[0]-j[0])+(j[1]*j[1]+j[0]*j[0])*(i[0]-k[0])+(k[1]*k[1]+k[0]*k[0])*(j[0]-i[0]))/(2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1])))
    c.append(np.sqrt((c[0]-i[0])*(c[0]-i[0])+(c[1]-i[1])*(c[1]-i[1])))

    # cX = []
    # cY = []
    # for ii in range(360):
    #     cX.append((c[2] * np.sin(ii)) + c[0])
    #     cY.append((c[2] * np.cos(ii)) + c[1])
    # plt.scatter(cX[:],cY[:], color='red')
    # plt.scatter(i[0],i[1], color='blue')
    # plt.scatter(j[0],j[1], color='blue')
    # plt.scatter(k[0],k[1], color='blue')
    # plt.scatter(c[0],c[1], color='green')

    return c

def summationDistanceFromCircle(c,P):
    totalDist = 0
    n = len(P)

    for i in range (n):
        dx = P[i][0]-c[0]
        dy = P[i][1]-c[1]

        dist = np.sqrt(dx*dx+dy*dy)

        totalDist += np.abs(dist-c[2])

    return totalDist
