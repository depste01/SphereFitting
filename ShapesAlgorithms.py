# import sys
# import random
# import math
# import array
# import numpy
# import sys
# import random
# import math
# import array
# import os
# from mpl_toolkits.mplot3d import Axes3D
# import skipy
# from skipy.signal import medfilt2d
#import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import medfilt2d
from scipy.spatial import ConvexHull
import numpy as np
import math
from scipy.optimize import leastsq
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import optimize
import MapleUtils

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def getLineWithPcs(P):

    X = n.asarray(P[0])
    #rng = np.random.RandomState(1)
    #X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal');

    pca = PCA(n_components=2)
    pca.fit(X)

    print(pca.components_)

    print(pca.explained_variance_)
    plt.plot([0,pca.components_[0][0]], [0,pca.components_[0][1]], 'ro-')

    # for length, vector in zip(pca.explained_variance_, pca.components_):
    #     v = vector * 3 * np.sqrt(length)
    #     draw_vector(pca.mean_, pca.mean_ + v)

    return

def getOpencvLineSegments(fileName):
    img = cv2.imread(fileName,0)

    lsd = cv2.createLineSegmentDetector(0)

    #Detect lines in the image
    lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
    print(lines[0][0][0])

    ver_lines = []

    for line in lines:
        angletan = math.degrees(math.atan2((round(line[0][3],2) - round(line[0][1],2)), (round(line[0][2],2) - round(line[0][0],2))))

        #if(angletan > 10 and angletan < 80):
        ver_lines.append(line)

    #Draw detected lines in the image
    drawn_img = lsd.drawSegments(img,np.array(ver_lines))

    #Show image
    cv2.imshow("LSD",drawn_img )
    cv2.waitKey(0)

def getMedianLines(P, expectedNumberOfLines):
    L = []
    n = len(P)
    U, s, V = np.linalg.svd(P)

    return L

def scipyCircleDetection(P):
    c = [0,0,0]
    n = len(P)

    #scipy.optimize.
    x = np.array(P)

    funcLine=lambda tpl,x : tpl[0]*x+tpl[1]
    funcDist=lambda tpl,x : tpl[0]*x**2+tpl[1]*x+tpl[2]

    return c


def generate2DNoisySphere(n,d,c,e):
    P = []
    for i in range(n):
        #Noisy vector
        r = c[d]+c[d]*e*(0.5-np.random.random_sample())
        rr = r/np.sqrt(d)
        p = np.repeat(rr,d)

        # Rotation Mx
        deg = 40*np.random.random_sample()
        Mx = np.array([[math.cos(deg), -math.sin(deg)],[math.sin(deg),math.cos(deg)]])

        p = Mx.dot(p)

        p[0] += c[0]
        p[1] += c[1]

        P.append(p)

    printNoisyCircle = 0
    if printNoisyCircle == 1:
        fig = plt.figure(26)
        for i in range (n):
            plt.scatter(P[i][0], P[i][1], color='orange')

        plt.show()

    return P

def generate3DNoisySphere(n,d,c,e):
    P = []
    for i in range(n):
        #Noisy vector
        r = c[d]+c[d]*e*(0.5-np.random.random_sample())
        rr = r/np.sqrt(d)
        p = np.repeat(rr,d)

        # Rotation Mx
        deg = 40*np.random.random_sample()
        a3dx = np.array([[1, 0,0],[0,math.cos(deg), -math.sin(deg)],[0, math.sin(deg),math.cos(deg)]])
        deg = 64*np.random.random_sample()
        a3dy = np.array([[math.cos(deg), 0,math.sin(deg)],[0,1,0],[-math.sin(deg),0,math.cos(deg)]])
        deg = 13*np.random.random_sample()
        a3dz = np.array([[math.cos(deg), -math.sin(deg),0],[math.sin(deg),math.cos(deg), 0],[0, 0,1]])

        p = a3dx.dot(p)
        p = a3dy.dot(p)
        p = a3dz.dot(p)

        p[0] += c[0]
        p[1] += c[1]
        p[2] += c[2]

        P.append(p)

    printNoisyCircle = 0
    if printNoisyCircle == 1:
        fig = plt.figure(34)
        ax = fig.add_subplot(111, projection='3d')
        for i in range (n):
            ax.scatter(P[i][0], P[i][1], P[i][2], color='orange')

        plt.show()

    return P

#########################       L_infinity_2D_sphere                        ###############################
def l_inf_circle(P):

    n = len(P)
    sv = scipy.spatial.Voronoi(P)

    circles = []
    numofcircles = 0
    for vor in sv.vertices:
        widthArray = []
        for p in P:
            widthArray.append((vor[0]-p[0])*(vor[0]-p[0])+(vor[1]-p[1])*(vor[1]-p[1]))
        widthArray.sort()
        circles.append(scipy.sqrt(widthArray[n-1]-widthArray[0]))
        numofcircles += 1

    idx = 0
    for ggg in range(numofcircles):
        if circles[ggg]<circles[idx]:
            idx = ggg


    cx = sv.vertices[idx][0]
    cy = sv.vertices[idx][1]
    cr = 0
    for p in P:
        cr += scipy.sqrt((cx-p[0])*(cx-p[0])+(cy-p[1])*(cy-p[1]))
    cr /= n

    return cx,cy,cr

def LInfCircle(P):

    numberOfRecords = len(P)

    CInf = []
    for rec in range(numberOfRecords):
        n = len(P[rec])
        sv = scipy.spatial.Voronoi(P[rec])

        circles = []
        numofcircles = 0
        for vor in sv.vertices:
            widthArray = []
            for p in P[rec]:
                widthArray.append((vor[0]-p[0])*(vor[0]-p[0])+(vor[1]-p[1])*(vor[1]-p[1]))

            widthArray.sort()
            circles.append(scipy.sqrt(widthArray[n-1]-widthArray[0]))
            numofcircles += 1

        idx = 0
        for ggg in range(numofcircles):
            if circles[ggg]<circles[idx]:
                idx = ggg


        cx = sv.vertices[idx][0]
        cy = sv.vertices[idx][1]
        cr = 10000000
        cR = 0
        for p in P[rec]:
            dist = scipy.sqrt((cx-p[0])*(cx-p[0])+(cy-p[1])*(cy-p[1]))
            if cr > dist:
                cr = dist
            if cR < dist:
                cR = dist

        CInf.append((cx,cy,0.5*(cr+cR)))

    # # Save result to file
    # if CInfFileName != "":
    #     file = open(CInfFileName,"w")
    #     for itr in range(numberOfRecords):
    #         file.write("%.10f"%CInf[itr][0])
    #         file.write(",%.10f"%CInf[itr][1])
    #         r = 0.5 * (CInf[itr][3] + CInf[itr][2])
    #         file.write(",%.10f"%r)
    #         file.write("\n")
    #
    #     file.close()

    return CInf

def l_infinity_2D_sphere(P):

    n = len(P)

    # Build external sphere
    H = []
    hull = ConvexHull(P)
    for v in hull.vertices:
        H.append(P[v])
    h = len(H)

    # Set NP <- P/H
    NP = []
    for nn in range(n):
        b = 0
        for hh in range (h):
            if P[nn] == H[hh]:
                b = 1
                #break
        if b == 0:
            NP.append(P[nn])
    np = len(NP)

    # Initialize best fit
    cx = 0
    cy = 0
    csr = 0
    csR = 100000

    hullLine = []

    # Pair of not external
    for j in range((np)):
        for k in range(j+1,(np)):
            if NP[j][1] != NP[k][1]:
                tt = 1.0*(NP[k][0]-NP[j][0])/(NP[j][1]-NP[k][1])
                hullLine.append([tt,0.5*(NP[k][1]+NP[j][1])-0.5*tt*(NP[k][0]+NP[j][0])])

    for i in range((h)):
        for j in range(i+1,(h)):
            if H[i][1] != H[j][1]:
                line1a = (H[j][0]-H[i][0])/(H[i][1]-H[j][1])
                line1b = 0.5*(H[j][1]+H[i][1])-0.5*line1a*(H[j][0]+H[i][0])

            for hl in hullLine:
                if line1a != hl[0]:
                    x = (hl[1]-line1b)/(line1a-hl[0])
                    y = line1a*x+line1b

                    r = 1000000
                    R = 0
                    for p in (P):
                        dist = (x-p[0])*(x-p[0])+(y-p[1])*(y-p[1])
                        if dist < r:
                            r = dist
                        if dist > R:
                            R = dist

                    rt = scipy.sqrt(r)
                    Rt = scipy.sqrt(R)
                    if (Rt-rt) < (csR-csr):
                        csR = Rt
                        csr = rt
                        cx = x
                        cy = y

    cr = (0.5*(csR+csr))

    return [cx,cy,cr]

def getFarest(P, D, k):

    S = []

    n = len(P)

    SP = []
    SD = []
    for i in range(k):
        SP.append(i)
        SD.append(D[i])

    for i in range (k,n):
        testind = i
        testval = D[i]
        for s in range(k):
            if testval > SD[s]:
                t = SP[s]
                td = SD[s]
                SP[s] = testind
                SD[s] = testval
                testind = t
                testval = td

    for i in range(k):
        S.append(P[SP[i]])

    return S

def getClosest(P, D, k):

    S = []

    n = len(P)

    SP = []
    SD = []
    for i in range(k):
        SP.append(i)
        SD.append(D[i])

    for i in range (k,n):
        testind = i
        testval = D[i]
        for s in range(k):
            if testval < SD[s]:
                t = SP[s]
                td = SD[s]
                SP[s] = testind
                SD[s] = testval
                testind = t
                testval = td

    for i in range(k):
        S.append(P[SP[i]])

    return S

def squarDistance(p, l):
    D = 0

    # l: ax+by+c=0 ; y=ax+b -> ...
    a=(l[0][1]-l[1][1])/(l[0][0]-l[1][0]) # a=dy/dx
    b=1.0
    c=l[0][1]-a*l[0][0] # c=y-ax


    #D = ((a*p[0]+b*p[1]+c)**)/(a**+b**) = (ap0+p1+c)^2/(a^2+1)
    return 1.0*((a*p[0]-p[1]+l[0][1]-a*l[0][0])*(a*p[0]-p[1]+l[0][1]-a*l[0][0]))/(a*a+1)

def ApproximationLInfCircleCenter(P):

    n = len(P)
    d = len(P[0])
    c = np.zeros(d+1)

    for p in P:
        for i in range(d):
            c[i] += p[i]

    for i in range(d):
        c[i] = c[i]/n

    # r = 10000000
    # R = 0
    # for p in P:
    #     dst = 0
    #     for i in range(d):
    #         dst += (c[i]-p[i])**2
    #     if dst < r:
    #         r = dst
    #     if dst > R:
    #         R = dst
    # c[d] = (0.5*(scipy.sqrt(r)+scipy.sqrt(R)))

    R = 0
    for p in P:
        dst = 0
        for i in range(d):
            dst += (c[i]-p[i])**2
        R += scipy.sqrt(dst)

    c[d] = R/n

    return c

#########################       L_infinity_2D_sphere                        ###############################

def meanLinesFitting(samples,k):
    L = [[]]
    return L
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
    if ((2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1]))) != 0:
        c.append(((i[1]*i[1]+i[0]*i[0])*(j[1]-k[1])+(j[1]*j[1]+j[0]*j[0])*(k[1]-i[1])+(k[1]*k[1]+k[0]*k[0])*(i[1]-j[1]))/(2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1])))
        c.append(((i[1]*i[1]+i[0]*i[0])*(k[0]-j[0])+(j[1]*j[1]+j[0]*j[0])*(i[0]-k[0])+(k[1]*k[1]+k[0]*k[0])*(j[0]-i[0]))/(2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1])))
        c.append(np.sqrt((c[0]-i[0])*(c[0]-i[0])+(c[1]-i[1])*(c[1]-i[1])))
    else:
        c.append(0)
        c.append(0)
        c.append(0)

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

def getCircleBySetOfThreePoints(P):

    i = P[0]
    j = P[1]
    k = P[2]

    c = []
    if ((2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1]))) != 0:
        c.append(1.0*((i[1]*i[1]+i[0]*i[0])*(j[1]-k[1])+(j[1]*j[1]+j[0]*j[0])*(k[1]-i[1])+(k[1]*k[1]+k[0]*k[0])*(i[1]-j[1]))/(2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1])))
        c.append(1.0*((i[1]*i[1]+i[0]*i[0])*(k[0]-j[0])+(j[1]*j[1]+j[0]*j[0])*(i[0]-k[0])+(k[1]*k[1]+k[0]*k[0])*(j[0]-i[0]))/(2*(i[0]*(j[1]-k[1])-i[1]*(j[0]-k[0])+j[0]*k[1]-k[0]*j[1])))
        c.append(np.sqrt((c[0]-i[0])*(c[0]-i[0])+(c[1]-i[1])*(c[1]-i[1])))
    else:
        c.append(0)
        c.append(0)
        c.append(0)

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

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    x = 0
    y = 0
    div = det(xdiff, ydiff)
    if div != 0:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        return x, y

def LineIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    a = ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )

    px = 0
    py = 0
    if a > 0:
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )

    if px < 0:
        px = 0
    if py < 0:
        py = 0

    return [px, py]

def split(u, v, points):
    # return points on left side of UV
    return [p for p in points if np.cross(p - u, v - u) < 0]

def extend(u, v, points):
    if not points:
        return []

    # find furthest point W, and split search to WV, UW
    w = min(points, key=lambda p: np.cross(p - u, v - u))
    p1, p2 = split(w, v, points), split(u, w, points)
    return extend(w, v, p1) + [w] + extend(u, w, p2)

def convex_hull(points):
    # find two hull points, U, V, and split to left and right search
    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    left, right = split(u, v, points), split(v, u, points)

    # find convex hull on each side
    return [v] + extend(u, v, left) + [u] + extend(v, u, right) + [v]

import numpy as n, pylab as p, time

def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = n.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += n.pi
    return res


def _draw_triangle(p1, p2, p3, **kwargs):
    tmp = n.vstack((p1,p2,p3))
    x,y = [x[0] for x in zip(tmp.transpose())]
    p.fill(x,y, **kwargs)
    #time.sleep(0.2)


def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return n.linalg.norm(n.cross((p2 - p1), (p3 - p1)))/2.


def convex_hulla(points, graphic=True, smidgen=0.0075):
    '''Calculate subset of points that make a convex hull around points

Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

:Parameters:
    points : ndarray (2 x m)
        array of points for which to find hull
    graphic : bool
        use pylab to show progress?
    smidgen : float
        offset for graphic number labels - useful values depend on your data range

:Returns:
    hull_points : ndarray (2 x n)
        convex hull surrounding points
'''
    if graphic:
        p.clf()
        p.plot(points[0], points[1], 'ro')
    n_pts = points.shape[1]
    assert(n_pts > 5)
    centre = points.mean(1)
    if graphic: p.plot((centre[0],),(centre[1],),'bo')
    angles = n.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:,angles.argsort()]
    if graphic:
        for i in xrange(n_pts):
            p.text(pts_ord[0,i] + smidgen, pts_ord[1,i] + smidgen, \
                   '%d' % i)
    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        if graphic: p.gca().patches = []
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i],     pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], \
                                   pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i],     pts[(i + 2) % n_pts])
            if graphic:
                _draw_triangle(centre, pts[i], pts[(i + 1) % n_pts], \
                               facecolor='blue', alpha = 0.2)
                _draw_triangle(centre, pts[(i + 1) % n_pts], \
                               pts[(i + 2) % n_pts], \
                               facecolor='green', alpha = 0.2)
                _draw_triangle(centre, pts[i], pts[(i + 2) % n_pts], \
                               facecolor='red', alpha = 0.2)
            if Aij + Ajk < Aik:
                if graphic: p.plot((pts[i + 1][0],),(pts[i + 1][1],),'go')
                del pts[i+1]
            i += 1
            n_pts = len(pts)
        k += 1
    return n.asarray(pts)

def findKLines(P, k, d, numberOfRecords):
    L = [[0 for i in range(2)] for j in range(k)]

    #MapleUtils.buildMapleCommandsForKLines(samplesFileName, mapleCommandFileName, mapleResultsFileName, numberOfRecords, 1, d)
    return L

def squareDistFromCircle(c, p):

    d = len(c)-1
    dist = 0
    for i in range(d):
        dist += (c[i]-p[i])**2
    dist = abs(dist-c[d]**2)

    return dist

def opencvCircle(imageFileName):

    res = []

    img = cv2.imread(imageFileName,0)
    drwimg = cv2.imread(imageFileName,0)
    #img = cv2.medianBlur(img,5)
    cimg = img#cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,5,200,param1=50,param2=30,minRadius=50,maxRadius=5000)

    circles = np.uint16(np.around(circles))

    for c in circles[0,:]:
        # draw the outer circle
        # cv2.circle(drwimg,(i[0],i[1]),i[2],(255,255,0),3)
        # imsave('opencv1'+imageFileName, drwimg)

        # draw the center of the circle

        # imtoshow = cv2.imread(imageFileName)
        # cv2.circle(imtoshow,(int(float(c[0])),int(float(c[1]))), int(float(c[2])), (255,0,0), 2)
        # imsave('opencv'+imageFileName, imtoshow)

        return c

        break
        cv2.circle(cimg,(i[0],i[1]),2,(0,2,255),3)


    #cv2.imshow('detected circles',cimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return

#import trig_items
#import numpy as np
#from trig_items import *
from numpy import *
from matplotlib import pyplot as p
from scipy import optimize
x = r_[0,0,0,0]
y = r_[0,0,0,0]
z = r_[0,0,0,0]

def calc_R(xc, yc, zc):
    """ calculate the distance of each 3D points from the center (xc, yc, zc) """
    return sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)

def func(c):
    """ calculate the algebraic distance between the 3D points and the mean circle centered at c=(xc, yc, zc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

def getBallByFourPoints(i,j,k,l):

    global x
    global y
    global z

    x = r_[i[0],j[0],k[0],l[0]]
    y = r_[i[1],j[1],k[1],l[1]]
    z = r_[i[2],j[2],k[2],l[2]]

    xm = mean(x)
    ym = mean(y)
    zm = mean(z)

    center_estimate = xm, ym, zm
    center, ier = optimize.leastsq(func, center_estimate)

    xc, yc, zc = center
    Ri       = calc_R(xc, yc, zc)
    R        = Ri.mean()
    residu   = sum((Ri - R)**2)

    return xc, yc, zc, R

def calc_R2D(xc, yc):
    return sqrt((x - xc) ** 2 + (y - yc) ** 2)

def func2D(c):
    Ri = calc_R2D(*c)
    return Ri - Ri.mean()

def getSphereByThreePoints(i,j,k):

    global x
    global y

    x = r_[i[0],j[0],k[0]]
    y = r_[i[1],j[1],k[1]]

    xm = mean(x)
    ym = mean(y)

    center_estimate = xm, ym
    center, ier = optimize.leastsq(func2D, center_estimate)

    xc, yc = center
    Ri       = calc_R2D(xc, yc)
    R        = Ri.mean()
    residu   = sum((Ri - R)**2)

    return xc, yc, R
