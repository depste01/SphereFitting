from scipy import optimize
from numpy import *

class circleLeastsquareClass:
    x = r_[2]
    y = r_[1]

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calc_R(self,xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x-xc)**2 + (self.y-yc)**2)

    def f_3(self,c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

def findLeastSquareCircle(P):

    C = []
    for i in range(len(P)):
        xl = []
        yl = []
        for p in P[i]:
            xl.append(p[0])
            yl.append(p[1])
        x = r_[xl]
        y = r_[yl]

        x_m = mean(x)
        y_m = mean(y)

        center_estimate = x_m,y_m
        lstSqr = circleLeastsquareClass(x,y)
        center_2, ier = optimize.leastsq(lstSqr.f_3, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = lstSqr.calc_R(*center_2)
        R_2        = Ri_2.mean()

        C.append([xc_2,yc_2,R_2])

    return C
