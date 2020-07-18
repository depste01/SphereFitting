import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
import math
from math import cos, sin, sqrt
from sympy import Point, Point3D, Line, Line3D, Plane
from sympy.abc import x
import ellipses as el
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

data = el.make_test_ellipse()

lsqe = el.LSqEllipse()
lsqe.fit(data)
center, width, height, phi = lsqe.parameters()

plt.close('all')
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.axis('equal')
ax.plot(data[0], data[1], 'ro', label='test data', zorder=1)

ellipse = Ellipse(xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
               edgecolor='b', fc='None', lw=2, label='Fit', zorder = 2)
ax.add_patch(ellipse)

plt.legend()
plt.show()
#
# def GetIntersectionPoints(L1,L2):
#     SMALL_NUM = 0.00000001
#     u = (L1[1][0]-L1[0][0],L1[1][1]-L1[0][1],L1[1][2]-L1[0][2])
#     v = (L2[1][0]-L2[0][0],L2[1][1]-L2[0][1],L2[1][2]-L2[0][2])
#     w = (L1[0][0]-L2[0][0],L1[0][1]-L2[0][1],L1[0][2]-L2[0][2])
#     a = dot(u, u) # always >= 0
#     b = dot(u, v)
#     c = dot(v, v) # always >= 0
#     d = dot(u, w)
#     e = dot(v, w)
#     D = a*c - b*b # always >= 0
#
#     if D < SMALL_NUM:
#         sc = 0
#         if b > c:
#             tc = d/b
#         else:
#             tc = e/c
#     else:
#         sc = (b*e - c*d) / D
#         tc = (a*e - b*d) / D
#
#     p1 = w[0] + (sc * u[0]) - (tc * v[0])
#     p2 = w[1] + (sc * u[1]) - (tc * v[1])
#     p3 = w[2] + (sc * u[2]) - (tc * v[2])
#
#
#
#     dP = [p1,p2,p3]
#
#     Ps_x = L1[0][0]+sc*u[0]
#     Ps_y = L1[0][1]+sc*u[1]
#     Ps_z = L1[0][2]+sc*u[2]
#
#     Qt_x = L2[0][0]+tc*v[0]
#     Qt_y = L2[0][1]+tc*v[1]
#     Qt_z = L2[0][2]+tc*v[2]
#
#     return ([Ps_x,Ps_y,Ps_z],[Qt_x,Qt_y,Qt_z])
#
# def dot(P1,P2):
#     return P1[0]*P2[0] + P1[1]*P2[1] + P1[2]*P2[2]
#
# def plot_lines(lines,ax):
#     #fig = plt.figure()
#     #ax = fig.gca(projection='3d')
#     for i in range(lines.__len__()):
#         ax.scatter(lines[i][0][0], lines[i][0][1], lines[i][0][2], zdir='z', c='r')
#         ax.scatter(lines[i][1][0], lines[i][1][1], lines[i][1][2], zdir='z', c='r')
#         ax.plot(xs=[lines[i][0][0], lines[i][1][0]], ys=[lines[i][0][1], lines[i][1][1]],zs=[lines[i][0][2], lines[i][1][2]])
#
#     #ax.legend()
#     #ax.set_xlim3d(-50, 50)
#     #ax.set_ylim3d(-50, 50)
#     #ax.set_zlim3d(-50, 50)
#
#     #plt.show()
#
# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     """
#     axis = np.asarray(axis)
#     axis = axis/math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta/2.0)
#     b, c, d = -axis*math.sin(theta/2.0)
#     aa, bb, cc, dd = a*a, b*b, c*c, d*d
#     bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
#     return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
#                      [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
#                      [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
#
# opt_line = ([0,0,0],np.random.randint(-50,50,3))
# line0= (np.random.randint(-50,50,3),np.random.randint(-50,50,3))
# line1= (np.random.randint(-50,50,3),np.random.randint(-50,50,3))
# lines=[]
# points=[]
#
# lines.append(line1)
# opt_line = ([0,0,0],line0[1])
# lines.append(opt_line)
#
#
#
# a = Plane(Point3D(0, 0, 0), Point3D(line0[0][0], line0[0][1], line0[0][2]),
#           Point3D(line0[1][0], line0[1][1], line0[1][2]))
#
# for j in range(360):
#     mat = rotation_matrix(a.normal_vector, (math.pi / 180))
#     opt_line = np.dot(mat, opt_line[1])
#     opt_line = ([0, 0, 0], [opt_line[0], opt_line[1], opt_line[2]])
#     # lines.append(opt_line)
#     #lines.append(opt_line)
#     p1, p2 = GetIntersectionPoints(opt_line, line1)
#     points.append(p1)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(points.__len__()):
#     ax.scatter(points[i][0], points[i][1], points[i][2], zdir='z', c='b')
#
# for i in range(lines.__len__()):
#     ax.scatter(lines[i][0][0], lines[i][0][1], lines[i][0][2], zdir='z', c='r')
#     ax.scatter(lines[i][1][0], lines[i][1][1], lines[i][1][2], zdir='z', c='r')
#     ax.plot(xs=[lines[i][0][0], lines[i][1][0]], ys=[lines[i][0][1], lines[i][1][1]],
#             zs=[lines[i][0][2], lines[i][1][2]])
#
# ax.legend()
# ax.set_xlim3d(-50, 50)
# ax.set_ylim3d(-50, 50)
# ax.set_zlim3d(-50, 50)
#
# plt.show()
#
# x=5
