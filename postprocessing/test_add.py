import numpy as np
from plyfile import PlyData
import math
import cv2 

_ZEROANGLES = 1E-10

def ADD(modelPoints,Rgt,Rest,Tgt,Test):

    gts = Rgt[None,:].dot(modelPoints.T) + Tgt.reshape((3,1))
    ests = Rest[None,:].dot(modelPoints.T) + Test.reshape((3,1))
    add = np.linalg.norm(ests.T-gts.T,axis=1).mean()
   
    return add

def load3DModel(modelpath):
    data = PlyData.read(modelpath)

    x = np.array(data['vertex']['x'])
    y = np.array(data['vertex']['y'])
    z = np.array(data['vertex']['z'])
    points = np.column_stack((x, y, z))

    return points
def _clamp(minn, maxn, number):
    if number > maxn:
        return maxn
    elif number < minn:
        return minn
    else: return number 

# convert 3D model points to cm if they are in mm
def to_cm(points):
    return points/10

def quat2vec(q):  # q[4] , r[3]

    r = np.zeros(3)
    mag = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    if mag < 0.99999 or mag > 1.00001:  # normalize
        mag = 1.0/math.sqrt(mag)
        q[0] *= mag
        q[1] *= mag
        q[2] *= mag
        q[3] *= mag

    th = math.acos(_clamp(-1.0, 1.0, q[0]))
    s = math.sin(th)
    if abs(s) > _ZEROANGLES:
        th = 2.0*th
        s = th/s
        r[0] = q[1]*s
        r[1] = q[2]*s
        r[2] = q[3]*s

    else:
        r[0] = r[1] = r[2] = 0.0  # s close to zero

    return r

def calculate_bbox(points): #using EPOS coordinate system

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    centroid = [np.sum(x)/x.size, np.sum(y)/y.size, np.sum(z)/z.size]

    return np.array([ [xmax, ymax, zmin],[xmax, ymax, zmax],[xmin, ymax, zmax], [xmin, ymax, zmin], [xmax, ymin, zmin], [xmax,ymin, zmax], [xmin, ymin, zmax], [xmin, ymin, zmin], centroid])

modelpath = '/home/foto1/linux_part/athena/isaac/dope_fork/3d_models/phase_II_objects/models12/obj_000001_opencv_axes.ply'
points = load3DModel(modelpath)
tgt = np.array([1.175125503540039, -13.253958129882812, 69.55150756835938])
rgt = np.array([0.49455928802490234, -0.8033257722854614, -0.331781268119812, -0.7654682397842407, -0.5833882093429565, 0.27150821685791016, -0.4116668105125427, 0.11969111859798431, -0.9034403562545776])
rgt = rgt.reshape((3,3))
test = np.array([1.95624924885312, -7.897506353129, 36.93330154001428])
quat_est = np.array([-0.11302115721344991, -0.7192811170493626, 0.42527757947652606, 0.5375870841760809])
rvec = quat2vec(quat_est)
print('RVEC',rvec)
[rest, _] = cv2.Rodrigues(rvec)
print('RGT', rgt)
print('REST', rest.shape)
print(rest)
# for r in rest:
#     print(r.shape)

#print(_clamp(-0.1, 0.1, -1.0))
pointscm = to_cm(points)
add = ADD(pointscm, rgt, rest, tgt, test)
print(add)
bbox3d = calculate_bbox(points)
print(bbox3d)
print(points[0,:])