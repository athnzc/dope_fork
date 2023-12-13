import numpy as np
import cv2 as cv
from plyfile import PlyData
import matplotlib.pyplot as plt
import math 
from pyrr import Quaternion
_ZEROANGLES = 1E-10

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
# convert quaternion from xyzw to wxyz
def quat2wxyz(q):
    qwxyz = np.array([q[3], q[0], q[1], q[2]])
    return qwxyz
def rotate3D_X(angle_deg):
    
    angle_deg = np.deg2rad(angle_deg)
    R_X = np.array([[1,0,0],
                    [0,np.cos(angle_deg),-np.sin(angle_deg)],
                    [0,np.sin(angle_deg),np.cos(angle_deg)]]) 
    return R_X

def rotate3D_Z(angle_deg):
    angle_deg = np.deg2rad(angle_deg)
    R_Z = np.array([[np.cos(angle_deg),-np.sin(angle_deg),0],
                    [np.sin(angle_deg),np.cos(angle_deg),0],
                    [0,0,1]]) 
    return R_Z

def rotate3D_Y(angle_deg):
    
    angle_deg = np.deg2rad(angle_deg)
    R_Y = np.array([[np.cos(angle_deg),0,np.sin(angle_deg)],
                    [0,1,0],
                    [-np.sin(angle_deg),0,np.cos(angle_deg)]])
    return R_Y

def convert_rvec_to_quaternion(rvec):
    '''Convert rvec (which is log quaternion) to quaternion'''
    theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    return Quaternion.from_axis_rotation(raxis, theta)

modelpath = "/home/foto1/linux_part/athena/isaac/dope_fork/3d_models/phase_II_objects/models12/obj_000001.ply"
imgpath = "/home/foto1/linux_part/athena/isaac/dope_fork/scripts/train2/test2/1.png"

points_2D_gt =   np.array([[798,179],
[716,220 ],
[610,234 ],
[668,211 ],
[902,508 ],
[831,562 ],
[709,477 ],
[755,446 ]
]).reshape(-1,2).astype(float)

points_2D_gt = np.array([[798.41222813, 180.48569834],
 [718.29573701, 221.89713044],
 [610.40585197, 237.37979401],
 [668.05564155, 210.00376333],
 [902.27809322, 508.40802816],
 [832.44488946, 558.86253877],
 [708.83133308, 477.27681864],
 [754.11171083, 447.8789133 ],
 [763.34930006, 365.53389506]]).reshape(-1,2).astype(float)

K = np.array([634.364, 0.0, 637.801, 0.0, 633.635, 364.958, 0.0, 0.0, 1.0]).reshape(3,3) #s/ 1.40625
print(K)


# points_2D = np.array([
#                 [
#                     846.6471220611728,
#                     256.02610771507204
#                 ],
#                 [
#                     607.3932571077772,
#                     347.0158731405768
#                 ],
#                 [
#                     366.27355860884825,
#                     311.39466929588235
#                 ],
#                 [
#                     963.621772082163,
#                     55.39353356684933
#                 ],
#                 [
#                     891.3192062649159,
#                     371.4656532586903
#                 ],
#                 [
#                     644.9242337156661,
#                     454.85668201951296
#                 ],
#                 [
#                     449.50123955775166,
#                     575.2770201245858
#                 ],
#                 [
#                     1094.3817021781324,
#                     369.8822282215447
#                 ],
#                 [
#                     726.499364030926,
#                     352.6084545678897
#                 ]
#             ]).reshape(-1,2)

# points_3D = np.array([[  7.949985,  12.426975,   3.70979 ],
#  [ -7.949985,  12.426975,   3.70979 ],
#  [ -7.949985, -12.426975,   3.70979 ],
#  [  7.949985, -12.426975,   3.70979 ],
#  [  7.949985,  12.426975,  -3.70979 ],
#  [ -7.949985,  12.426975,  -3.70979 ],
#  [ -7.949985, -12.426975,  -3.70979 ],
#  [  7.949985, -12.426975,  -3.70979 ]]).reshape(-1,3) * 10

# points_3D_ = np.array(
# [[  7.949985, -12.426975 ,  3.70979 ],
#  [ -7.949985, -12.426975  , 3.70979 ],
#  [ -7.949985,  12.426975 ,  3.70979 ],
#  [  7.949985 , 12.426975  , 3.70979 ],
#  [  7.949985 ,-12.426975 , -3.70979 ],
#  [ -7.949985 ,-12.426975  ,-3.70979 ],
#  [ -7.949985 , 12.426975 , -3.70979 ],
#  [  7.949985 , 12.426975 , -3.70979 ]]).reshape(-1,3) * 10

# points_3D_ours = np.array([[  79.49984741  ,124.2697525   ,-37.09790039],
#  [  79.49984741 , 124.2697525   , 37.09790039],
#  [ -79.49984741 , 124.2697525   , 37.09790039],
#  [ -79.49984741 , 124.2697525   ,-37.09790039],
#  [  79.49984741 ,-124.2697525   ,-37.09790039],
#  [  79.49984741 ,-124.2697525    ,37.09790039],
#  [ -79.49984741 ,-124.2697525    ,37.09790039],
#  [ -79.49984741 ,-124.2697525   ,-37.09790039]] ).reshape(-1,3) #bb we calculated

points_2D = np.array([
                [
                    624.3973635858632,
                    125.81950938823746
                ],
                [
                    581.6739923368873,
                    119.73526163998173
                ],
                [
                    518.7324185596469,
                    240.09120969863486
                ],
                [
                    561.436126261706,
                    236.598458549669
                ],
                [
                    782.1496166103178,
                    276.4495186843516
                ],
                [
                    750.4243869425591,
                    284.90611613107956
                ],
                [
                    659.1307048528065,
                    395.928139763825
                ],
                [
                    694.4454564997513,
                    379.4998312999461
                ],
                [
                    641.2664169006495,
                    257.00403469983496
                ]
            ]).reshape(-1,2)
#  self._vertices_original = [
#                 [right, top, front],    # Front Top Right
#                 [left, top, front],     # Front Top Left
#                 [left, bottom, front],  # Front Bottom Left
#                 [right, bottom, front], # Front Bottom Right
#                 [right, top, rear],     # Rear Top Right
#                 [left, top, rear],      # Rear Top Left
#                 [left, bottom, rear],   # Rear Bottom Left
#                 [right, bottom, rear],  # Rear Bottom Right
#                 self.center_location,   # Center
#             ]
 

# map_ = {
#     "0": "1",
#     "1": "2",
#     "2": "6",
#     "3": "5",
#     "4": "0",
#     "5": "3",
#     "6": "7",
#     "7": "4"  
# }


width,height,depth = [158.9997, 248.5395, 74.1958]
cx, cy, cz = [0,0,0]
# X axis point to the right
right = cx + width / 2.0
left = cx - width / 2.0
# Y axis point downward
top = cy + height / 2.0
bottom = cy - height / 2.0
# Z axis point forward
front = cz + depth / 2.0
rear = cz - depth / 2.0

# List of 8 vertices of the box       
_vertices = [
    [right, top, rear],    # Front Top Right
    [right, top, front],     # Front Top Left
    [left, top, front],  # Front Bottom Left
    [left, top, rear], # Front Bottom Right
    [right, bottom, rear],     # Rear Top Right
    [right, bottom, front],      # Rear Top Left
    [left, bottom, front],   # Rear Bottom Left
    [left, bottom, rear], # Rear Bottom Right  # Center
    [cx,cy,cz]
]

vertices = np.array(_vertices)
#print(vertic)
for i in _vertices:
    print(f"\t{i}\n")

# point_3D = R @ point_3d_ours
#R = points_3D @ np.linalg.pinv(points_3D_ours)


_, rvec, tvec = cv.solvePnP(objectPoints=vertices, 
                                            imagePoints=points_2D, 
                                            cameraMatrix=K, 
                                            distCoeffs=None)
tvec_dope = np.array([                0.00375328704887592,
                -0.1170561567293738,
                0.6870602451978979]) * 1000
q = np.array([
                
                0.8678853359344804,
                -0.37783103223583653,
                -0.32210497949602157,
                0.01634432419417263
            
            ])

qwxyz = quat2wxyz(q)
rvec_dope = quat2vec(qwxyz)
print(q.shape)
q_dope = convert_rvec_to_quaternion(rvec)
print('q dope', q_dope)
print('rvec', rvec)
print('rvec_dope', rvec_dope)
print('tvec', tvec)
rotation_mat,_ = cv.Rodrigues(rvec_dope)
RT = np.eye(4)
RT[:3,:3] = rotation_mat #@ rotate3D_X(90) #@ rotate3D_Y(90)
RT[:-1,-1] = tvec_dope.reshape(3,)


print(RT)
modelPoints = load3DModel(modelpath)
    

tpoints = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in modelPoints] 


tpoints = np.array([i/i[-1] for i in tpoints])


cuboid = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in vertices] 
cuboid = np.array([i/i[-1] for i in cuboid])
print(list(cuboid))

img = cv.imread(imgpath)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)


#img = cv.resize(img,(int(1280 / (720/512)),512))


plt.imshow(img)
plt.scatter(tpoints[:,0],tpoints[:,1],marker='x',color='red',s=0.002)

plt.scatter(points_2D[:,0],points_2D[:,1],marker='o',color='yellow')

#plt.scatter(cuboid[:,0],cuboid[:,1],marker='o',color='yellow')
plt.show()



