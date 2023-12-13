import numpy as np 
from plyfile import PlyData
import math
import cv2 as cv
from matplotlib import pyplot as plt
import json

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

def quat2wxyz(q):
    qwxyz = np.array([q[3], q[0], q[1], q[2]])
    return qwxyz

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    obj_data = data['objects'][0]
    tvec = np.array(obj_data['location'])
    q = np.array(obj_data['quaternion_xyzw'])
    points_2D = np.array(obj_data['projected_cuboid']).reshape(-1,2)

    return tvec, q, points_2D

# tvec = np.array([
#                 -0.02949965367163712,
#                 0.05490672922571305,
#                 0.3999019231168212
#             ])*100

# rest = np.array([
#                 0.4225919560784482,
#                 -0.1251170413215124,
#                 -0.745754167077019,
#                 0.4996123366331175
#             ])
# points_2D =  np.array([
#                 [
#                     740.1173932264135,
#                     269.2477123061968
#                 ],
#                 [
#                     652.3755903895598,
#                     254.54281886308797
#                 ],
#                 [
#                     677.1030512860622,
#                     430.2397878676978
#                 ],
#                 [
#                     752.823432436641,
#                     458.7169586053387
#                 ],
#                 [
#                     413.65250467670603,
#                     496.69077494014084
#                 ],
#                 [
#                     330.91323982011744,
#                     441.0900194132141
#                 ],
#                 [
#                     432.0256489381905,
#                     646.756993678855
#                 ],
#                 [
#                     508.738078700956,
#                     716.5936500856421
#                 ],
#                 [
#                     591.0057304101418,
#                     451.956394760836
#                 ]
#             ]).reshape(-1,2)

# points_3D = np.array([[  7.949985,  12.426975,  -3.70979 ],
#  [  7.949985,  12.426975,   3.70979 ],
#  [ -7.949985,  12.426975,   3.70979 ],
#  [ -7.949985,  12.426975,  -3.70979 ],
#  [  7.949985, -12.426975,  -3.70979 ],
#  [  7.949985, -12.426975,   3.70979 ],
#  [ -7.949985, -12.426975,   3.70979 ],
#  [ -7.949985, -12.426975,  -3.70979 ],
#  [  0.0,         0.0,         0.0      ]]
# ).reshape(-1,3) * 10
points_3D = np.array([[  7.949985,  12.426975,  -3.70979 ],
 [  7.949985,  12.426975,   3.70979 ],
 [ -7.949985,  12.426975,   3.70979 ],
 [ -7.949985,  12.426975,  -3.70979 ],
 [  7.949985, -12.426975,  -3.70979 ],
 [  7.949985, -12.426975,   3.70979 ],
 [ -7.949985, -12.426975,   3.70979 ],
 [ -7.949985, -12.426975,  -3.70979 ],
 [  0.0,         0.0,         0.0      ]]
).reshape(-1,3) * 10

K = np.array([634.364, 0.0, 637.801, 0.0, 633.635, 364.958, 0.0, 0.0, 1.0]).reshape(3,3) #s/ 1.40625

print(K)
modelpath = "/home/foto1/linux_part/athena/isaac/dope_fork/3d_models/phase_II_objects/models12/obj_000001.ply"
imgpath = "/home/foto1/linux_part/athena/isaac/dope_fork/outputs/experiment22/9.png"
jsonpath = "/home/foto1/linux_part/athena/isaac/dope_fork/outputs/experiment22/9.json"
tvec, q, points_2D = read_json(jsonpath)
qwxyz = quat2wxyz(q)
print(qwxyz)
print(points_2D)
print(points_3D)

tvec = tvec * 1000
print('tvec', tvec)
rvec = quat2vec(qwxyz)
print('rvec', rvec)
rotation_mat,_ = cv.Rodrigues(rvec)
RT = np.eye(4)


RT[:3,:3] = rotation_mat
RT[:-1,-1] = tvec.reshape(3,)
#print(tvec)

print('RT\n', RT)
print(rotation_mat,tvec)


modelPoints = load3DModel(modelpath)
    

tpoints = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in modelPoints] 


tpoints = np.array([i/i[-1] for i in tpoints])


cuboid = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in points_3D] 
cuboid = np.array([i/i[-1] for i in cuboid])

print('cuboid', list(cuboid))

img = cv.imread(imgpath)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)


#img = cv.resize(img,(int(1280 / (720/512)),512))


plt.imshow(img)
plt.scatter(tpoints[:,0],tpoints[:,1],marker='x',color='red',s=0.002)

# #plt.scatter(points_2D_gt[:,0],points_2D_gt[:,1],marker='o',color='yellow')

plt.scatter(points_2D[:,0],points_2D[:,1],marker='o',color='yellow')
#plt.scatter(cuboid[:,0],cuboid[:,1],marker='o',color='cyan')
plt.show()
