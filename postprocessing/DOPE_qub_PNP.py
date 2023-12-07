import numpy as np
import cv2 as cv
from plyfile import PlyData
import matplotlib.pyplot as plt

def load3DModel(modelpath):
    
    data = PlyData.read(modelpath)

    x = np.array(data['vertex']['x'])
    y = np.array(data['vertex']['y'])
    z = np.array(data['vertex']['z'])
    points = np.column_stack((x, y, z))

    return points

modelpath = "/home/foto1/linux_part/athena/isaac/dope_fork/3d_models/phase_II_objects/models12/obj_000001_opencv_axes.ply"
imgpath = "/home/foto1/linux_part/athena/isaac/dope_fork/datasets/epos_sample_test/000048.png"
# points_3D = np.array([[  79.49984741  ,124.2697525   ,-37.09790039],
#  [  79.49984741 , 124.2697525   , 37.09790039],
#  [ -79.49984741 , 124.2697525   , 37.09790039],
#  [ -79.49984741 , 124.2697525   ,-37.09790039],
#  [  79.49984741 ,-124.2697525   ,-37.09790039],
#  [  79.49984741 ,-124.2697525    ,37.09790039],
#  [ -79.49984741 ,-124.2697525    ,37.09790039],
#  [ -79.49984741 ,-124.2697525   ,-37.09790039],
#  [  18.61763246  ,  6.76022065    ,1.25508838]] ).reshape(-1,3) bb we calculated

points_2D_gt = np.array([
                [605.2860770959938,119.08332358952686],
                [577.6595390196597,113.3130272672194],
                [512.9097250732864,245.20495755659894],
                [543.7964724588015,239.1052787043335],
                [789.7347056562941,243.17489769574937],
                [780.8940898634863,250.6637406544419],
                [693.7440264934177,375.1029407683921],
                [709.6314316482853,357.1022118129264],
                [651.7442110226884,226.35262080202583]
            ]).reshape(-1,2)

# points_2D =   np.array([[446.8572755710199,150.3867111498704],
#                 [515.3941184160616,11.512650536319256],
#                 [417.41355300743083,116.93203585701033],
#                 [342.9919095586238,298.1313030526258],
#                 [503.2550864879097,183.0632451898672],
#                 [585.8953209634074,54.314082861677264],
#                 [546.5523973782202,192.07785037388533],
#                 [431.5283975227343,346.64694205679683],
#                 [477.4410031307124,163.17640858508022]
#             ]).reshape(-1,2)

points_2D =  np.array([[812.5680189090923,186.27874950390915],
                [725.1879167615009,231.6477829587856],
                [608.3714532456638,241.8280886702449],
                [679.419568890647,207.49184252991654],
                [899.5111319188127,499.2327452411309],
                [832.8779173126793,555.2589018266242],
                [707.1092190276785,489.1520795244044],
                [764.5292532971197,448.4289849565009],
                [761.2706912891252,363.0722394743573 ]]).reshape(-1,2)

# points_2D = np.array([[
#                     917.8033161743382,
#                     90.45121122362303
#                 ],
#                 [
#                     407.8800744409805,
#                     447.7631323692482
#                 ],
#                 [
#                     575.8361433914924,
#                     340.9797021158142
#                 ],
#                 [
#                     780.7411982619682,
#                     198.11984566720938
#                 ],
#                 [
#                     1079.709818979982,
#                     347.86595052744826
#                 ],
#                 [
#                     562.6788586259813,
#                     656.9693307849725
#                 ],
#                 [
#                     639.56859774506,
#                     431.6086310127985
#                 ],
#                 [
#                     845.6841620874355,
#                     296.70085002073233
#                 ],
#                 [
#                     711.9647420846543,
#                     342.8401195020448
#                 ]]).reshape(-1,2)

points_3D = np.array([[  7.949985, -12.426975,   3.70979 ],
 [ -7.949985, -12.426975,   3.70979 ],
 [ -7.949985, 12.426975,   3.70979 ],
 [  7.949985,  12.426975,   3.70979 ],
 [  7.949985, -12.426975,  -3.70979 ],
 [ -7.949985, -12.426975,  -3.70979 ],
 [ -7.949985,  12.426975,  -3.70979 ],
 [  7.949985,  12.426975,  -3.70979 ],
 [  0.0,         0.0,         0.0      ]]
).reshape(-1,3)

print(points_3D.shape)

K = np.array([634.364, 0.0, 637.801, 0.0, 633.635, 364.958, 0.0, 0.0, 1.0]).reshape(3,3) #s/ 1.40625
print(K)

_, rvec, tvec = cv.solvePnP(objectPoints=points_3D, 
                                            imagePoints=points_2D, 
                                            cameraMatrix=K, 
                                            distCoeffs=None)

rotation_mat,_ = cv.Rodrigues(rvec)
RT = np.eye(4)
print(tvec.shape)
RT[:3,:3] = rotation_mat
RT[:-1,-1] = tvec.reshape(3,)


print('RT', RT)
print(rotation_mat,tvec)


modelPoints = load3DModel(modelpath)
    

tpoints = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in modelPoints] 


tpoints = np.array([i/i[-1] for i in tpoints])


cuboid = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in points_3D] 
cuboid = np.array([i/i[-1] for i in cuboid])
print(list(cuboid))

img = cv.imread(imgpath)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)


#img = cv.resize(img,(int(1280 / (720/512)),512))


plt.imshow(img)
plt.scatter(tpoints[:,0],tpoints[:,1],marker='x',color='red',s=0.002)

# #plt.scatter(points_2D_gt[:,0],points_2D_gt[:,1],marker='o',color='yellow')

# plt.scatter(points_2D[:,0],points_2D[:,1],marker='o',color='yellow')
plt.show()



