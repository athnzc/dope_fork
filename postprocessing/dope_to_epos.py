# Convert DOPE annotations/predictions to EPOS annotations
# location is in meters in annotation data, rotation is a xyzw quaternion
# For now it is assumed that there is only one object in each annotation

import glob
import json
from absl import app
from absl import flags
from absl import logging
import os 
from plyfile import PlyData
import numpy as np
from natsort import natsorted 
import math 
import cv2 as cv
from matplotlib import pyplot as plt

_ZEROANGLES = 1E-10

FLAGS = flags.FLAGS

flags.DEFINE_string('input_folder', None,'Folder with annotations')
flags.DEFINE_string('outf', None, 'Folder to store output EPOS annotations')
flags.DEFINE_integer('obj_id', 1, 'object ID')
flags.DEFINE_string('modelpath', None, 'Path of the 3D model in order to visualize the pose for debugging')

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def get_K(annot_data):
    K=np.zeros((3,3))
    K[2][2] = 1
    cam_data = annot_data['camera_data']['intrinsics']
    K[0][0] = cam_data['fx']
    K[0][2] = cam_data['cx']
    K[1][1] = cam_data['fy']
    K[1][2] = cam_data['cy']

    #logging.info(str(K))
    return K

def _clamp(minn, maxn, number):
    if number > maxn:
        return maxn
    elif number < minn:
        return minn
    else: return number 


# quaternion to rotation vector, quaternion must be in the form WXYZ
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

# get RT matrix and rotation matrix from rotation and translation vector (tvec should be in millimeters)
def get_RT_matrix(rvec, tvec):
    rotation_mat,_ = cv.Rodrigues(rvec)
    RT = np.eye(4)
    RT[:3,:3] = rotation_mat
    RT[:-1,-1] = tvec.reshape(3,)
    return RT, rotation_mat

# Get location in mm and rotation expressed as a wxyz quaternion from annotation data

def get_loc_q(data):
    loc = np.array(data["objects"][0]["location"]) * 1000
    q =  np.array(data["objects"][0]["quaternion_xyzw"])
    q = quat2wxyz(q)
    return loc, q

def load3DModel(modelpath):
    data = PlyData.read(modelpath)

    x = np.array(data['vertex']['x'])
    y = np.array(data['vertex']['y'])
    z = np.array(data['vertex']['z'])
    points = np.column_stack((x, y, z))

    return points

# projects the model points on the image read from image path using K, R, T
# and saves the rendered figure to outpath for debugging purposes. Image should have the same (base) name
# as the annotation file
def visualize_pose(modelpoints, K, RT, imgpath, outpath):
        tpoints = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in modelpoints] 
        tpoints = np.array([i/i[-1] for i in tpoints])
        #cuboid = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in points_3D] 
        #cuboid = np.array([i/i[-1] for i in cuboid])
        logging.info('Read image from '+imgpath)
        img = cv.imread(imgpath)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)


        #img = cv.resize(img,(int(1280 / (720/512)),512))
        plt.figure()

        plt.imshow(img)
        plt.scatter(tpoints[:,0],tpoints[:,1],marker='x',color='red',s=0.002)

# #plt.scatter(points_2D_gt[:,0],points_2D_gt[:,1],marker='o',color='yellow')

        #plt.scatter(points_2D[:,0],points_2D[:,1],marker='o',color='yellow')
        #plt.scatter(cuboid[:,0],cuboid[:,1],marker='o',color='cyan')
        #plt.show()
        figname = os.path.join(outpath, os.path.splitext(os.path.basename(imgpath))[0]+'_render.png')
        logging.info('Saving render to '+figname)
        plt.savefig(figname)
        plt.clf()

def main(argv):
    if FLAGS.input_folder is None:
        logging.fatal('No input folder given')

    if not os.path.exists(FLAGS.outf):
        os.makedirs(FLAGS.outf)
    logging.info('Output will be saved in '+FLAGS.outf)

    if FLAGS.modelpath is not None and not os.path.exists(os.path.join(FLAGS.outf, 'figures')):
        os.makedirs(os.path.join(FLAGS.outf, 'figures'))

    cam_data = {}
    scene_gt = {}
    files = natsorted(glob.glob(FLAGS.input_folder + "/*.json"))
    logging.info(str(files) + str(len(files)))

    for i, filename in enumerate(files):
        with open(filename, 'r') as f:
            data = json.load(f)
        img_id = str(int(os.path.splitext(os.path.basename(filename))[0])) # hack to remove leading zeros
        logging.info('image ID ' + img_id)
        K = get_K(data)
        K = K.reshape(1,9)
        #logging.info(str(K))
        cam_data[img_id] = {"cam_K": list(K[0]), "depth_scale": 1.0}
        logging.info(str(cam_data))
        if len(data['objects']) > 0:
            tvec, q = get_loc_q(data)
            rvec = quat2vec(q)
            RT, rot_mat = get_RT_matrix(rvec, tvec)
            rot_mat = rot_mat.reshape(1,9)
            scene_gt[img_id] = [{"cam_R_m2c" : list(rot_mat[0]), "cam_t_m2c": list(tvec), "obj_id": FLAGS.obj_id}]
            logging.info(str(scene_gt))


            if FLAGS.modelpath is not None:
                modelPoints = load3DModel(FLAGS.modelpath)
                imgpath = os.path.splitext(filename)[0]+".png"
                K = K.reshape((3,3))
                logging.info(K)
                visualize_pose(modelPoints, K, RT, imgpath, os.path.join(FLAGS.outf, 'figures'))
        else:
            logging.info('No pose in '+filename)
            scene_gt[img_id] = [{"cam_R_m2c" : [], "cam_t_m2c": [], "obj_id": FLAGS.obj_id}]
            logging.info(str(scene_gt))

    
    cam_file = os.path.join(FLAGS.outf, 'scene_camera.json')
    with open(cam_file, 'w') as f:
        json.dump(cam_data, f, indent = 4)
    gt_file = os.path.join(FLAGS.outf, 'scene_gt.json')
    with open(gt_file, 'w') as f:
        json.dump(scene_gt, f, indent = 4)

        
if __name__=='__main__':
    app.run(main)