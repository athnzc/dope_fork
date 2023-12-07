import cv2
import json
from absl import app
from absl import flags
from absl import logging
import os 
from plyfile import PlyData
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('json_file', None, 'Path to json file with annotations in NViSII/DOPE format.')
flags.DEFINE_string('img', None, 'Path to img.')
flags.DEFINE_string('model', None, 'Path to 3D model (.ply format)')
flags.DEFINE_string('out', 'render.png', 'Path to output file (by default it saves the rendered image as\nrender.png in the directory invoking the script)')

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data 

# reads model from ply file and transforms the data into an N X 3 matrix [[x0,y0,z0]...[xN-1, yN-1, zN-1]]
def load_ply(model_path):
    with open(model_path, 'rb') as f:
        plydata = PlyData.read(f)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    logging.info('x shape'+str(x.shape))
    logging.info(str(x[0])+' '+str(y[0])+' '+str(z[0]))

    vertices = np.column_stack((x,y,z))
    logging.info(str(vertices[0]) + str(vertices.shape))

    return vertices

# get Camera matrix 3x3, A = [fx 0 cx; 0 fy cy; 0 0 1]

def get_K(annot_data):
    K=np.zeros((3,3))
    K[2][2] = 1
    cam_data = annot_data['camera_data']['intrinsics']
    K[0][0] = cam_data['fx']
    K[0][2] = cam_data['cx']
    K[1][1] = cam_data['fy']
    K[1][2] = cam_data['cy']

    logging.info(str(K))
    return K 

#def get_location(anno)
def main(argv):
    if FLAGS.json_file is None or not os.path.exists(FLAGS.json_file):
        logging.info('Annotation file was not provided or does not exist')
        exit()
    if FLAGS.img is None or not os.path.exists(FLAGS.img):
        logging.info('Image was not provided or does not exist')
        exit()
    if FLAGS.model is None or not os.path.exists(FLAGS.model):
        logging.info('3D model file was not provided or does not exist')
        exit()
    points3d = load_ply(FLAGS.model)
    data = read_json(FLAGS.json_file)
    K = get_K(data)

    

if __name__== '__main__':
    app.run(main)