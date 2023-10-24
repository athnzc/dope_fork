# If the dataset you want to use is divided in scenes (like the HOPE dataset), run this script to unify it

import glob
from natsort import natsorted
from absl import app
from absl import flags
from absl import logging
import os 
import json 
from PIL import Image
import shutil

FLAGS = flags.FLAGS

flags.DEFINE_string('data_folder', None, 'Folder with scene subfolders.')
flags.DEFINE_string('output_folder', None, 'Folder with unified dataset. If not specified, a default output folder is created in the directory the script is invoked')
flags.DEFINE_boolean('train', False, 'Whether you want to create a train or a test set. A test set won\'nt include depth images')
flags.DEFINE_spaceseplist('scenes', None, 'Space-separated list of which scenes you want to process. Default will process all scenes')
flags.DEFINE_integer('digits', 4, '')

# TODO: Remove duplicates from list of scenes - maaaybe

# Returns absolute paths of folders specified by a list of scene IDs
def get_folders(scenes, data_folder):
    if scenes is None:    
        folders = natsorted(glob.glob(data_folder+"/*"))
    else:
        folders = []
        scenes = natsorted(scenes)
        for s in scenes:
            
            logging.info(s)
            for p in glob.glob(data_folder +"/*"+s):
                folders.append(p)

    return folders 

def convert_to_png(image_name, new_image_name):
    im = Image.open(image_name)
    im.save(new_image_name)

def main(argv):
    if FLAGS.data_folder is None:
        logging.info('No input folder given!')
        exit()
   
    if not os.path.exists(FLAGS.data_folder):
        logging.info('Path '+FLAGS.data_folder+' does not exist.')
        exit()

    if FLAGS.output_folder is None:
        FLAGS.output_folder = os.path.join(os.path.dirname(__file__), "output") 
        #logging.info("Output will be stored in" + FLAGS.output_folder)

    if not os.path.exists(FLAGS.output_folder):
        logging.info('Creating '+str(FLAGS.output_folder))
        os.makedirs(FLAGS.output_folder)

    #FLAGS.scenes = natsorted(FLAGS.scenes)
    logging.info(FLAGS.scenes)
    folders = get_folders(FLAGS.scenes, FLAGS.data_folder)

    logging.info("Folders to be processed: " +str(folders) + "("+str(len(folders))+" scenes)")
    count = 0
    for f in folders:
        annots = natsorted(glob.glob(f+"/*.json"))
        depth_img = natsorted(glob.glob(f+"/*depth*"))
        logging.info(depth_img)
        rgb_img = natsorted(glob.glob(f+"/*rgb*"))

        if len(rgb_img) == 0: # rgb images are of the form 0000.png, 0001.png (or jpg)
            rgb_img = natsorted(list(set(natsorted(glob.glob(f+"/*"))).difference(set(depth_img)).difference(set(annots))))   #(turn them into sets or just detract a list from another)

        logging.info(rgb_img)

        for idx, image in enumerate(rgb_img):
            new_basename = (FLAGS.digits - len(str(count)))*'0'+str(count)
            new_rgb_name = os.path.join(FLAGS.output_folder, new_basename+"_rgb.png")

            if os.path.splitext(image)[-1].split('.')[-1] not in ['png']: # if image extension is not png
                convert_to_png(image, new_rgb_name)
                logging.info('Converting rgb image '+image+' to '+new_rgb_name)
            else:
                logging.info('Copying rgb image '+image+' to '+new_rgb_name)
                shutil.copy(image, new_rgb_name)
            
            if len(depth_img) > 0 and FLAGS.train:
                new_depth_name = os.path.join(FLAGS.output_folder, new_basename+"_depth.png")
                d_img = depth_img[idx]

                if os.path.splitext(d_img)[-1].split('.')[-1] not in ['png']:
                    logging.info(os.path.splitext(d_img)[-1].split('.')[-1])
                    convert_to_png(d_img, new_depth_name)
                    logging.info('Converting depth image '+d_img+' to '+new_depth_name)
                else:
                    logging.info('Copying depth image '+d_img+' to '+new_depth_name)
                    shutil.copy(d_img, new_depth_name)

            if len(annots) > 0:
                new_json_name = os.path.join(FLAGS.output_folder, new_basename+".json")
                logging.info('Copying annotation file '+annots[idx]+' to '+new_json_name)
                shutil.copy(annots[idx], new_json_name)  
            count = count + 1  
    
if __name__ == '__main__':
    app.run(main)