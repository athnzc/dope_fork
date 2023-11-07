# Keep only one instance with a specific name in the GT data, for testing purposes


import json
from absl import app
from absl import flags
from absl import logging
import glob
import shutil
from natsort import natsorted 
import os
import copy 

FLAGS = flags.FLAGS

flags.DEFINE_string('data', None, 'Folder with GT data.')
flags.DEFINE_string('output', None, 'Folder to save the new annotations')
flags.DEFINE_string('name', None, 'Name of the instance to keep')

def main(argv):
    if FLAGS.data is None:
        logging.info('No input folder given!')
        exit()
   
    if not os.path.exists(FLAGS.data):
        logging.info('Path '+FLAGS.data+' does not exist.')
        exit()

    files = natsorted(glob.glob(FLAGS.data + "/*.json"))

    if FLAGS.output is None:
        FLAGS.output = os.path.join(os.path.dirname(__file__), "output") 
        #logging.info("Output will be stored in" + FLAGS.output_folder)

    if not os.path.exists(FLAGS.output):
        logging.info('Creating '+str(FLAGS.output))
        os.makedirs(FLAGS.output)

    for filename in files:
        with open(filename, 'r') as f:
            logging.info('Loading annotations from '+filename)
            data = json.load(f)
            data_out = {}
            for k in data.keys():
                if k == "objects":
                    objs = data[k]
                    for obj in objs:
                        if obj["name"] == FLAGS.name:
                            data_out[k] = [obj]
                            break
                        else:
                            continue
                else:
                    data_out[k] = data[k] 
        with open(os.path.join(FLAGS.output, os.path.basename(filename)), 'w') as f:
            json.dump(data_out, f, indent=4)

if __name__ == '__main__':
    app.run(main)