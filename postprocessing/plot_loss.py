# Utility script for plotting the loss values in 
# <experiment folder>/log/log_<timestamp>.json

import json
import matplotlib.pyplot as plt
from absl import app
from absl import flags
from absl import logging
import os
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('logfile', None, 'Path to log file with loss')
flags.DEFINE_string('out', None, 'Folder to save the plot')

def main(argv):
    if not os.path.exists(FLAGS.logfile) or FLAGS.logfile is None:
        logging.info('Path does not exist or was not given')
        exit()
    
    if not os.path.exists(FLAGS.out):
        os.makedirs(FLAGS.out)
    with open(FLAGS.logfile, 'r') as f:
        data = json.load(f)
    epochs = []
    values = []

    for k in data.keys():
           #logging.info(str(k) + ' ')
            #logging.info(str(len(data[k]['loss'])))
        e = k.split("_")[0]
            #logging.info(e)
        epochs.append(int(e))
        values.append(np.mean(data[k]['loss']))

    logging.info(str(len(values)))
    min_loss = values[0]
    min_epoch = epochs[0]
    for i in range(4, len(values), 5):
        logging.info(str(epochs[i]) + ' , '+str(values[i]))
        if values[i] < min_loss:
            min_loss = values[i]
            min_epoch = epochs[i]

    
    logging.info('Minimum loss = '+str(min_loss)+' (epoch '+str(min_epoch)+')')
    plt.plot(epochs, values, 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.show()
    plt.savefig(os.path.join(FLAGS.out, 'figures', 'plot.png'))
    
    
    #logging.info(str(epochs))

if __name__ == '__main__':
    app.run(main)

