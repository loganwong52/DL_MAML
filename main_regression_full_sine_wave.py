"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pickle
import random
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags

# To time it
import time

import os

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


# calculated for omniglot
NUM_TEST_POINTS = 600
   
def test_entire_sine_wave_correctly(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    np.random.seed(1)
    random.seed(1)

    all_predictions = []

    for _ in range(NUM_TEST_POINTS):
        # Generate one sinusoid task
        batch_x, batch_y, amp, phase = data_generator.generate(train=False)

        amp = float(amp[0])
        phase = float(phase[0])
        
        # Calculate (x,y) points of the ENTIRE sine wave
        dense_x = np.linspace(-5, 5, 200).reshape(1, 200, 1)
        dense_x = dense_x.astype(np.float32)
        dense_y = amp * np.sin(dense_x - phase)
        dense_y = dense_y.astype(np.float32)
        
        # Save batch dimension
        dense_x_flat = dense_x[0,:,0]
        dense_y_flat = dense_y[0,:,0]

        # Pre-update predictions, shape (200,)
        preupdate = sess.run(model.forward(dense_x, model.weights, reuse=True))[0,:,0]   

        # Original support set for adaptation (not actually used)        
        inputa = batch_x[:, :FLAGS.update_batch_size, :]
        labela = batch_y[:, :FLAGS.update_batch_size, :]
        
        # Original query set (not actually used)
        inputb = batch_x[:,FLAGS.update_batch_size:, :]
        labelb = batch_y[:,FLAGS.update_batch_size:, :]
           
        # Prepare feed dict
        feed = {
            model.inputa: inputa,
            model.labela: labela,
            model.inputb: dense_x,
            model.labelb: np.zeros_like(dense_x),
            model.meta_lr: 0.0
        }

        # Post-update predictions
        postupdate_list = sess.run(model.outputbs, feed)
        # Convert each to be a (200,) array  
        postupdate = [p[0,:,0] for p in postupdate_list]

        # Record
        all_predictions.append({
            'dense_x': dense_x_flat,
            'dense_y': dense_y_flat,
            'preupdate': preupdate,
            'postupdate': postupdate,
            'support_x': inputa,
            'support_y': labela,
            'query_x': inputb,
            'query_y': labelb,
            'amp': amp,
            'phase': phase
        })

    print(f"exp_string: {exp_string}")

    # Save
    out_file = (
        FLAGS.logdir + '/' + exp_string + '/' 
        + 'test_ubs' + str(FLAGS.update_batch_size)
        + '_stepsize' + str(FLAGS.update_lr)
        + '/full_sine_wave_predictions_NEW.pkl'
    )
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(all_predictions, f)

    print("Saved all predictions")


    
    
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def main():
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    
    tf_data_load = False
    input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)
    
    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    
    if FLAGS.baseline:
        # adds 'oracle' or 'pretrained' to folder name         
        exp_string += FLAGS.baseline
        
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    print(f"MODEL FILE: {model_file}")
    exp_string += "_eval_Regression_Full_Sine_Wave"

    # Only meant for Regression of Full Sine Wave
    test_entire_sine_wave_correctly(model, saver, sess, exp_string, data_generator, test_num_updates=None)
        
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    
    # Calculate run time
    training_duration = end_time - start_time
    time_in_minutes_and_seconds = convert(training_duration)
    print(f"Time taken: {time_in_minutes_and_seconds}")
