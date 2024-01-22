import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_probability as tfp
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import time

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

args = sys.argv
num_params = len(sys.argv)
tf.print(args)
start_time = float(sys.argv[2])


nepochs = int(float(sys.argv[3].split("=")[1]))

if num_params > 1:
    name = sys.argv[1]
    save_path = name
sys.path.insert(0, save_path)
from Functions import *
from Func_Fac import *

runtime_func(start_time)


tf.print("Current version of Python is ", sys.version)
tf.print("TensorFlow version:", tf.__version__)
tf.print(tf.config.list_physical_devices(device_type=None))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.keras.utils.set_random_seed(20221220)
tf.config.experimental.enable_op_determinism()
#https://www.tensorflow.org/guide/gpu
#for pip requirements check out pipreqs and piga
"""
pip install pipreqs
then
pipreqs path/to/project

"""

#gpus = tf.config.list_logical_devices()
#strategy = tf.distribute.MirroredStrategy()

if __name__ == "__main__":


    tf.keras.backend.set_floatx("float64")

    # if Ra == 1e5:
    #     #with strategy.scope():
    #     pred_model = tf.keras.models.load_model('1e4_Model')
    # else: 
    #with strategy.scope():
    input_layer = keras.layers.Input(shape=[3,])
    hidden_1 = keras.layers.Dense(50,activation="tanh")(input_layer)
    hidden_2 = keras.layers.Dense(50,activation="tanh")(hidden_1)
    hidden_3 = keras.layers.Dense(50,activation="tanh")(hidden_2)
    hidden_4 = keras.layers.Dense(50,activation="tanh")(hidden_3)
    hidden_5 = keras.layers.Dense(50,activation="tanh")(hidden_4)
    hidden_6 = keras.layers.Dense(50,activation="tanh")(hidden_5)
    hidden_7 = keras.layers.Dense(50,activation="tanh")(hidden_6)
    #concat_layer = keras.layers.Concatenate()([hidden_1,hidden_7])
    #output_layer = keras.layers.Dense(3,None)(concat_layer)

    output_layer = keras.layers.Dense(2,None)(hidden_7)
    pred_model = keras.Model(inputs=[input_layer],outputs=[output_layer])

    pred_model.summary()


    # x_input,z_input = create_points(10000,1000)
    # x_walls,z_walls = create_wall_points(1000)

    x_input,z_input,Ra_input = create_points(50000,3000)
    x_walls,z_walls,Ra_walls = create_wall_points(3000)

    xlen = 30
    ylen = 30
    Ra_len = 30
    x = np.linspace(0,1,endpoint=True,num = xlen)
    y = np.linspace(0,1,endpoint=True,num=ylen)
    Ra = np.logspace(3,6,num=Ra_len,endpoint=True)
    xx,yy,RaRa = np.meshgrid(x,y,Ra)
    x_test_input = np.ravel(xx[:,:,0]).reshape(-1,1)
    z_test_input = np.ravel(yy[:,:,0]).reshape(-1,1)
    Ra_test_input = np.ravel(RaRa[:,:,0]).reshape(-1,1)
    # input_test_mat = np.concatenate((x_test_input,z_test_input,Ra_test_input),axis=1)

    input_mat = np.concatenate((x_input,z_input,Ra_input),axis=1)
    bc_input_mat = np.concatenate((x_walls,z_walls,Ra_walls),axis=1)
    inps_tensor = tf.Variable(input_mat,dtype=tf.float64)
    bc_input_tensor = tf.Variable(bc_input_mat,dtype=tf.float64)


    n_segments = 1
    iterations_per_segment = int(nepochs/n_segments)
    iter_value=0
    loss_hist = []
    loss_hist_list = []
    func = function_factory(pred_model,inps_tensor,bc_input_tensor,nepochs,iter_value,loss_hist,loss_hist_list)
    for ii in range (0,n_segments):
        max_value = int(float(nepochs)*float(ii)/n_segments)
        tf.print("\nCheckpoint "+str(ii)+" of "+str(n_segments))
        runtime_func(start_time)
        for Ra_string in ['1e3','1e4','1e5','1e6']:
            Ra_value = float(Ra_string)
            Ra_input_value = (np.log10(Ra_value)-3)/3
            Ra_test_input = Ra_input_value*np.ones_like(Ra_test_input)
            input_test_mat = np.concatenate((x_test_input,z_test_input,Ra_test_input),axis=1)
            outputs = output_transform(pred_model,input_test_mat).numpy()
            big_plot(xx[:,:,0],yy[:,:,0],outputs,ylen,xlen,Ra_string,save_path,iteration=max_value,nepochs=nepochs)
            test_compare(pred_model,Ra_value,Ra_string)
        tf.print("-"*120)
        
        BFGS_training(func,pred_model,iterations_per_segment)
        loss_hist = func.history
        loss_hist_list = func.hist_list
        iter_value = func.iter
        plot_loss_history(func,save_path+"/Loss_Plots/"+str(int(max_value+nepochs/n_segments))+"_of_"+str(nepochs))
        pred_model.save(save_path+"Checkpoints/"+str(int(max_value+nepochs/n_segments))+"_of_"+str(nepochs))
    
    tf.print("\nCheckpoint "+str(n_segments)+" of "+str(n_segments))
    for Ra_string in ['1e3','1e4','1e5','1e6']:
        Ra_value = float(Ra_string)
        Ra_input_value = (np.log10(Ra_value)-3)/3
        Ra_test_input = Ra_input_value*np.ones_like(Ra_test_input)
        input_test_mat = np.concatenate((x_test_input,z_test_input,Ra_test_input),axis=1)
        outputs = output_transform(pred_model,input_test_mat).numpy()
        big_plot(xx[:,:,0],yy[:,:,0],outputs,ylen,xlen,Ra_string,save_path,iteration=nepochs,nepochs=nepochs)
        test_compare(pred_model,Ra_value,Ra_string)
        save_mat = np.concatenate((input_test_mat,outputs),axis=1)
        np.savetxt(save_path+'Ra='+Ra_string+'results.txt',save_mat)
        individual_plots(xx[:,:,0],yy[:,:,0],outputs,ylen,xlen,Ra_string,save_path)


    runtime_func(start_time)