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

Ra_string = sys.argv[3].split("=")[1]
Ra = float(Ra_string)

nepochs = int(float(sys.argv[4].split("=")[1]))

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
    input_layer = keras.layers.Input(shape=[2,])
    hidden_1 = keras.layers.Dense(64,activation="tanh")(input_layer)
    hidden_2 = keras.layers.Dense(64,activation="tanh")(hidden_1)
    hidden_3 = keras.layers.Dense(64,activation="tanh")(hidden_2)
    hidden_4 = keras.layers.Dense(64,activation="tanh")(hidden_3)
    hidden_5 = keras.layers.Dense(64,activation="tanh")(hidden_4)
    hidden_6 = keras.layers.Dense(64,activation="tanh")(hidden_5)
    hidden_7 = keras.layers.Dense(64,activation="tanh")(hidden_6)
    
    #concat_layer = keras.layers.Concatenate()([hidden_1,hidden_7])
    #output_layer = keras.layers.Dense(3,None)(concat_layer)

    output_layer = keras.layers.Dense(3,None)(hidden_7)
    pred_model = keras.Model(inputs=[input_layer],outputs=[output_layer])

    pred_model.summary()


    # x_input,z_input = create_points(10000,1000)
    # x_walls,z_walls = create_wall_points(1000)

    x_input,z_input = create_points(10000,1000)
    x_walls,z_walls = create_wall_points(1000)

    xlen = 200
    ylen = 200
    x = np.linspace(0,1,endpoint=True,num = xlen)
    y = np.linspace(0,1,endpoint=True,num=ylen)
    xx,yy = np.meshgrid(x,y)
    x_test_input = np.ravel(xx).reshape(-1,1)
    z_test_input = np.ravel(yy).reshape(-1,1)
    input_test_mat = np.concatenate((x_test_input,z_test_input),axis=1)

    input_mat = np.concatenate((x_input,z_input),axis=1)
    bc_input_mat = np.concatenate((x_walls,z_walls),axis=1)
    inps_tensor = tf.Variable(input_mat,dtype=tf.float64)
    bc_input_tensor = tf.Variable(bc_input_mat,dtype=tf.float64)



    # if Ra == 1e5:
    #     temp_epochs = int(1e3)
    #     temp_func = function_factory(pred_model,inps_tensor,bc_input_tensor,Ra=1e4,nepochs=temp_epochs)
    #     BFGS_training(temp_func,pred_model,temp_epochs)

    # func = function_factory(pred_model,inps_tensor,bc_input_tensor,Ra,nepochs)

    n_segments = 1
    iterations_per_segment = int(nepochs/n_segments)
    iter_value=0
    loss_hist = []
    loss_hist_list = []
    func = function_factory(pred_model,inps_tensor,bc_input_tensor,Ra,nepochs,iter_value,loss_hist,loss_hist_list)
    for ii in range (0,n_segments):
        #func = function_factory(pred_model,inps_tensor,bc_input_tensor,Ra,iterations_per_segment)
        max_value = int(float(nepochs)*float(ii)/n_segments)
        tf.print("\nCheckpoint "+str(ii)+" of "+str(n_segments))
        runtime_func(start_time)
        outputs = output_transform(pred_model,input_test_mat).numpy()
        big_plot(xx,yy,outputs,ylen,xlen,Ra_string,save_path,iteration=max_value,nepochs=nepochs)
        test_compare(pred_model,Ra,Ra_string)
        tf.print("-"*120)
        
        BFGS_training(func,pred_model,iterations_per_segment)
        loss_hist = func.history
        loss_hist_list = func.hist_list
        iter_value = func.iter
        plot_loss_history(func,Ra,save_path+"/Loss_Plots/"+str(int(max_value+nepochs/n_segments))+"_of_"+str(nepochs))
        pred_model.save(save_path+"Checkpoints/"+str(int(max_value+nepochs/n_segments))+"_of_"+str(nepochs))
    
    tf.print("\nCheckpoint "+str(n_segments)+" of "+str(n_segments))
    outputs = output_transform(pred_model,input_test_mat).numpy()

    big_plot(xx,yy,outputs,ylen,xlen,Ra_string,save_path,iteration=nepochs,nepochs=nepochs)
    test_compare(pred_model,Ra,Ra_string)

    
    save_mat = np.concatenate((input_test_mat,outputs),axis=1)
    np.savetxt(save_path+'results.txt',save_mat)
    
    individual_plots(xx,yy,outputs,ylen,xlen,Ra_string,save_path)


    runtime_func(start_time)