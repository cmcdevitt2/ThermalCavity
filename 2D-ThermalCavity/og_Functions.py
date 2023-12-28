import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_probability as tfp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime

import sys
import os
import time

def runtime_func(start__time):
    total_time = time.time()-start__time
    day_calc = int(np.floor(total_time/(24*60*60)))
    hours_calc = int(np.floor((total_time-day_calc*60*60*24)/(60*60)))
    minutes_calc = int(np.floor((total_time-day_calc*60*60*24-hours_calc*60*60)/(60)))
    seconds_calc = int(np.floor((total_time-day_calc*60*60*24-hours_calc*60*60-minutes_calc*60)/(1)))
    tf.print("\nCode Runtime:")
    tf.print(str(day_calc)+":"+str(hours_calc)+":"+str(minutes_calc)+":"+str(seconds_calc),"(Days:Hours:Minutes:Seconds)\n")
def create_points(n_domain,n_boundary):
    x_points = np.random.rand(n_domain)
    y_points = np.random.rand(n_domain)

    x_walls_vec = np.random.rand(n_boundary)
    z_walls_vec = np.zeros_like(x_walls_vec)

    z_walls = np.random.rand(n_boundary)
    x_walls = np.zeros_like(z_walls)

    x_walls_vec = np.append(x_walls_vec,x_walls)
    z_walls_vec = np.append(z_walls_vec,z_walls)

    x_walls = np.random.rand(n_boundary)
    z_walls = np.ones_like(x_walls)

    x_walls_vec = np.append(x_walls_vec,x_walls)
    z_walls_vec = np.append(z_walls_vec,z_walls)

    z_walls = np.random.rand(n_boundary)
    x_walls = np.ones_like(z_walls)

    x_walls_vec = np.append(x_walls_vec,x_walls)
    z_walls_vec = np.append(z_walls_vec,z_walls)

    x_points = np.append(x_points,x_walls_vec)
    y_points = np.append(y_points,z_walls_vec)

    return x_points.reshape(-1,1),y_points.reshape(-1,1)

def create_wall_points(n_boundary):
    x_walls_vec = np.random.rand(n_boundary)
    z_walls_vec = np.zeros_like(x_walls_vec)

    x_walls = np.random.rand(n_boundary)
    z_walls = np.ones_like(x_walls)

    x_walls_vec = np.append(x_walls_vec,x_walls)
    z_walls_vec = np.append(z_walls_vec,z_walls)

    return x_walls_vec.reshape(-1,1),z_walls_vec.reshape(-1,1)
def output_transform(model,input):
    x = tf.Variable([input[:,0]],dtype=tf.float64)
    z = tf.Variable([input[:,1]],dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        input_tensor=  tf.transpose(tf.concat([x,z],0))
        outputs = model(input_tensor)

        bcv = 16*x*(1-x)*z*(1-z)

        stream = -bcv*bcv*outputs[:,0]
        #stream = look into integrating u and w to find the stream function something along the lines of x*bcv+ z*bcv
        stream = tf.reshape(stream,(1,-1))
        u = tape.gradient(stream,z)
        stream_x = tape.gradient(stream,x)
        w = -1*stream_x
        vort = -1*(tape.gradient(u,z)+tape.gradient(stream_x,x))
        T_hard = 1-(1-0)*x
        T = T_hard+tf.sin(np.pi*x)*outputs[:,1]
        T = tf.reshape(T,(1,-1))

        T_z = tape.gradient(T,z)

        u_vort_dx = tape.gradient(u*vort,x)
        w_vort_dz = tape.gradient(w*vort,z)
        
        vort_x = tape.gradient(vort,x)
        vort_z = tape.gradient(vort,z)
        T_x = tape.gradient(T,x)

        uT_x = tape.gradient(u*T,x)
        wT_z = tape.gradient(w*T,z)

        "Second Derivatives"
        vort_xx = tape.gradient(vort_x,x)
        vort_zz = tape.gradient(vort_z,z)
        T_xx = tape.gradient(T_x,x)
        T_zz = tape.gradient(T_z,z)

        u = tf.reshape(u,(-1,1))
        w = tf.reshape(w,(-1,1))
        stream = tf.reshape(stream,(-1,1))
        vort = tf.reshape(vort,(-1,1))
        T = tf.reshape(T,(-1,1))
        T_z = tf.reshape(T_z,(-1,1))
        u_vort_dx = tf.reshape(u_vort_dx,(-1,1))
        w_vort_dz = tf.reshape(w_vort_dz,(-1,1))
        T_x = tf.reshape(T_x,(-1,1))
        uT_x = tf.reshape(uT_x,(-1,1))
        wT_z = tf.reshape(wT_z,(-1,1))
        vort_xx = tf.reshape(vort_xx,(-1,1))
        vort_zz = tf.reshape(vort_zz,(-1,1))
        T_xx = tf.reshape(T_xx,(-1,1))
        T_zz = tf.reshape(T_zz,(-1,1))

        return tf.concat((u,w,stream,vort,T,T_z,u_vort_dx,w_vort_dz,T_x,uT_x,wT_z,vort_xx,vort_zz,T_xx,T_zz),axis=1)
def plot_loss_history(func,Ra,save_path):
    plt.figure(dpi=300)
    #plt.plot(func.history[100:])
    loss_plot_start = Ra*2
    if len(func.history) < loss_plot_start-1:
        loss_plot_start = 0
    loss_plot_start=0
    loss_plot_start = int(loss_plot_start)
    plt.plot(func.history[loss_plot_start:],label = 'Total Loss')
    loss_labels = ["Loss_PDE_1","Loss_PDE_3","Loss_BC"]
    hist_list = np.array(func.hist_list)
    for ii in range(len(loss_labels)):
        plt.plot(hist_list[loss_plot_start:,ii],label=loss_labels[ii])
    plt.legend()
    plt.yscale('log')
    plt.ylabel("loss")
    plt.xlabel('Epochs')
    #plt.savefig(fname=save_path+"loss_history.png")
    plt.savefig(fname=save_path)
    plt.close('all')
def test_compare(pred_model,Ra,Ra_string):
    if Ra ==1e3:
        tf.print("Ra = "+Ra_string)
        psi_mid_point = [0.5,0.5]
        psi_mid = 1.174
        u_max_z = 0.813
        u_max = 3.649
        u_max_point = [0.5,u_max_z]
        w_max_x = 0.178
        w_max = 3.697
        w_max_point = [w_max_x,0.5]
        model_test_mat = np.concatenate((np.array([psi_mid_point]),np.array([u_max_point]),np.array([w_max_point])),axis=0)
        model_test_tensor = tf.Variable(model_test_mat,dtype=tf.float64)
        model_test_output = output_transform(pred_model,model_test_tensor).numpy()
        #print(model_test_mat)
        #print(model_test_output)
        tf.print("Psi_Mid_Point: \t",model_test_output[0,2],"\tReference:\t",psi_mid)
        tf.print("u_max: \t",model_test_output[1,0],"\tReference:\t",u_max)
        tf.print("w_max: \t",model_test_output[2,1],"\tReference:\t",w_max)
    if Ra ==1e4:
        tf.print("Ra = "+Ra_string)
        psi_mid_point = [0.5,0.5]
        psi_mid = 5.071
        u_max_z = 0.823
        u_max = 16.178
        u_max_point = [0.5,u_max_z]
        w_max_x = 0.119
        w_max = 19.617
        w_max_point = [w_max_x,0.5]
        model_test_mat = np.concatenate((np.array([psi_mid_point]),np.array([u_max_point]),np.array([w_max_point])),axis=0)
        model_test_tensor = tf.Variable(model_test_mat,dtype=tf.float64)
        model_test_output = output_transform(pred_model,model_test_tensor).numpy()
        #print(model_test_mat)
        #print(model_test_output)
        tf.print("Psi_Mid_Point: \t",model_test_output[0,2],"\tReference:\t",psi_mid)
        tf.print("u_max: \t",model_test_output[1,0],"\tReference:\t",u_max)
        tf.print("w_max: \t",model_test_output[2,1],"\tReference:\t",w_max)
    if Ra ==1e5:
        tf.print("Ra = "+Ra_string)
        psi_mid_point = [0.5,0.5]
        psi_mid = 9.111
        u_max_z = 0.855
        u_max = 34.73
        u_max_point = [0.5,u_max_z]
        w_max_x = 0.066
        w_max = 68.59
        w_max_point = [w_max_x,0.5]
        model_test_mat = np.concatenate((np.array([psi_mid_point]),np.array([u_max_point]),np.array([w_max_point])),axis=0)
        model_test_tensor = tf.Variable(model_test_mat,dtype=tf.float64)
        model_test_output = output_transform(pred_model,model_test_tensor).numpy()
        #print(model_test_mat)
        #print(model_test_output)
        tf.print("Psi_Mid_Point: \t",model_test_output[0,2],"\tReference:\t",psi_mid)
        tf.print("u_max: \t",model_test_output[1,0],"\tReference:\t",u_max)
        tf.print("w_max: \t",model_test_output[2,1],"\tReference:\t",w_max)
    if Ra ==1e6:
        tf.print("Ra = "+Ra_string)
        psi_mid_point = [0.5,0.5]
        psi_mid = 16.32
        u_max_z = 0.850
        u_max = 64.63
        u_max_point = [0.5,u_max_z]
        w_max_x = 0.0379
        w_max = 219.36
        w_max_point = [w_max_x,0.5]
        model_test_mat = np.concatenate((np.array([psi_mid_point]),np.array([u_max_point]),np.array([w_max_point])),axis=0)
        model_test_tensor = tf.Variable(model_test_mat,dtype=tf.float64)
        model_test_output = output_transform(pred_model,model_test_tensor).numpy()
        #print(model_test_mat)
        #print(model_test_output)
        tf.print("Psi_Mid_Point: \t",model_test_output[0,2],"\tReference:\t",psi_mid)
        tf.print("u_max: \t",model_test_output[1,0],"\tReference:\t",u_max)
        tf.print("w_max: \t",model_test_output[2,1],"\tReference:\t",w_max)
def big_plot(xx,yy,outputs,ylen,xlen,Ra_string,save_path,iteration,nepochs):
    plot_list = ['u','w','stream','uXvort_x','wXvort_z','T','T_x','T_z','T_xx','T_zz','vort','vort_xx','vort_zz','uT_x','wT_z']
    plot_title = [
        r"$u$",
        r"$w$",
        r"$\psi$",
        r"$\frac{{\partial}(u\zeta)}{{\partial}x}$",
        r"$\frac{{\partial}(w\zeta)}{{\partial}z}$",
        r"$T$",
        r"$\frac{{\partial}T}{{\partial}x}$",
        r"$\frac{{\partial}T}{{\partial}z}$",
        r"$\frac{{\partial}^{2}T}{{\partial}x^{2}}$",
        r"$\frac{{\partial}^{2}T}{{\partial}z^{2}}$",
        r"$\zeta$",
        r"$\frac{{\partial}^{2}\zeta}{{\partial}x^{2}}$",
        r"$\frac{{\partial}^{2}\zeta}{{\partial}z^{2}}$",
        r"$\frac{{\partial}(uT)}{{\partial}x}$",
        r"$\frac{{\partial}(wT)}{{\partial}z}$",
    ]
    plot_order = [0,1,2,6,7,4,8,5,13,14,3,11,12,9,10]
    nlevels = 50

    plt.figure(figsize=(30,16),dpi=200)

    for ii in range(len(plot_list)):
        plt.subplot(3,5,ii+1)
        
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        divider = make_axes_locatable(ax)
        plt.contourf(xx,yy,outputs[:,plot_order[ii]].reshape(ylen,xlen),levels=nlevels,cmap='jet')

        #cb = plt.colorbar("right",cax=ax,fraction=0.1)
        plt.xlabel('x',fontsize=25)
        plt.ylabel('z',fontsize=25)
        plt.title(plot_title[ii],fontsize=25,pad=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        colorbar_axes = divider.append_axes("right",
                                        size="5%",
                                        pad=0.2)
        cb = plt.colorbar(cax=colorbar_axes)
        cb.ax.tick_params(labelsize=18)

        
    plt.suptitle("Ra = "+Ra_string,fontsize=35,y=0.99)
    plt.tight_layout()
    plt.savefig(fname=save_path+"Big_Plots/Ra="+Ra_string+"_outputs_"+str(iteration)+"_"+str(nepochs)+".png")
    plt.close('all')
def individual_plots(xx,yy,outputs,ylen,xlen,Ra_string,save_path):
    nlevels = 50
    plot_list = ['u','w','stream','vort','T','T_z','uXvort_x','wXvort_z','T_x','uT_x','wT_z','vort_xx','vort_zz','T_xx','T_zz']
    plot_title = [
        r"$u$",
        r"$w$",
        r"$\psi$",
        r"$\frac{{\partial}(u\zeta)}{{\partial}x}$",
        r"$\frac{{\partial}(w\zeta)}{{\partial}z}$",
        r"$T$",
        r"$\frac{{\partial}T}{{\partial}x}$",
        r"$\frac{{\partial}T}{{\partial}z}$",
        r"$\frac{{\partial}^{2}T}{{\partial}x^{2}}$",
        r"$\frac{{\partial}^{2}T}{{\partial}z^{2}}$",
        r"$\zeta$",
        r"$\frac{{\partial}^{2}\zeta}{{\partial}x^{2}}$",
        r"$\frac{{\partial}^{2}\zeta}{{\partial}z^{2}}$",
        r"$\frac{{\partial}(uT)}{{\partial}x}$",
        r"$\frac{{\partial}(wT)}{{\partial}z}$",
    ]
    for ii in range(len(plot_list)):
        plt.figure(dpi=200)
        plt.rcParams.update({'font.size':13})
        plt.contourf(xx,yy,outputs[:,ii].reshape(ylen,xlen),levels=nlevels,cmap='jet')
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(plot_title[ii])
        plt.savefig(fname=save_path+"Individual_Plots/"+Ra_string+"_"+plot_list[ii]+".png")
        plt.close()
def BFGS_training(func,pred_model,iterations_per_segment):
#def BFGS_training(func,parameters,iterations_per_segment):
        init_params = tf.dynamic_stitch(func.idx, pred_model.trainable_variables)
        results = tfp.optimizer.bfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=iterations_per_segment,parallel_iterations=100)
        #tf.print(parameters)
        #results = tfp.optimizer.bfgs_minimize(value_and_gradients_function=func, initial_position=parameters, max_iterations=iterations_per_segment)
        func.assign_new_model_parameters(results.position)