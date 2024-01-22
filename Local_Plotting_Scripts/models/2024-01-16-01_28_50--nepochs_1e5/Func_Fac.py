from builtins import iter
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

def function_factory(model,train_x,train_bc,nepochs,iter_val,history,hist_list):

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)
    x = tf.Variable([train_x[:,0]],dtype=tf.float64)
    z = tf.Variable([train_x[:,1]],dtype=tf.float64)
    Ra = tf.Variable([train_x[:,2]],dtype=tf.float64) #this will overwrite the other Ra (which is good for 3d)
    Ra_values = 10**(3+Ra*3)

    x_bc = tf.Variable([train_bc[:,0]],dtype=tf.float64)
    z_bc = tf.Variable([train_bc[:,1]],dtype=tf.float64)
    Ra_bc = tf.Variable([train_bc[:,2]],dtype=tf.float64)

    #hist_list = []


    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape(persistent=True) as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            train_x_tensor = tf.transpose(tf.concat([x,z,Ra],0))

            outputs = model(train_x_tensor,training=True)
            bcv = 16*x*(1-x)*z*(1-z)

            stream = -bcv*bcv*outputs[:,0]
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

            Pr=0.71

            #print("Train_y: ",train_y)
            
            # Ra_points = Ra_points

            loss_1 = u_vort_dx+w_vort_dz-(vort_xx+vort_zz)*Pr-T_x*Ra_values*Pr
            loss_2 = 0
            loss_3 = uT_x+wT_z-(T_xx+T_zz)


            """BC Loss Calculations"""
            train_x_bc_tensor = tf.transpose(tf.concat([x_bc,z_bc,Ra_bc],0))

            outputs_bc = model(train_x_bc_tensor,training=True)

            T_hard_bc = 1-(1-0)*x_bc
            T_bc = T_hard_bc+tf.sin(np.pi*x_bc)*outputs_bc[:,1]
            T_bc = tf.reshape(T_bc,(1,-1))

            T_z_bc = tape.gradient(T_bc,z_bc)
            # bc_loss = tf.reduce_mean(tf.square(T_z_bc),axis=1)
            # """Output loss"""
            # loss_vec_tensor = tf.transpose(tf.concat((loss_1/Ra_values,loss_3),axis=0))
            # mean_loss_vec_tensor = tf.reduce_mean(tf.square(loss_vec_tensor),axis=0)
            # output_loss_vec_tensor = tf.concat((mean_loss_vec_tensor,bc_loss),axis=0)

            # loss_value = tf.reduce_sum(output_loss_vec_tensor)
            pde1_loss = tf.reduce_mean(tf.square(loss_1/Ra_values))
            pde3_loss = tf.reduce_mean(tf.square(loss_3))

            bc_loss = tf.reduce_mean(tf.square(T_z_bc))

            output_loss_vec_tensor = tf.stack((pde1_loss,pde3_loss,bc_loss),axis=0)
            loss_value = tf.reduce_sum(output_loss_vec_tensor)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter % int(nepochs/10) == 0:
            tf.print("Epoch:", f.iter,"/",nepochs,"\tLoss:", loss_value,"\tLoss vec: ",output_loss_vec_tensor)

        # store loss value so we can retrieve later
        tf.py_function(history.append, inp=[loss_value], Tout=[])
        tf.py_function(hist_list.append, inp=[output_loss_vec_tensor], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    try:
        f.iter = f.iter
    except:
        f.iter = tf.Variable(iter_val)

    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = history
    f.hist_list = hist_list

    return f