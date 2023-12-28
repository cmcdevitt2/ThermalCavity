import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import keras
# from keras.layers import Layer
# from keras.models import Sequential
# from keras.layers import Dense
# import tensorflow_probability as tfp
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import datetime

# import sys
# import os
# import time
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
def output_transform3d(model,input):
    x = tf.Variable([input[:,0]],dtype=tf.float64)
    z = tf.Variable([input[:,1]],dtype=tf.float64)
    Ra = tf.Variable([input[:,2]],dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        input_tensor=  tf.transpose(tf.concat([x,z,Ra],0))
        outputs = model(input_tensor,training=False)

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
def big_plot(outputs,xx,yy,theLength,Ra_string,save_path):
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

    plt.figure(figsize=(40,20),dpi=200)
    xlen = ylen = theLength
    for ii in range(len(plot_list)):
        plt.subplot(3,5,ii+1)
        
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        divider = make_axes_locatable(ax)
        plt.contourf(xx,yy,outputs[:,plot_order[ii]].reshape(ylen,xlen),levels=nlevels,cmap='jet')

        plt.xlabel('x',fontsize=40)
        plt.ylabel('z',fontsize=40)
        plt.title(plot_title[ii],fontsize=45,pad=18)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        colorbar_axes = divider.append_axes("right",
                                        size="7%",
                                        pad=0.2)
        cb = plt.colorbar(cax=colorbar_axes)
        cb.ax.tick_params(labelsize=25)

        
    plt.suptitle(f"Ra = {Ra_string}",fontsize=45,y=0.99)
    plt.tight_layout()
    plt.savefig(fname=f"{save_path}BigPlots_{Ra_string}.png")
    plt.close('all')

def plotRaDerivatives(model3d,inputsr,nx,Ra_string,save_path):
    x = tf.Variable([inputsr[:,0]],dtype=tf.float64)
    z = tf.Variable([inputsr[:,1]],dtype=tf.float64)
    Ra = tf.Variable([inputsr[:,2]],dtype=tf.float64)

    xx = x.numpy().reshape(nx,nx)
    yy = z.numpy().reshape(nx,nx)

    nlevels = 50

    with tf.GradientTape(persistent=True) as tape:
        input_tensor=  tf.transpose(tf.concat([x,z,Ra],0))
        outputs = model3d(input_tensor,training=False)

        bcv = 16*x*(1-x)*z*(1-z)

        stream = -bcv*bcv*outputs[:,0]
        #stream = look into integrating u and w to find the stream function something along the lines of x*bcv+ z*bcv
        stream = tf.reshape(stream,(1,-1))
        u = tape.gradient(stream,z)
        stream_x = tape.gradient(stream,x)
        w = -1*stream_x
        T_hard = 1-(1-0)*x
        T = T_hard+tf.sin(np.pi*x)*outputs[:,1]
        T = tf.reshape(T,(1,-1))

        stream_ra = tape.gradient(stream,Ra)
        T_ra = tape.gradient(T,Ra)
        u_ra = tape.gradient(u,Ra)
        w_ra = tape.gradient(w,Ra)

        titles = [
            r"$\frac{{\partial}\psi}{{\partial}Ra}$",
            r"$\frac{{\partial}T}{{\partial}Ra}$",
            r"$\frac{{\partial}u}{{\partial}Ra}$",
            r"$\frac{{\partial}w}{{\partial}Ra}$",
        ]

        plots = [
            stream_ra,
            T_ra,
            u_ra,
            w_ra
        ]

        plt.figure(figsize=(10,10))
        for ii,title in enumerate(titles):
            plt.subplot(2,2,ii+1)
            ax = plt.gca()
            ax.set_aspect('equal', 'box')
            divider = make_axes_locatable(ax)
            plt.contourf(xx,yy,plots[ii].numpy().reshape(nx,nx),levels=nlevels,cmap='jet')
            plt.xlabel('x',fontsize=30)
            plt.ylabel('z',fontsize=30)
            plt.title(title,fontsize=35,pad=18)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            colorbar_axes = divider.append_axes("right",
                                            size="5%",
                                            pad=0.2)
            cb = plt.colorbar(cax=colorbar_axes)
            cb.ax.tick_params(labelsize=20)

        plt.suptitle(f"Ra = {Ra_string}")
        plt.tight_layout()
        plt.savefig(fname=f"{save_path}Derivatives_{Ra_string}.png")
        plt.close('all')



