import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

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
def output_transform(model, input):
    x = tf.Variable([input[:, 0]], dtype=tf.float64)
    z = tf.Variable([input[:, 1]], dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        input_tensor = tf.transpose(tf.concat([x, z], 0))
        outputs = model(input_tensor)

        bcv = 16 * x * (1 - x) * z * (1 - z)

        stream = -bcv * bcv * outputs[:, 0]
        # stream = look into integrating u and w to find the stream function something along the lines of x*bcv+ z*bcv
        stream = tf.reshape(stream, (1, -1))
        u = tape.gradient(stream, z)
        stream_x = tape.gradient(stream, x)
        w = -1 * stream_x
        vort = -1 * (tape.gradient(u, z) + tape.gradient(stream_x, x))
        T_hard = 1 - (1 - 0) * x
        T = T_hard + tf.sin(np.pi * x) * outputs[:, 1]
        T = tf.reshape(T, (1, -1))

        T_z = tape.gradient(T, z)

        u_vort_dx = tape.gradient(u * vort, x)
        w_vort_dz = tape.gradient(w * vort, z)

        vort_x = tape.gradient(vort, x)
        vort_z = tape.gradient(vort, z)
        T_x = tape.gradient(T, x)

        uT_x = tape.gradient(u * T, x)
        wT_z = tape.gradient(w * T, z)

        "Second Derivatives"
        vort_xx = tape.gradient(vort_x, x)
        vort_zz = tape.gradient(vort_z, z)
        T_xx = tape.gradient(T_x, x)
        T_zz = tape.gradient(T_z, z)

        u = tf.reshape(u, (-1, 1))
        w = tf.reshape(w, (-1, 1))
        stream = tf.reshape(stream, (-1, 1))
        vort = tf.reshape(vort, (-1, 1))
        T = tf.reshape(T, (-1, 1))
        T_z = tf.reshape(T_z, (-1, 1))
        u_vort_dx = tf.reshape(u_vort_dx, (-1, 1))
        w_vort_dz = tf.reshape(w_vort_dz, (-1, 1))
        T_x = tf.reshape(T_x, (-1, 1))
        uT_x = tf.reshape(uT_x, (-1, 1))
        wT_z = tf.reshape(wT_z, (-1, 1))
        vort_xx = tf.reshape(vort_xx, (-1, 1))
        vort_zz = tf.reshape(vort_zz, (-1, 1))
        T_xx = tf.reshape(T_xx, (-1, 1))
        T_zz = tf.reshape(T_zz, (-1, 1))

        return tf.concat(
            (
                u,
                w,
                stream,
                vort,
                T,
                T_z,
                u_vort_dx,
                w_vort_dz,
                T_x,
                uT_x,
                wT_z,
                vort_xx,
                vort_zz,
                T_xx,
                T_zz,
            ),
            axis=1,
        )


def output_transform3d(model, input):
    x = tf.Variable([input[:, 0]], dtype=tf.float64)
    z = tf.Variable([input[:, 1]], dtype=tf.float64)
    Ra = tf.Variable([input[:, 2]], dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        input_tensor = tf.transpose(tf.concat([x, z, Ra], 0))
        outputs = model(input_tensor, training=False)

        bcv = 16 * x * (1 - x) * z * (1 - z)

        stream = -bcv * bcv * outputs[:, 0]
        # stream = look into integrating u and w to find the stream function something along the lines of x*bcv+ z*bcv
        stream = tf.reshape(stream, (1, -1))
        u = tape.gradient(stream, z)
        stream_x = tape.gradient(stream, x)
        w = -1 * stream_x
        vort = -1 * (tape.gradient(u, z) + tape.gradient(stream_x, x))
        T_hard = 1 - (1 - 0) * x
        T = T_hard + tf.sin(np.pi * x) * outputs[:, 1]
        T = tf.reshape(T, (1, -1))

        T_z = tape.gradient(T, z)

        u_vort_dx = tape.gradient(u * vort, x)
        w_vort_dz = tape.gradient(w * vort, z)

        vort_x = tape.gradient(vort, x)
        vort_z = tape.gradient(vort, z)
        T_x = tape.gradient(T, x)

        uT_x = tape.gradient(u * T, x)
        wT_z = tape.gradient(w * T, z)

        "Second Derivatives"
        vort_xx = tape.gradient(vort_x, x)
        vort_zz = tape.gradient(vort_z, z)
        T_xx = tape.gradient(T_x, x)
        T_zz = tape.gradient(T_z, z)

        u = tf.reshape(u, (-1, 1))
        w = tf.reshape(w, (-1, 1))
        stream = tf.reshape(stream, (-1, 1))
        vort = tf.reshape(vort, (-1, 1))
        T = tf.reshape(T, (-1, 1))
        T_z = tf.reshape(T_z, (-1, 1))
        u_vort_dx = tf.reshape(u_vort_dx, (-1, 1))
        w_vort_dz = tf.reshape(w_vort_dz, (-1, 1))
        T_x = tf.reshape(T_x, (-1, 1))
        uT_x = tf.reshape(uT_x, (-1, 1))
        wT_z = tf.reshape(wT_z, (-1, 1))
        vort_xx = tf.reshape(vort_xx, (-1, 1))
        vort_zz = tf.reshape(vort_zz, (-1, 1))
        T_xx = tf.reshape(T_xx, (-1, 1))
        T_zz = tf.reshape(T_zz, (-1, 1))

        return tf.concat(
            (
                u,
                w,
                stream,
                vort,
                T,
                T_z,
                u_vort_dx,
                w_vort_dz,
                T_x,
                uT_x,
                wT_z,
                vort_xx,
                vort_zz,
                T_xx,
                T_zz,
            ),
            axis=1,
        )


def big_plot(outputs, xx, yy, theLength, Ra_string, save_path):
    choiceList = (
        ("stream", r"$\psi$", 2),
        ("u", r"$u$", 0),
        ("w", r"$w$", 1),
        ("uXvort_x", r"$\frac{{\partial}(u\zeta)}{{\partial}x}$", 6),
        ("wXvort_z", r"$\frac{{\partial}(w\zeta)}{{\partial}z}$", 7),
        ("T", r"$T$", 4),
        ("T_x", r"$\frac{{\partial}T}{{\partial}x}$", 8),
        ("T_z", r"$\frac{{\partial}T}{{\partial}z}$", 5),
        ("T_xx", r"$\frac{{\partial}^{2}T}{{\partial}x^{2}}$", 13),
        ("T_zz", r"$\frac{{\partial}^{2}T}{{\partial}z^{2}}$", 14),
        ("vort", r"$\zeta$", 3),
        ("vort_xx", r"$\frac{{\partial}^{2}\zeta}{{\partial}x^{2}}$", 11),
        ("vort_zz", r"$\frac{{\partial}^{2}\zeta}{{\partial}z^{2}}$", 12),
        ("uT_x", r"$\frac{{\partial}(uT)}{{\partial}x}$", 9),
        ("wT_z", r"$\frac{{\partial}(wT)}{{\partial}z}$", 10),
    )

    nlevels = 10

    plt.figure(figsize=(15, 9), dpi=200)
    xlen = ylen = theLength
    for ii in range(len(choiceList)):
        plt.subplot(3, 5, ii + 1)

        ax = plt.gca()
        ax.set_aspect("equal", "box")
        divider = make_axes_locatable(ax)
        name, LatexName, outputIndex = choiceList[ii]
        plt.contourf(
            xx,
            yy,
            outputs[:, outputIndex].reshape(ylen, xlen),
            levels=nlevels,
            cmap="jet",
        )

        plt.xlabel(r"$x$", fontsize=20)
        plt.ylabel(r"$z$", fontsize=20)
        plt.title(LatexName, fontsize=25, pad=18)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        colorbar_axes = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(cax=colorbar_axes)
        cb.ax.tick_params(labelsize=10)

    plt.suptitle(f"Ra = {Ra_string}", fontsize=25, y=0.98)
    plt.tight_layout()
    plt.savefig(fname=f"{save_path}BigPlots_{Ra_string}.png")
    plt.close("all")


def plotRaDerivatives(model3d, inputsr, nx, Ra_string, save_path):
    x = tf.Variable([inputsr[:, 0]], dtype=tf.float64)
    z = tf.Variable([inputsr[:, 1]], dtype=tf.float64)
    Ra = tf.Variable([inputsr[:, 2]], dtype=tf.float64)

    xx = x.numpy().reshape(nx, nx)
    yy = z.numpy().reshape(nx, nx)

    nlevels = 10

    with tf.GradientTape(persistent=True) as tape:
        input_tensor = tf.transpose(tf.concat([x, z, Ra], 0))
        outputs = model3d(input_tensor, training=False)

        bcv = 16 * x * (1 - x) * z * (1 - z)

        stream = -bcv * bcv * outputs[:, 0]
        stream = tf.reshape(stream, (1, -1))
        u = tape.gradient(stream, z)
        stream_x = tape.gradient(stream, x)
        w = -1 * stream_x
        T_hard = 1 - (1 - 0) * x
        T = T_hard + tf.sin(np.pi * x) * outputs[:, 1]
        T = tf.reshape(T, (1, -1))

        stream_ra = tape.gradient(stream, Ra)
        T_ra = tape.gradient(T, Ra)
        u_ra = tape.gradient(u, Ra)
        w_ra = tape.gradient(w, Ra)

        titles = [
            r"$\frac{{\partial}\psi}{{\partial}Ra}$",
            r"$\frac{{\partial}T}{{\partial}Ra}$",
            r"$\frac{{\partial}u}{{\partial}Ra}$",
            r"$\frac{{\partial}w}{{\partial}Ra}$",
        ]

        plots = [stream_ra, T_ra, u_ra, w_ra]

        raPrime = inputsr[0, 2]

        derivScaleRa = 1 / (10 ** (raPrime * 3 + 3))

        plt.figure(figsize=(10, 10), dpi=200)
        for ii, title in enumerate(titles):
            plt.subplot(2, 2, ii + 1)
            ax = plt.gca()
            ax.set_aspect("equal", "box")
            divider = make_axes_locatable(ax)
            plt.contourf(
                xx,
                yy,
                derivScaleRa * plots[ii].numpy().reshape(nx, nx),
                levels=nlevels,
                cmap="jet",
            )
            plt.xlabel(r"$x$", fontsize=30)
            plt.ylabel(r"$z$", fontsize=30)
            plt.title(title, fontsize=35, pad=18)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            colorbar_axes = divider.append_axes("right", size="5%", pad=0.2)
            cb = plt.colorbar(cax=colorbar_axes)
            cb.ax.tick_params(labelsize=15)

        plt.suptitle(f"Ra = {Ra_string}", size=45)
        plt.tight_layout()
        plt.savefig(fname=f"{save_path}Derivatives_{Ra_string}.png")
        plt.close("all")


def midPlanePlots(plotDict2D, plotsWanted3D, model3D):
    npoints = 100
    plotPoints = np.linspace(0, 1, endpoint=True, num=npoints).reshape(-1, 1)
    midPoints = np.ones_like(plotPoints)*0.5

    """
    This section is for the horizontal cross section
    constant z = 0.5
    """

    x = tf.Variable([plotPoints[:, 0]], dtype=tf.float64)
    z = tf.Variable([midPoints[:, 0]], dtype=tf.float64)

    raBase = np.ones_like(midPoints)

    listData2d = []
    listData3d = []
    listData3dGrads = []

    raKeyList = list(plotDict2D.keys())

    for ra in raKeyList:
        path = plotDict2D[ra]
        model = keras.models.load_model(path)
        with tf.GradientTape(persistent=True) as tape:
            input_tensor = tf.transpose(tf.concat([x, z], 0))
            model = keras.models.load_model(path)
            outputs = model(input_tensor, training=False)
            bcv = 16 * x * (1 - x) * z * (1 - z)

            stream = -bcv * bcv * outputs[:, 0]
            stream = tf.reshape(stream, (1, -1))
            u = tape.gradient(stream, z)
            stream_x = tape.gradient(stream, x)
            w = -1 * stream_x
            T_hard = 1 - (1 - 0) * x
            T = T_hard + tf.sin(np.pi * x) * outputs[:, 1]
            T = tf.reshape(T, (1, -1))

            vort = -1 * (tape.gradient(u, z) + tape.gradient(stream_x, x))

            stream = tf.reshape(stream, (-1, 1))
            T = tf.reshape(T, (-1, 1))
            u = tf.reshape(u, (-1, 1))
            w = tf.reshape(w, (-1, 1))
            vort = tf.reshape(vort, (-1, 1))
            answers = tf.concat((stream, T, u, w, vort), axis=1)

        dataPlot2d = answers.numpy()
        listData2d.append(dataPlot2d)
        raval = (np.log10(plotsWanted3D[ra])-3)/3
        Ra = tf.Variable([raBase[:, 0] * raval], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            input_tensor3d = tf.transpose(tf.concat([x, z, Ra], 0))
            outputs = model3D(input_tensor3d, training=False)
            bcv = 16 * x * (1 - x) * z * (1 - z)

            stream = -bcv * bcv * outputs[:, 0]
            stream = tf.reshape(stream, (1, -1))
            u = tape.gradient(stream, z)
            stream_x = tape.gradient(stream, x)
            w = -1 * stream_x
            T_hard = 1 - (1 - 0) * x
            T = T_hard + tf.sin(np.pi * x) * outputs[:, 1]
            T = tf.reshape(T, (1, -1))

            vort = -1 * (tape.gradient(u, z) + tape.gradient(stream_x, x))

            answers3d = tf.concat((stream, T, u, w, vort), axis=1)

            stream_ra = tape.gradient(stream, Ra)

            derivScaleRa = 1 / (10 ** (raval * 3 + 3))
            T_ra = tape.gradient(T*derivScaleRa, Ra)
            u_ra = tape.gradient(u*derivScaleRa, Ra)
            w_ra = tape.gradient(w*derivScaleRa, Ra)
            vort_ra = tape.gradient(vort*derivScaleRa, Ra)

            answers3dGrads = tf.concat((stream_ra, T_ra, u_ra, w_ra, vort_ra), axis=1)
        dataPlot3d = answers3d.numpy()
        listData3d.append(dataPlot3d)
        dataPlot3dGrads = answers3dGrads.numpy()
        listData3dGrads.append(dataPlot3dGrads)

    nRaPoints = len(listData3d)
    
    plotNameList = [r"$\Psi$",r"$T$",r"$u$",r"$w$"]

    nPlots = len(plotNameList)

    plt.figure()
    for ii in range(nRaPoints):
        for jj in range(nPlots):
            plt.subplot(2, 2, jj+1)
            plt.plot(plotPoints, listData2d[ii][:,jj],label=f"{raKeyList[ii]}")

    for jj in range(nPlots):
        plt.subplot(2, 2, jj+1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")
        plt.title(plotNameList[jj])
        plt.legend()
        plt.grid()


    plt.suptitle(r"$z = 0.5$")
    plt.tight_layout()
    plt.show()
