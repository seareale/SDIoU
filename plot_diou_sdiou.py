# SDIoU(Stadardized Distance-based IoU)
# Written by seareale
# https://github.com/seareale

"""
plot_ioudist(dist_diou, idx=0)
plot_ioudist(dist_sdiou_mean, idx=0)
plot_ioudist(dist_sdiou_var, idx=0)
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def dist_(d, theta, h, w, h_,w_):
    h_d = np.abs(d*np.sin(theta))
    w_d = np.abs(d*np.cos(theta))
    dist = ((w_d)**2 + (h_d)**2)
    
    return dist

def dist_diou(d, theta, h, w, h_,w_):
    h_d = np.abs(d*np.sin(theta))
    w_d = np.abs(d*np.cos(theta))
    h_a = (h+h_)/2+h_d
    w_a = (w+w_)/2+w_d
    dist = ((w_d)**2 + (h_d)**2)/(w_a**2 + h_a**2)
    
    return dist

def dist_sdiou(d, theta, h, w, h_,w_):
    h_d = np.abs(d*np.sin(theta))
    w_d = np.abs(d*np.cos(theta))
    h_a = (h+h_)/2
    w_a = (w+w_)/2
    # dist = (w_d**2 + h_d**2)/(w_a**2 + h_a**2)
    dist = w_d**2/w_a**2 + h_d**2/h_a**2
    
    return dist/2

def dist_sdiou_var(d, theta, h, w, h_,w_):
    h_d = np.abs(d*np.sin(theta))
    w_d = np.abs(d*np.cos(theta))
    h_a = (h**2+h_**2)/2
    w_a = (w**2+w_**2)/2
    dist = w_d**2/w_a + h_d**2/h_a
    
    return dist/2

def dist_init(idx=0):
    d=100
    theta = math.pi/4
    gt = np.array([[100,100], [200, 100], [100,200],[600,600], [1200, 600], [600,1200]])
    pred = np.array([[100,100], [200, 100], [100,200],[600,600], [1200, 600], [600,1200]])
    h, w = gt[idx]
    h_, w_ = pred[idx]
    
    return d, theta, h, w, h_, w_

def plot_ioudist(metric, idx=0):
    _, ax = plt.subplots(1, 3, figsize=(20,5))

    d, theta, h, w, h_, w_ = dist_init(idx)
    
    offset = min(h,w) / 10
    d = d * offset / 10
    offset=10
    
    theta = np.arange(0, 360, 1.)
    theta *= (math.pi / 180)
    dist = metric(d, theta, h, w, h_,w_)
    [ax[i].plot(theta, dist, '--', c='r') for i in range(3)]

    for i in range(3):
        if i == 0:
            rate = h_ / w_
            h_c = np.arange(h_-5*offset*rate, h_+5*offset*rate, offset*rate)    
            w_c = np.arange(w_-5*offset, w_+5*offset, offset)
        elif i == 1:
            w_c = np.arange(w_-5*offset, w_+5*offset, offset)
            h_c = [h_ for _ in np.arange(len(w_c))]
        elif i == 2:
            h_c = np.arange(h_-5*offset, h_+5*offset, offset)
            w_c = [w_ for _ in np.arange(len(h_c))]
            
        for h_v, w_v in zip(h_c,w_c):
            if h_v == h_ and w_v == w_:
                continue
            dist = metric(d, theta, h, w, h_v,w_v)
            ax[i].plot(theta, dist, c='k')         
    for i in range(3):
        ax[i].set_xticks(np.arange(0,361*(math.pi/180),45*(math.pi/180)))
        ax[i].set_xticklabels(np.arange(0, 361,45).astype(np.int32))
        ax[i].set_xlabel('Degree')
        if metric == dist_diou:
            ax[i].set_ylabel('$R_{DIoU}$', fontsize=13)
        elif metric == dist_sdiou:
            ax[i].set_ylabel('$R_{SDIoU-mean}$', fontsize=13)
        elif metric == dist_sdiou_var:
            ax[i].set_ylabel('$R_{SDIoU-var}$', fontsize=13)
           
