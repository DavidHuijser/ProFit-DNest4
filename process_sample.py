#!/usr/bin/python
from pylab import *

import colormaps as cmaps

import dnest4
import dnest4.classic as dn4
#dn4.postprocess(single_precision=True)
#import display
import numpy as np
import pyfits

import corner

import os
import sys
import pylab
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches #for rectancles
from collections import Counter

from matplotlib._png import read_png
import matplotlib.cbook as cbook
from matplotlib.cbook import get_sample_data

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

from scipy.misc import imread

from copy import deepcopy

import scipy
from sklearn import metrics
from scipy.ndimage import rotate
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

import collections

import corner
posterior = True
#posterior = Falsez
Images = False
Postprocess = False

# Number of bins
bins = 100 #
data_name  = 'G575285'

print("Version 1.0")

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.set_cmap(cmaps.viridis)
name = 'viridis'
# setttings like colors and constants
cluster_colors = ['r', 'g', 'b', 'c', 'm']
markers = [",","v","^","<",">"]

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def DaviesBouldin(X, labels):
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []
    
    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))

    return(np.max(db) / n_cluster)


# this find the number of sererate peaks
def find_max_list(hist,xxx,yyy, top, min_dist):
    x = xxx[:,0] # x-coordinates
    y = yyy[0,:] # x-coordinates
    print(type(xxx), type(yyy))
    new_hist = hist.reshape(len(hist[0])*len(hist))
    dev = np.std(new_hist)
    # new_hist = hist
    # obtained the index of the sorted histogram values
    # reverse to get the higstest value first
    index = np.argsort(new_hist)[::-1]
    ordered = new_hist[index]
    x_values = np.zeros(top)
    y_values = np.zeros(top)
    hist_list = np.zeros(top)
    #print("ordered",new_hist[index])
    full = False
    # obtain the first object
    hist_list[0] = ordered[0]
    x_values[0] = x[index[0] / len(x)]
    y_values[0] = y[index[0] % len(x)]
    print("Peak at (x,y) = (%d, %d) has frequency %d" % (x[index[0] / len(x)],x[index[0] % len(x)],ordered[0]))
    #print("histogram value",ordered[0])
    #print("xcoordinate", x[index[0] / len(x)])
    #print("ycoordinate",  x[index[0] % len(x)])
    i = 1    #
    j = 1    #
    
    while full == False:
            print("histogram value",ordered[j])
            # print("xcoordinate", x[index[j] / len(x)])
            # print("ycoordinate",  x[index[j] % len(x)])
            #print("histogram value (check)", hist[index[i] / len(x), index[i] % len(x) ])
            x_values[i] = x[index[j] / len(x)]
            y_values[i] = y[index[j] % len(x)]
            hist_list[i] = ordered[j]
            accept = True
            point2 = np.array([x_values[i], y_values[i]], dtype=np.float32)
            for counter in range(0,i):
               point1 = np.array([x_values[counter], y_values[counter]],dtype=np.float32)
               distance =  np.linalg.norm(point1 - point2)
               #print("distance", distance, point1, point2)
               if (distance < min_dist):
                     accept= False
                     print("rejected")
            if i == top:
                 full=True
                 print("****** set completely filled")
            if j== len(ordered)-1:
                 full=True
                 x_values = x_values[0:i]
                 y_values = y_values[0:i]
                 hist_list = hist_list[0:i]
                 print("****** not enough separate peaks, this algorithm fails ******")
                 print("****** take only subset; number of found peaks: ", i)
            if hist_list[i] < var(ordered):   # if the peaks are just grass
                full=True
                print("****** the remaining peaks or not high enough", hist_list[i])
                x_values = x_values[0:i]
                y_values = y_values[0:i]
                hist_list = hist_list[0:i]
                print("***", len(hist_list))
                print("****** take only subset; number of found peaks: ", i)
            if accept == True:
                i=i+1
                print("accepted")
            j=j+1
    
    for i in range(0,len(x_values)):
        print("Peak at (x,y) = (%d, %d) has frequency %d" % (x[index[i] / len(x)],x[index[i] % len(x)],ordered[i]))
        #print("histogram value",hist_list[i])
        #print("xcoordinate", x_values[i])
#print("ycoordinate",  y_values[i])
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    # ax.set_xlim(0, ysize)
    # ax.set_ylim(0, xsize)
    # ext = [0, ysize,0, xsize]
    # imshow(np.transpose(data), extent=ext,interpolation='nearest',cmap=plt.get_cmap(name))
       #imshow(data, interpolation='nearest',extent=ext,  vmin=-0.1*temp.max(), vmax=temp.max())
       # plt.scatter(f1, f2,  c='grey' , alpha=0.2)
       #  for i in range(0,len(x_values)):
       #     plt.scatter(x_values[i], y_values[i],  c='red', alpha=0.7,s=80,marker='*')
              #plt.scatter(x_values[i], y_values[i], marker='x', color='red')
              # plt.text(x_values[i]+0.3, y_values[i]+0.3, type, fontsize=9)
              #     ax.annotate((i+1), (x_values[i], y_values[i]),fontsize=20, color='red')
              #  gca().set_xticks([])
              #  gca().set_yticks([])
              #  axis( [0,ysize,0,xsize])
              
              
    points, sub = hist2d_bubble(x_values,y_values,bins=bins)
    filename = data_name + '_2D_scatterplot.png'
    plt.savefig(filename)
    plt.close(fig)
    return np.array(list(zip(x_values, y_values)))


#Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def calc_CH_index(X,C,k, clusters,reduced):
    print("--Calculating CH index--")
    # print("X", np.shape(X))
    #print("cluster",np.shape(clusters))
    #print("k",k)
    #if reduced == False:
        #print("unreduced", len(X))
        #   X = X[clusters != k]
        #else:
        #print("red:",len(X))
    n = len(X)
    #    print("number of points is",n)
    g = [np.mean(X[:,0]),np.mean(X[:,1])]
    #print("mean g",np.mean(X[:,0]),np.mean(X[:,1]))
    #print(g)
    # print(g,C[0,:])
    teller = 0.0
    for i in range(0,k):
        teller = teller +  dist(g ,C[i],ax=0)
    #   print(np.linalg.norm(g - C[i]))
    teller = teller / (k-1)
    noemer = 0.0
    for i in range(0,k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        for j in range(0,len(points)):
            noemer = noemer + dist(points[j],C[i],ax=0)
    noemer = noemer / (len(X) - k)
    CH = teller/noemer
    print("--Finished Calculating CH index--")
    return CH

# get all names uniquely
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def rev(a, axis = -1):
    a = np.asarray(a).swapaxes(axis, 0)
    a = a[::-1,...]
    a = a.swapaxes(0, axis)
    return a


def find_index_str(array, string):
    index = np.empty(len(array), dtype=int)
    for i in range(0, len(array)):
         if  (array[i] == string):
            index[i] = True
         else: 
            index[i] = False
    return index


def list_to_array_str(old_list):
    new_array = ["" for x in range(len(old_list))]
#    new_array = np.empty(len(old_list))
    for i in range(0, len(new_array)):
               new_array[i] = old_list[i]
    return new_array


def list_to_2Darray(old_list):
    dim1  = len(old_list)
    dim2  = len(old_list[0])
    new_array = np.zeros(dim1*dim2,dtype=double).reshape(dim1,dim2)
    for i in range(0, dim1):
               for j in range(0, dim2):
                 new_array[i,j] = old_list[i][j]
    return new_array



# 2D histogram function
def hist2d_bubble(x_data, y_data, bins=bins):
    xsize =  data_shape[1]         
    ysize =  data_shape[0]
    
    bins = 20

    ax = np.histogram2d(x_data, y_data, bins=bins, weights= None, range=[[0, xsize], [0,ysize]])
    xs = ax[1]
    dx = xs[1] - xs[0]
    ys = ax[2]
    dy = ys[1] - ys[0]
    def rdn():
        return (1-(-1))*np.random.random() + -1
    points = []
    for (i, j),v in np.ndenumerate(ax[0]):
  #       points.append((ys[i], xs[j], v))
         points.append((xs[i], ys[j], v))
        
    points = np.array(points)

    major_ticksx = np.arange(0, xsize, 20)
    major_ticksy = np.arange(0, ysize, 20)                                                                                            
    
    binx = xs[1] - xs[0]
    biny = ys[1] - ys[0]

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0,xsize)
    ax.set_ylim(0,ysize)
    imshow(data, cmap='jet', interpolation='nearest')

#    sub = pyplot.scatter(points[:, 0],points[:, 1], color='black', marker='o', s=128*points[:, 2], alpha=0.2)
#    sub = pyplot.scatter(points[:, 0]+0.5*binx,points[:, 1]+0.5*biny, color='black', marker='o', s=128*points[:, 2], alpha=0.2)
    sub = pyplot.scatter(points[:, 0]+0.5*binx,points[:, 1]+0.5*biny, color='black', marker='o', s=12*points[:, 2], alpha=0.2)
    plt.scatter(x_data, y_data,  c='w' , alpha=0.5)

    sub.axes.set_xticks(major_ticksx)
    sub.axes.set_yticks(major_ticksy)

    sub.axes.set_xticks(xs, minor=True)
    sub.axes.set_yticks(ys, minor=True)

    pyplot.grid(which='both')                                                            

    # or if you want differnet settings for the grids:                               
    pyplot.grid(which='minor', alpha=1.0, c='b')                                                
    pyplot.grid(which='major', alpha=0.2) 

    return points, sub

# this function detects the local maximal above a certain threshhold and returns and array with the locations of the maxima
# the neighborhood_size is inverse, so the higher the number the smaller number of points

#def detect_local_maxima(data, threshold,neighborhood_size):
#neighborhood_size = 5
#threshold = 100
#   data_max = filters.maximum_filter(data, neighborhood_size)
#  maxima = (data == data_max)
#  data_min = filters.minimum_filter(data, neighborhood_size)

#diff = ((data_max - data_min) > threshold)
#  maxima[diff == 0] = 0

#  labeled, num_objects = ndimage.label(maxima)
#  xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

#  return xy



#set starter time
start = time.time()


# post process.
if Postprocess:
   dnest4.postprocess(single_precision=True)

# load data from fits files 
print("loading data and samples")
hdulist = pyfits.open('Data/fitim.fits')
data = hdulist[0].data     # assuming the first extension is a table
data_shape = data.shape   
xsize =  data_shape[0]     # obtain x-dimension of original image  (92 wide)
ysize =  data_shape[1]     # obtain y-dimension of original image  (101 high)

# load sigma file
hdulist2 = pyfits.open('Data/sigma.fits')
sigma = hdulist2[0].data # assuming the first extension is a table

print(min(map(min, data)),max(map(max, data)))
from matplotlib.colors import LogNorm
data = np.abs(data)
#store data as png image
norm = matplotlib.colors.LogNorm(vmin=min(map(min, data)), vmax=max(map(max, data)))
fig,ax = plt.subplots(1)
plt.imshow(data, norm=LogNorm(), cmap='viridis')
plt.savefig(data_name+'.png', bbox_inches = 'tight')
plt.close(fig)

# load sample or posterior
if (posterior == True):
  posterior_sample = dn4.my_loadtxt("posterior_sample.txt")
  indices = dn4.load_column_names("posterior_sample.txt")
else:  
  posterior_sample = dn4.my_loadtxt("sample.txt")
  indices = dn4.load_column_names("sample.txt")

# obtain the indices / column names from the object
I = indices.get('indices')                              # I is a dictionary
Z = indices.get('colnames')                             # Z is a list

S =  posterior_sample.shape[0]  #  number of samples
L =  posterior_sample.shape[1]  #  total length of array parameters + hyperparameters + image 

# Print some output info 
print("The dimensions of the image are: ", xsize , " x " , ysize)
print("Posterior shape: ", posterior_sample.shape)
print("Length: ", posterior_sample.shape[1])
print("Sample Size: ", posterior_sample.shape[0])
print("Starting Position of image: ", posterior_sample.shape[1] - xsize*ysize)
starting = posterior_sample.shape[1] - xsize*ysize

image = posterior_sample[0,starting:posterior_sample.shape[1]].reshape(xsize,ysize)

#starter index for image (where the image starts in the store array)
start_index = posterior_sample.shape[1] - xsize*ysize
ndim = start_index
#final index  for image (where the image ends in the store array)
final_index = posterior_sample.shape[1]

# obtain starter indices for each type of parameter
all_names =  np.asarray(['x', 'y', 'mag','Re','n', 'q','theta', 'boxi', 'mag-bar', 'Rout',  'a', 'b', 'q-bar', 'theta-bar', 'box-bar'])

fancy_names =  np.asarray(['x', 'y', '$I$','$R_e$','$n$', '$q$','$\\theta$', 'boxi', '$I_{bar}$', '$R_{out}$',  '$a$', '$b$', '$q_{bar}$', '$\\theta_{bar}$', 'box-bar'])


index = np.zeros(final_index,dtype=bool)
sample_index = np.ones(len(all_names),dtype=bool)

for i in range(len(all_names)):
    if all_names[i] == 'boxi' or all_names[i] == 'box-bar' or all_names[i] == 'x' or all_names[i] == 'y':
          index[i] = False
          sample_index[i] = False
    else:
          index[i] = True

#print(all_names[index])
print("------------------- DISPLAY IMAGE -----------------------")
#print("Start index image: ",start_index," End index image: ", final_index, " length of image interval " , final_index - start_index)

#img = np.array(img).reshape(xsize,ysize)
#all_names = set(Z[0: int(start_index)])

#imshow(img-data, cmap='jet', interpolation='nearest')
#show()

#imshow(image-data, cmap='jet', interpolation='nearest')
#show()


# define ranges for plots
ranges = np.zeros(2*ndim).reshape(ndim, 2)  # [min,max]
ranges[0,0] = 0                  #x
ranges[0,1] = xsize
ranges[1,0] = 0                  #y
ranges[1,1] = ysize
ranges[2,0] = -20                #mag
ranges[2,1] = 50
ranges[3,0] = 1                   #Re
ranges[3,1] = 50
ranges[4,0] = 0.5                 #n
ranges[4,1] = 10
ranges[5,0] = 0                     #q
ranges[5,1] = 1
ranges[6,0] = 0                     #theta
ranges[6,1] = 180
ranges[7,0] = -1                    #boxi
ranges[7,1] = 1

ranges[8,0] = 0                   #mag bar
ranges[8,1] = 50
ranges[9,0] = 0                    # rout
ranges[9,1] = 50
ranges[10,0] = 0                   #a 
ranges[10,1] = 10 
ranges[11,0] = 0                   #b
ranges[11,1] = 2
ranges[12,0] = 0                   #q	
ranges[12,1] = 1
ranges[13,0] = 0                   # theta
ranges[13,1] = 180
ranges[14,0] = -1                   #boxi
ranges[14,1] = 1


print(" \n ")
print(all_names)
print(all_names[sample_index])
samples = posterior_sample[:,index]
#corner_ranges = [(18.,19.), (10.5,11.5),(0.4,0.6), (0.6,0.8),(60.,90.), (18.,19),(2.5,3.0), (3.,4),(1.,2),(0.3,0.4), (40,45)]
corner_ranges = [(1.0), (1.0),(0.49,0.52), (1.0),(1.0), (1.0),(2.7, 3.1), (1.0),(1.8,1.92),(1.0), (1.0)]
print(np.shape(samples),len(all_names[sample_index]))

figure(figsize=(14, 14))
fig = corner.corner(samples, range=corner_ranges, labels=fancy_names[sample_index],plot_contours=False,quantiles=(0.16,0.5, 0.84), labels_kwargs={"fontsize": 14}, title_kwargs={"fontsize": 14},show_titles=False, hist2d_kwargs={"plot_contours": False,"quiet":True,"fill_contours": False,"no_fill_contours" : True})
#fig = corner.corner(samples, labels=fancy_names[sample_index],plot_contours=False, labels_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 14},quantiles=(0.16,0.5, 0.84),show_titles=False, levels=(0.68,0.9,0.95), hist2d_kwargs={"plot_contours": False,"quiet":True})
fig.savefig(data_name + '_triangle.pdf', bbox_inches='tight', dpi=300)
fig.savefig(data_name + '_triangle.eps', bbox_inches='tight', dpi=300)
fig.savefig(data_name + '_triangle.png', bbox_inches='tight', dpi=300)
close()

# create cornerplots
#                'mag',   'Re',  'n',       'q' 'theta'  'mag-bar' 'Rout',  'a', 'b', 'q-bar', 'theta-bar',
corner_ranges = [(1.0), (1.0),(0.49,0.52), (1.0),(1.0), (1.0),(2.7, 3.1), (1.0),(1.8,1.92),(1.0), (1.0)]

figure(figsize=(14, 14), dpi=300)
fig = corner.corner(samples, labels=fancy_names[sample_index],show_titles=True, title_kwargs={"fontsize":  12}, \
label_kwargs={"fontsize": 14},quantiles=(0.16, 0.5, 0.84),plot_contours=(False),plot_density=(False),\
plot_datapoints=(True),color=("black"),data_kwargs ={"marker":".","ms":4**2, "alpha": \
 0.3},range=corner_ranges )
fig.savefig(data_name + "_triangle_new.pdf", bbox_inches="tight", dpi=300)
fig.savefig(data_name + "_triangle_new.eps", bbox_inches="tight", dpi=300)
fig.savefig(data_name + "_triangle_new.png", bbox_inches="tight", dpi=300)
close()


stop


# Making histograms, loop over all parameters, and excluding the zero-padding 
print("Creating histograms ")
for i in range(0,ndim):
      figure(figsize=(14, 9))
     # print(" \n ")
     # print(all_names[i],   " ", Z.index(all_names[i]))
     # this_index = Z.index(all_names[i])
      # if it is one of the basic statistics
      title = 'Histogram ' +all_names[i]
      plt.title(title)
      plt.xlabel(all_names[i])
      plt.grid(True)
      plt.hist(posterior_sample[:,i], alpha=0.5, bins=50)
      filename = 'Images/'+data_name + '_histogram_' +all_names[i] +'.png'
      plt.savefig(filename)
      plt.close()

if Images:
# start to make images
  print(" ---------------- STARTING TO GENERATE SAMPLE IMAGES -----------------")
  figure(figsize=(14, 9))
  for i in range(0, posterior_sample.shape[0]):
    clf()
    image = posterior_sample[i,starting:posterior_sample.shape[1]].reshape(xsize,ysize)
    
    subplot(1,3,1)
    #title_string = "Model Source " + str(i+1)
    plt.imshow(image, interpolation='nearest', cmap='viridis',origin='lower')
    plt.title('Model Source ' +  '%d'%(i+1))
    
    subplot(1,3,2)
    plt.imshow(data, interpolation='nearest', cmap='viridis',origin='lower')
    
    subplot(1,3,3)
    plt.imshow(((image - data)/sigma),origin='lower', cmap='jet')
    draw()
    
    plt.savefig('Frames/' + '%0.6d'%(i+1) + '.png', bbox_inches='tight')
    print('Frames/' + '%0.6d'%(i+1) + '.png')





# creating a 2D histogram imposed on plot
upper = posterior_sample.shape[0]
total_objects = int(np.sum(posterior_sample[0:upper,num_comp_index]) )# 'total number of objects across the entire sample'
print(total_objects)
fig = plt.figure()
f1 = np.empty(total_objects)
f2 = np.empty(total_objects)
counter = 0

for i in range(0,upper):
    number  = int(posterior_sample[i,5])  # obtain number of component in current particular sample
    #print(number)
    #create x and y subset
    x_index = Z.index('x') 
    xsubset = posterior_sample[i,x_index:x_index+number]
  
    y_index = Z.index('y') 
    ysubset = posterior_sample[i,y_index:y_index+number]

#    print(number,xsubset)
    if (number > 1):                            # add all occurances in one long are
      for j in range(0, number):
           f1[counter] = xsubset[j]
           f2[counter] = ysubset[j]
           counter = counter + 1
    else:
           f1[counter] = xsubset[0]
           f2[counter] = ysubset[0]
           counter = counter + 1

# i = sample point
# j = parameter
# l = sub-sample points

fx = np.empty(total_objects*15).reshape(total_objects,15)
print(np.shape(fx))    # (2045, 15)
print(all_names)
for j in range(0,15):
    name = all_names[j+6]
    print(name)
    index = Z.index(name)
    counter = 0
    for i in range(0,upper):
         number  = int(posterior_sample[i,5]) # obtain number of component in current particular sample
         subset = posterior_sample[i,index:index+number]
         if (number > 1):                            # add all occurances in one long are
             for l in range(0, number):
                 fx[counter,j] = subset[l]
                 counter = counter + 1
         else:
                 fx[counter,j] = subset[0]
                 counter = counter + 1


print("x", min(fx[:,0]), max(fx[:,0])  )
print("y", min(fx[:,1]), max(fx[:,1])  )

print("x", min(f1), max(f1), xsize)
print("y", min(f2), max(f2), ysize)

figure(figsize=(14, 9))
imshow(data,interpolation='nearest',cmap=plt.get_cmap('viridis'))
#plot(all_substructures_x, all_substructures_y, 'w.', markersize=3, alpha=0.1)
sc = plt.scatter(f1, f2, s=1, cmap=plt.cm.get_cmap('viridis'),color='red')
#cbar = plt.colorbar(sc, pad=0.025)
#cbar.set_label('Satellite Mass $[M_{\odot}]$', rotation=270,labelpad=25)
gca().set_xticks([])
gca().set_yticks([])
axis( [0,xsize,0,ysize])
#title('Substructure Positions')
savefig(data_name + 'position.pdf', bbox_inches='tight')
savefig(data_name + 'position.eps', bbox_inches='tight')
savefig(data_name + 'position.png', bbox_inches='tight')
close()


X = np.array(list(zip(f1, f2)))

#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#img = image
#x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
#print(np.shape(x),np.shape(y))
#tens = np.ones(len(x)*len(y)).reshape(len(x),len(y))
#norm = matplotlib.colors.LogNorm(vmin=min(map(min, img)), vmax=max(map(max, img)))
#ax.plot_surface(x, y, img, facecolors=plt.cm.jet(norm(img)))
#m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
#m.set_array([])
#plt.colorbar(m)
#plt.savefig('testme.png', dpi=1000)
#plt.close(fig)

#second plot
img = np.transpose(img)
# if number of bins is equal to number of pixels the image can be overlaid on top of image
xx, yy = ogrid[0:img.shape[0], 0:img.shape[1]]
#X = xx
#Y = yy
Z1 = -5*np.ones(xx.shape)
#Z = np.cos(xx/10) * np.cos(xx/10) + np.sin(yy/10) * np.sin(yy/10)
fig = plt.figure()
ax = fig.gca(projection='3d')
norm = matplotlib.colors.LogNorm(vmin=min(map(min, img)), vmax=max(map(max, img)))
ax.plot_surface(xx, yy,img, rstride=1, cstride=1, facecolors=plt.cm.jet(norm(img)), shade=False)
#surf = ax.plot_surface(X, Y, img, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.axis('off')
plt.savefig('testme2.png', dpi=1000)
plt.close(fig)


# make histogram (xsize and ysize are interchange because
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
histo, xedges, yedges = np.histogram2d(f1, f2, bins=(bins,bins),range=[[0,ysize],[0,xsize]])
xpos, ypos = np.meshgrid(xedges[:-1] , yedges[:-1])

xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)
# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = histo.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.savefig(data_name + '_2DHistogram.png', bbox_inches='tight')
plt.close(fig)

xxx = np.reshape(xpos,(bins,bins))
yyy = np.reshape(ypos, (bins,bins))
zzz = np.reshape(dz, (bins,bins))

fig = plt.figure()
fig.clf()
ax = plt.axes(projection='3d')
#ax.set_xlim(0,ysize)
#ax.set_ylim(0,xsize)

#img=scipy.misc.imresize(rotate(img, -90), (bins, bins))
#img = rotate(img, -90)


print(np.shape(img),np.shape(xxx),np.shape(yyy),np.shape(zzz))
norm = matplotlib.colors.Normalize(vmin=min(map(min, img)), vmax=max(map(max, img)))
#norm = matplotlib.colors.LogNorm(vmin=min(map(min, img)), vmax=max(map(max, img)))
#ax.plot_surface(xxx, yyy, zzz,linewidth=0, antialiased=False,rstride=1, cstride=1,cmap='jet',facecolors=plt.cm.jet(norm(img)), edgecolor='none')
ax.plot_surface(xxx, yyy, zzz,linewidth=0, antialiased=False,rstride=1, cstride=1,cmap='jet', edgecolor='none')
surf = ax.plot_surface(xxx, yyy, zzz, cmap='jet', linewidth=0, antialiased=False)

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(zzz)
plt.colorbar(m)
plt.savefig(data_name + '2Durface_plot.png', bbox_inches='tight')
plt.close(fig)

end = time.time()
print("Time from start from until end of main loop (in minutes", (end - start)/60)
# set stats constants
lower =  2                # lower limit clusters
upper = 15                # upper limit clusters
distance = 7.5            # distance between peak to generate initial points for algorithm
#threshold = 0.5*distance # cluster radius,is half of the minimal distance between two peaks

binlist= find_max_list(histo,xxx,yyy,upper,distance)

if len(binlist) < upper:
    print("List is too short")
    upper = len(binlist)+1

print("Shape of the binlist", shape(binlist))
k_values = np.arange(lower,upper,dtype=int)
CH_values = np.arange(lower,upper,dtype=float)
reduced_CH_values = np.arange(lower,upper,dtype=float)
davies_bouldin_values = np.arange(lower,upper,dtype=float)
silhouette_score_values = np.arange(lower,upper,dtype=float)




X = np.array(list(zip(f1, f2)))

#print(binlist[:,0],binlist[:,1])
for k in range(lower,upper):
    print("THE MAXIMUM NUMBER OF CLUSTERS IS:", k)
    # X coordinates of random centroids
    C_x = binlist[0:k,0]
    C_y = binlist[0:k,1]
    freq = np.zeros(k)
    #C_y = 100*np.ones(k) - C_y
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    #C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    print("Initial Centroids positions")
    for i in range(0,k):
        print("Object %s, (x, y)= (%4.2f, %4.2f)" % (i+1, C[i,0],C[i,1]))
    
    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)       # calc distance point-cluster
            cluster = np.argmin(distances)  # determin closest
            if distances[cluster] < distance:
                 clusters[i] = cluster
            else:
                 clusters[i] = k
        #print("x ",X[i],"distances ",distances, clusters[i])
       # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            #   print(points)
            freq[i] = sum(clusters == i)
            C[i] = np.mean(points, axis=0)
            #  print("cluster %d at (%f,%f), freq %d" % (i,C[i,0],C[i,1],freq[i]))
        error = dist(C, C_old, None)

# print(C)
#       print(C_old)
#       print(error)

# if min(freq) < 5:
#            lowest_index = min(xrange(len(freq)), key=freq.__getitem__)
#            print("Lowest index %d with %d itemes" % (lowest_index,freq[lowest_index] ))
             # replace smallest cluster
             #            C[lowest_index,:] = C[k+1]
             #upper = upper-1
             #print("cluster centres:" , C)                # 3 x2
             #print("cluster has length" , len(clusters))  # 3000
    #calculate CH coefficient
    counters=collections.Counter(clusters)
    print(counters.keys())
    print(counters.values())
    CH_values[lower-k] = calc_CH_index(X,C,k,clusters,False)
    reduced_CH_values[lower-k] = metrics.calinski_harabaz_score(X, clusters)
    #    davies_bouldin_values[lower-k] = DaviesBouldin(X, clusters)
    silhouette_score_values[lower-k] = metrics.silhouette_score(X, clusters, metric='euclidean')
    print("Final Centroids positions and unreduced sets")
    # Loop will run till the error becomes zero
    for i in range(k):
        #  print("cluster",i)
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
            print("Object %s, (x, y)= (%4.2f, %4.2f),  #Objects= %4.0f" \
        % (i, C[i,0],C[i,1],sum(clusters == i)))
    print("Outliers = %4.0f" % (sum(clusters == k)))
    
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:navy')
    ax.grid(color='w', linestyle='-', linewidth=2)
    #ax = fig.add_subplot(111)
#ax.set_xlim(metadata[2], metadata[3])
#    ax.set_ylim(metadata[4], metadata[5])
    name = 'viridis'
    #name = 'coolwarm'
    imshow(data[::-1], extent=[0, ysize,0,xsize],interpolation='nearest',cmap=plt.get_cmap(name))
    #    sub = pyplot.scatter(points[:, 0],points[:, 1], color='black', marker='o', s=128*points[:, 2], alpha=0.2)
    #    sub = pyplot.scatter(points[:, 0]+0.5*binx,points[:, 1]+0.5*biny, color='black', marker='o', s=128*points[:, 2], alpha=0.2)
   # sub = pyplot.scatter(points[:, 0]+0.5*binx,points[:,  1]+0.5*biny, color='white', marker='o', s=64*points[:, 2], alpha=0.5)
# plt.scatter(x_data, y_data,  c='grey' , alpha=0.5)
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        #print("New_Length points",type(points))
        m = markers[i / len(cluster_colors)]
        ax.scatter(points[:, 0], points[:, 1], s=7,marker=m, c=cluster_colors[i  % len(cluster_colors)],)
        ax.annotate(i+1, (C[i,0]+0.05, C[i,1]+0.05),fontsize=20,color='white')
    #print outliers
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == k])
    ax.scatter(points[:, 0], points[:, 1], s=6, c='y')
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    ax.set_xlim(0, ysize)
    ax.set_ylim(0, xsize)
    axis( [0,ysize,0,xsize])

    #gca().set_xticks([])
#   gca().set_yticks([])
#axis( [2*metadata[2],2*metadata[3],2*metadata[4],2*metadata[5]])
    plt.grid(True)
    savefig(data_name +"cluster_data_k=" + str(k) +".png", bbox_inches='tight')
    plt.close
    print("Length data",len(f1),len(f2))
    print("Length cluster",len(clusters))
    print("Length X",len(X))


print(reduced_CH_values)
#CH_values[lower-k] = calc_CH_index(X,C,k,clusters,True)
# calinsky-plot
fig = plt.figure()
ax = plt.axes()
#ax.plot(k_values , reduced_CH_values,c='red',label='reduced');
#ax.plot(k_values , CH_values,c='blue',label='original');
#ax.plot(k_values , davies_bouldin_values,c='green',label='Davies-Bouldin');
ax.plot(k_values , silhouette_score_values,c='orange',label='Silhouette');
plt.legend();
plt.savefig(data_name + "_CH_plot.png", bbox_inches='tight')
plt.close

end = time.time()
print("Time from start from until the end of the CH-part (in minutes)", (end - start)/60)
#print("k values  orginal values  reduced values" )

best_silout_val = k_values[max(xrange(len(silhouette_score_values)), key=silhouette_score_values.__getitem__)]
print("Best silout", best_silout_val, max(silhouette_score_values))


for i in range(0, best_silout_val):
   for j in range(0, 15):
      name = all_names[j+6]
      selection = fx[clusters == i,j]
      test = np.sort(selection)
      num = np.shape(test)[0]
      lower_index = int(16.0*num/100.)
      middle_index = int(50.0*num/100.)
      upper_index = int(84.0*num/100.)
      print(name +" of object %d:  %.4g + %.4g - %.4g" % (i+1,test[middle_index],(test[upper_index]-test[middle_index]),(test[middle_index] - test[lower_index]) ))
stop
#

for i in range(0, best_silout_val):
   print("Object %d" % (i+1))
   x_index = Z.index('x')
   xsubset = posterior_sample[i,x_index:x_index+number]
   xsubset= xsubset[clusters == i]

   y_index = Z.index('y')
   ysubset = posterior_sample[i,y_index:y_index+number]
   ysubset= ysubset[clusters == i]

   mag_index = Z.index('mag')
   magsubset = posterior_sample[i,mag_index:mag_index+number]

   Re_index = Z.index('Re')
   Resubset = posterior_sample[i,Re_index:Re_index+number]

   n_index = Z.index('n')
   nsubset = posterior_sample[i,n_index:n_index+number]

   q_index = Z.index('q')
   qsubset = posterior_sample[i,q_index:q_index+number]

   theta_index = Z.index('theta')
   thetasubset = posterior_sample[i,theta_index:theta_index+number]

   magbar_index = Z.index('mag-bar')
   magbarsubset = posterior_sample[i,magbar_index:magbar_index+number]
       
   Rout_index = Z.index('Rout')
   Routsubset = posterior_sample[i,Rout_index:Rout_index+number]

   a_index = Z.index('a')
   asubset = posterior_sample[i,a_index:a_index+number]

   b_index = Z.index('b')
   bsubset = posterior_sample[i,b_index:b_index+number]

   qbar_index = Z.index('q-bar')
   qbarsubset = posterior_sample[i,qbar_index:qbar_index+number]

   thetabar_index = Z.index('theta-bar')
   thetabarsubset = posterior_sample[i,thetabar_index:thetabar_index+number]


   #

#'x', 'y', 'mag', 'Re', 'n', 'q', 'theta', 'boxi', 'mag-bar', 'Rout', 'a', 'b', 'q-bar', 'theta-bar', 'box-bar'




# make plot for each
stop


for i in range(0, upper):
    #print(k_values[i],CH_values[i],reduced_CH_values[i])

    cluster_masses = [all_substructures_m[j] for j in range(len(X)) if clusters[j] == i]
    cluster_distances = [all_substructures_w[j] for j in range(len(X)) if clusters[j] == i]
    cluster_distances =  1000*np.asarray(cluster_distances)*(np.pi/(3600.*180.))*DL/mpc
    cluster_masses = np.asarray(cluster_masses)*Einstein_angle_factor/np.pi

    testhist, newbins, _ = plt.hist(cluster_masses, bins=50)
    logbins = np.logspace(np.log10(newbins[0]),np.log10(newbins[-1]),len(newbins))
    
    #figure(figsize=(14, 9))
    fig = plt.figure()
    subplots_adjust(hspace=0.4,wspace=0.4)
    clf()
    subplot(1,2,1)
    width=0.6
    hist(cluster_masses, bins=logbins, alpha=0.2, color="k",range =(1e7,1e12))
    xlabel('Substucture mass $M_{\odot}$')
    ylabel('Posterior Samples')
    #legend(loc="upper right",frameon=False,labelspacing=1, fontsize=20)
    plt.xscale('log')
    plt.grid(False)
    
    subplot(1,2,2)
    hist(cluster_distances, 50, alpha=0.5, color="k")
    xlabel('Radius [kpc]')
    ylabel('Posterior Samples')
    #title('Magnification = {a:.3f} +- {b:.3f}'.format(a=magnification.mean(), b=magnification.std()))
    plt.grid(False)
    plt.savefig(data_name +"Histo_clusters_k=" + str(i) +".png", bbox_inches='tight')
    plt.close

    num = len(cluster_masses)
    print("length", num)
    lower_index = int(16.0*num/100.)
    middle_index = int(50.0*num/100.)
    upper_index = int(84.0*num/100.)
    test = np.sort(cluster_masses)
    print("Satellite mass %.4g + %.4g - %.4g" % (test[middle_index]/1e9,(test[upper_index]-test[middle_index])/1e9,(test[middle_index] - test[lower_index])/1e9))
    test = np.sort(cluster_distances)
    print("Satellite radius %.4g + %.4g - %.4g" % (test[middle_index],(test[upper_index]-test[middle_index]),(test[middle_index] - test[lower_index])))




stop
fig = plt.figure()
fig.clf()
ax = plt.axes(projection='3d')
ax.set_xlim(0,ysize)
ax.set_ylim(0,xsize)
img=scipy.misc.imresize(rotate(img), (bins, bins))
norm = matplotlib.colors.Normalize(vmin=min(map(min, img)), vmax=max(map(max, img)))
#norm = matplotlib.colors.LogNorm(vmin=min(map(min, img)), vmax=max(map(max, img)))
#ax.plot_surface(xxx, yyy, zzz,linewidth=0, antialiased=False,rstride=1, cstride=1,cmap='jet', edgecolor='none')
surf = ax.plot_surface(xxx, yyy, zzz, cmap='jet', linewidth=0, antialiased=False,facecolors=img)

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(zzz)
plt.colorbar(m)
plt.savefig(data_name + '_2DSurface_plot.png', bbox_inches='tight')
plt.close(fig)





#


# scatter plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

stat_numbers = np.zeros(S*number_of_regions).reshape(S,number_of_regions)
print(stat_numbers.shape)
print("The first object is", posterior_sample[0,6:16])
for i in range(0,S):
      zz = np.transpose(posterior_sample[i,6:starting].reshape(15,10))
      print(zz.shape,posterior_sample[i,5] )                                                     # shape of z is (10, 15), which means 10 objects with 15 parameters.
      for j in range(0,int(NOO)):
          for k in range(0,number_of_regions):
              if (zz[j][0] >regions[k,0]) & (zz[j][0] <regions[k,1]) & (zz[j][1] >regions[k,2]) & (zz[j][1] <regions[k,3]):
	              if (k == 0):
        	              setA.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 1):
                              setB.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
			      temp = zz[j,:]
        	      if (k == 2):
                              setC.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 3):
        	              setD.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 4):
        	              setE.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 5):
        	              setF.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1

      print("stat numbers ",stat_numbers[i,:])
      if (stat_numbers[i,1] > 1 and stat_numbers[i,3] > 1 and stat_numbers[i,1] > stat_numbers[i,3] ):
          print("case 1")
          temp_numb = stat_numbers[i,1]
          temp_len = len(setB)
          print("lenB",len(setB))
          print("lenB[0]",len(setB[0][:]))

      if (stat_numbers[i,1] > 1 and stat_numbers[i,3] > 1 and stat_numbers[i,1] < stat_numbers[i,3] ):
          print("case 2")


      # scatter plot B and D
      if (stat_numbers[i,1] == stat_numbers[i,3]):
          print("case 3")
          temp_numb = stat_numbers[i,1]
          temp_len = len(setB)
          print("lenB",len(setB))
          print(temp_len - temp_numb, ' ' ,temp_len)
     #     print("lenB[0]",len(setB[0]))
          subsetB = setB
          print("lenB[0]",setB[int(temp_len-temp_numb):temp_len][:])

          subsetB = list_to_2Darray(setB[int(len(setB)-temp_numb):len(setB)][:])
          subsetD = list_to_2Darray(setD[int(len(setD)-temp_numb):len(setD)][:])
          print("print subset B:", subsetB.shape)
          print("print subset D:", subsetD.shape)
          ax1.scatter(subsetB[0,2], subsetD[0,2],  c='w' , alpha=0.75)
          ax1.scatter(subsetB[0,8], subsetD[0,8],  c='r' , alpha=0.75)
          ax2.scatter(subsetB[0,2]+subsetB[0,8], subsetD[0,2]+subsetD[0,8],  c='w' , alpha=0.75)
          ax1.set_xlabel('Mag Region B')
          ax1.set_ylabel('Mag Region D')

          ax2.set_xlabel('Mag Region B')
          ax2.set_ylabel('Mag Region D')


#           stop
                             # subsetA = setA[temp_numb
                              #subsetB = setB
                              #subsetC = setC
                              #3subsetD = setD
#             ax1.scatter(subsetB[:,2], subsetD[:,2],  c='w' , alpha=0.5)

 #            ax1.scatter(subsetB[:,8], subsetD[:,8],  c='r' , alpha=0.5)
         # ax2.scatter(zz[j,2],zz[j,8],  c='r' , alpha=0.5)

name1 = "scatter_plot_regions B-D mag AND  mag-bar.png"
name2 = "scatter_plot_regions B-D mag PLUS mag-bar.png"
fig1.savefig(name1)
fig2.savefig(name2)
plt.close(fig1)
plt.close(fig2)
stop

#make sample graphs
figure(figsize=(14, 9))
for i in range(0, posterior_sample.shape[0]):
    clf()
    n_substructures = posterior_sample[i,Z.index('num_components')]
    image = posterior_sample[i,starting:posterior_sample.shape[1]].reshape(xsize,ysize)
    
    subplot(2,2,1)
    plt.imshow(image, interpolation='nearest', cmap='viridis')
    plt.title('Model Source ' + str(i+1))
    
    subplot(2,2,2)
    plt.imshow(data, interpolation='nearest', cmap='viridis')
    
    subplot(2,2,3)
    plt.imshow(((image - data)/sigma))
    draw()
    
    savefig('Frames/' + '%0.6d'%(i+1) + '.png', bbox_inches='tight')
    print('Frames/' + '%0.6d'%(i+1) + '.png')

stop

points, sub = hist2d_bubble(x, y, bins=bins)
name1 = 'x'
name2 = 'y'
filename = '2D_histogram_'+name1 + '_'+ name2 +'.png'
plt.savefig(filename)
plt.close(fig)


stop 

# now we  group the data into different regions 
setA = []
setB = []
setC = []
setD = []
setE = []
setF = []

fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

stat_numbers = np.zeros(S*number_of_regions).reshape(S,number_of_regions)
print(stat_numbers.shape)
print("The first object is", posterior_sample[0,6:16])
for i in range(0,S):
      zz = np.transpose(posterior_sample[i,6:starting].reshape(15,10))
      print(zz.shape,posterior_sample[i,5] )                                                     # shape of z is (10, 15), which means 10 objects with 15 parameters.
      for j in range(0,int(NOO)):
          for k in range(0,number_of_regions):
              if (zz[j][0] >regions[k,0]) & (zz[j][0] <regions[k,1]) & (zz[j][1] >regions[k,2]) & (zz[j][1] <regions[k,3]):                   
	              if (k == 0):
        	              setA.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 1):                
                              setB.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
			      temp = zz[j,:] 	
        	      if (k == 2):
                              setC.append(zz[j,:])		
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 3):
        	              setD.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 4):
        	              setE.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1
        	      if (k == 5):
        	              setF.append(zz[j,:])
                              stat_numbers[i,k] =  stat_numbers[i,k]+1

      print("stat numbers ",stat_numbers[i,:])      
      if (stat_numbers[i,1] > 1 and stat_numbers[i,3] > 1 and stat_numbers[i,1] > stat_numbers[i,3] ):
          print("case 1")
          temp_numb = stat_numbers[i,1]
          temp_len = len(setB)
          print("lenB",len(setB))
          print("lenB[0]",len(setB[0][:]))

      if (stat_numbers[i,1] > 1 and stat_numbers[i,3] > 1 and stat_numbers[i,1] < stat_numbers[i,3] ):
          print("case 2")


      # scatter plot B and D 
      if (stat_numbers[i,1] == stat_numbers[i,3]):      
          print("case 3")
          temp_numb = stat_numbers[i,1]
          temp_len = len(setB)
          print("lenB",len(setB))
          print(temp_len - temp_numb, ' ' ,temp_len)
     #     print("lenB[0]",len(setB[0]))
          subsetB = setB
          print("lenB[0]",setB[int(temp_len-temp_numb):temp_len][:])

          subsetB = list_to_2Darray(setB[int(len(setB)-temp_numb):len(setB)][:])
          subsetD = list_to_2Darray(setD[int(len(setD)-temp_numb):len(setD)][:])
          print("print subset B:", subsetB.shape)
          print("print subset D:", subsetD.shape)              
          ax1.scatter(subsetB[0,2], subsetD[0,2],  c='w' , alpha=0.75) 
          ax1.scatter(subsetB[0,8], subsetD[0,8],  c='r' , alpha=0.75)
          ax2.scatter(subsetB[0,2]+subsetB[0,8], subsetD[0,2]+subsetD[0,8],  c='w' , alpha=0.75)
          ax1.set_xlabel('Mag Region B')
          ax1.set_ylabel('Mag Region D')

          ax2.set_xlabel('Mag Region B')
          ax2.set_ylabel('Mag Region D')


#           stop
                             # subsetA = setA[temp_numb
                              #subsetB = setB
                              #subsetC = setC
                              #3subsetD = setD
#             ax1.scatter(subsetB[:,2], subsetD[:,2],  c='w' , alpha=0.5)

 #            ax1.scatter(subsetB[:,8], subsetD[:,8],  c='r' , alpha=0.5)
         # ax2.scatter(zz[j,2],zz[j,8],  c='r' , alpha=0.5)

name1 = "scatter_plot_regions B-D mag AND  mag-bar.png"
name2 = "scatter_plot_regions B-D mag PLUS mag-bar.png"                        
fig1.savefig(name1)
fig2.savefig(name2)
plt.close(fig1) 
plt.close(fig2) 



print(stat_numbers)
                     
#stop 
#        ('print subset B:', [array([  50.8678  ,   50.4411  ,   22.4319  ,   47.7954  ,    5.9418  ,
#          0.90962 ,  167.856   ,   -0.951724,   20.5116  ,    0.366183,
#          6.37106 ,    0.46685 ,    0.635491,   94.7367  ,    0.334602])])
#('print subset D:', [array([  4.24633000e+01,   4.18862000e+01,   2.27295000e+01,
#         4.08790000e+01,   8.33000000e+00,   8.28635000e-01,
#         1.25034000e+02,  -5.22513000e-02,   2.17868000e+01,
#         9.30582000e-01,   5.23939000e+00,   1.41949000e-01,
#         7.88904000e-01,   1.03616000e+02,  -9.17293000e-01])])       

#print(setC)
print(len(setA))
print(len(setB))
print(len(setC))
print(len(setD))
print(len(setE))
print(len(setF))



new = list_to_2Darray(setC)
print(new.shape)
#print(new[0])
#print(setC[0])


new_range = np.zeros(2)
print(ranges.shape)
print(new_range.shape)


# now we can make histogram of these regions
for k in range(0,number_of_regions):
        if k == 0:
                new = list_to_2Darray(setA)
        if k == 1:
                new = list_to_2Darray(setB)
   #                   plt.hist(newB[:,j], range=new_range, bins=30)
        if k == 2:
                new = list_to_2Darray(setC)
  #                    plt.hist(newC[:,j], range=new_range, bins=30)
        if k == 3:
                new = list_to_2Darray(setD)
 #                     plt.hist(newD[:,j], range=new_range, bins=30)
        if k == 4:
                new = list_to_2Darray(setE)
        if k == 5:
                new = list_to_2Darray(setF)


        for j in range(0,int(ndim)):

                new_range[0] = ranges[j,0]
                new_range[1] = ranges[j,1]
		fig = pyplot.figure()
		ax = fig.add_subplot(111)
#		ax.set_xlim(0,xsize)
#		ax.set_ylim(0,ysize)
            
		      #plt.hist(setD[:][j])
                plt.hist(new[:,j], range=new_range, bins=30)
#		plt.scatter(x_arr, y_arr,  c='w' , alpha=0.5)
                name = "Histogram_region: " + str(k) + " param " + all_names[6+j] + ".png"
                #ax.text(2, 6, write_str, style='italic',   bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})      
            
                plt.savefig(name)
                plt.close(fig) 
        

	# create regions x-y-scatter plot
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim(0,xsize)
	ax.set_ylim(0,ysize)
	imshow(data, cmap='jet', interpolation='nearest')
	plt.scatter(new[:,0], new[:,1],  c='w' , alpha=0.5)
	write_str = "Region: " + str(k) + " with " + str(len(new)) + " objects "
	ax.text(2, 6, write_str, style='italic',   bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})     
	name1 = 'x'
	name2 = 'y'
	filename = 'regions_'+ str(k) +'histogram_'+name1 + '_'+ name2 +'.png'
	plt.savefig(filename)
	plt.close(fig) 




fig = plt.figure()
ax = fig.add_subplot(111)
newA = list_to_2Darray(setA)
plt.scatter(newA[:,2], newA[:,8],  c='w' , alpha=1.0, label='Region A')
newB = list_to_2Darray(setB)
plt.scatter(newB[:,2], newB[:,8],  c='b' , alpha=1.0, label='Region B')
newC = list_to_2Darray(setC)
plt.scatter(newC[:,2], newC[:,8],  c='r' , alpha=1.0, label='Region C')
newD = list_to_2Darray(setD)
plt.scatter(newD[:,2], newD[:,8],  c='g' , alpha=1.0, label='Region D')
newE = list_to_2Darray(setE)
plt.scatter(newE[:,2], newE[:,8],  c='g' , alpha=1.0, label='Region E')
newF = list_to_2Darray(setF)
plt.scatter(newF[:,2], newF[:,8],  c='g' , alpha=1.0, label='Region F')


ax.set_xlabel('Mag')
ax.set_ylabel('Mag-Bar')
#legend((newA, newB, newC, newD),  ('Region A', 'Region B', 'Region C', 'Region D'),   scatterpoints=1,    loc='lower left',  ncol=3,   fontsize=8)
ax.legend()
name1 = 'mag'
name2 = 'mag-bar'
filename = 'scatter_'+name1 + '_'+ name2 +'.png'
plt.savefig(filename)
plt.close(fig) 



fig = plt.figure()
ax = fig.add_subplot(111)
newA = list_to_2Darray(setA)
plt.scatter(newA[:,0], newA[:,1],  c='w' , alpha=1.0, label='Region A')
newB = list_to_2Darray(setB)
plt.scatter(newB[:,0], newB[:,1],  c='b' , alpha=1.0, label='Region B')
newC = list_to_2Darray(setC)
plt.scatter(newC[:,0], newC[:,1],  c='r' , alpha=1.0, label='Region C')
newD = list_to_2Darray(setD)
plt.scatter(newD[:,0], newD[:,1],  c='g' , alpha=1.0, label='Region D')
newE = list_to_2Darray(setE)
plt.scatter(newE[:,0], newE[:,1],  c='g' , alpha=1.0, label='Region E')
newF = list_to_2Darray(setF)
plt.scatter(newF[:,0], newF[:,1],  c='g' , alpha=1.0, label='Region F')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='lower left')
name1 = 'x'
name2 = 'y'
filename = 'scatter'+name1 + '_'+ name2 +'.png'
plt.savefig(filename)
plt.close(fig) 


fig = plt.figure()
ax = fig.add_subplot(111)
newA = list_to_2Darray(setA)
plt.scatter(newA[:,0], newA[:,1],  c='w' , alpha=1.0, label='Region A')
newB = list_to_2Darray(setB)
plt.scatter(newB[:,0], newB[:,1],  c='b' , alpha=1.0, label='Region B')
newC = list_to_2Darray(setC)
plt.scatter(newC[:,0], newC[:,1],  c='r' , alpha=1.0, label='Region C')
newD = list_to_2Darray(setD)
plt.scatter(newD[:,0], newD[:,1],  c='g' , alpha=1.0, label='Region D')
newE = list_to_2Darray(setE)
plt.scatter(newE[:,0], newE[:,1],  c='g' , alpha=1.0, label='Region E')
newF = list_to_2Darray(setF)
plt.scatter(newF[:,0], newF[:,1],  c='g' , alpha=1.0, label='Region F')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='lower left')
name1 = 'x'
name2 = 'y'
filename = 'scatter'+name1 + '_'+ name2 +'.png'
plt.savefig(filename)
plt.close(fig) 



stop 

#['sigma', 'dim_components', 'max_num_components', 'hyper_location', 'hyper_scale', 'num_components', 'x', 'y', 'mag', 'Re', 'n', 'q', 'theta', 'boxi', 'mag-bar', 'Rout', 'a', 'b', 'q-bar', 'theta-bar', 'box-bar']
#print(x[0:9])
#print(y[0:9])


objects = np.zeros(number_of_regions*S*ndim*NOO).reshape(number_of_regions,S*NOO,ndim)      # shape( regions, maximum number of objects region, parameters) 
#print("Posterior sample", posterior_sample[0,6:start_index])
for i in range(0,number_of_regions):                                # loop over all regions
            counts = 0;                                              # number of objects found in this region
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(0,xsize)
            ax.set_ylim(0,ysize)
            imshow(data, cmap='jet', interpolation='nearest')
            for j in range(0,S):                                                         
                    for k in range(0, int(NOO)):					 # loop over all objects
#                             print(' ',posterior_sample[j,6:start_index-1] )					
                             x_val = posterior_sample[j,6+k]
                             y_val = posterior_sample[j,6+int(NOO)+k]
                         #    print('  (', x_val, ' ,', y_val, ')' ) 
                             if  (x_val > regions[i,0]) & (x_val < regions[i,1]) &  (y_val > regions[i,2]) & (y_val < regions[i,3]):
                              #      print('ACCEPTED: (', x_val, ' ,', y_val, ') falls in region: ', i) 
                                    for l in range(0,ndim):                                 # loop over all parameters (0 to 10)
                                          objects[i,counts,l] = posterior_sample[j,6+l*int(NOO)+k]
                                          #print('ACCEPTED: ', Z[6+l*int(NOO)+k],' ',posterior_sample[j,6+l*int(NOO)+k])
                                    counts =  counts+1
             
            print("Region ",i, " has ",counts, " counts")
            x_arr = np.zeros(counts)
            y_arr = np.zeros(counts)
            counter = 0
            if (counts > 1):                            
                for j in range(0, counts):
                    x_arr[counter] = objects[i,j,0]
                    y_arr[counter] = objects[i,j,1]
                    counter = counter + 1
            else:
                    x_arr[counter] = objects[i,j,0]
                    y_arr[counter] = objects[i,j,1]
                    counter = counter + 1

            #create x and y subset
#            plt.scatter(objects[i,j,0], objects[i,j,1],  c='w' , alpha=0.5)
            plt.scatter(x_arr, y_arr,  c='w' , alpha=0.5)
            write_str = "Region: " + str(i) + " with " + str(counts) + " objects "
            ax.text(2, 6, write_str, style='italic',   bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
            name1 = 'x'
            name2 = 'y'
            filename = 'regions_'+ str(i) +'histogram_'+name1 + '_'+ name2 +'.png'
            plt.savefig(filename)
            plt.close(fig) 



# FLUXES
fluxes = np.zeros(S*number_of_regions).reshape(S, number_of_regions)

# RECREATE IMAGES
print("Re-creating images ")
#for i in range(0,posterior_sample.shape[0]):
for i in range(0,S):
    number  = int(posterior_sample[i,5])  # obtain number of objects in this sample
#    print(number) 

    #create x and y subset
    x_index = Z.index('x') 
    xsubset = posterior_sample[i,x_index:x_index+number]
  
    y_index = Z.index('y') 
    ysubset = posterior_sample[i,y_index:y_index+number]


    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    imshow(data, cmap='jet')

    for l in range(0, number_of_regions):
        for p in [
             patches.Rectangle(
             (regions[l,0], regions[l,2]), regions[l,1]-regions[l,0], regions[l,3]-regions[l,2],
#            (0.1, 0.1), 0.3, 0.6,
             fill=False 
         ),
    ]:
         ax1.add_patch(p)

    plt.gca().invert_yaxis()

    ax2 = fig.add_subplot(222)
    img = posterior_sample[i, start_index : final_index ].reshape(xsize, ysize)
    imshow(img, cmap='jet', interpolation='nearest')

    
#$       matplotlib.pyplot.axhspan(regions[1,2], regions[1,3], xmin=regions[1,0], xmax=regions[1,1], hold=None)
    plt.gca().invert_yaxis()

    ax3 = fig.add_subplot(223)
    sigma = sqrt(sig**2 + posterior_sample[i,-1]**2)
    imshow(-(img - data)/sigma, cmap='coolwarm', interpolation='nearest')
    plt.gca().invert_yaxis()

    ax4 = fig.add_subplot(224)
    ax4.set_xlim(0,ysize)
    ax4.set_ylim(0,xsize)
    imshow(img, cmap='gray', zorder=1, interpolation='nearest')
    plt.scatter(xsubset, ysubset,  c='red', s=8, zorder=2)

    filename = 'new_multi_img'+ str(i) + '.png'
    plt.savefig(filename)
    plt.close(fig)


    # fill the fluxes 
    for j in range(0,number_of_regions):
         print(np.sum(img[regions[j,0]:regions[j,1],regions[j,2]:regions[j,3]]))
         fluxes[i,j]  = np.sum(img[regions[j,0]:regions[j,1],regions[j,2]:regions[j,3]])
         

#subset = subset[:, index]
print('Shape:', fluxes.shape)

#disc
print('Make temporary cornerplot for fluxes in 3 neighouring regions:')
labelz = ["$Region A$","$Region B$", "$Region C$", "$Region D$"]
fig = corner.corner(fluxes[:,:],labels=labelz,quantiles=None,plot_contours=True)
#fig = corner.corner(fluxes[:,1:4],labels=labelz,quantiles=None,plot_contours=False)
fig.savefig("CornerPlot_Regions.png")
close(fig)



stop



#subset = subset[:, index]
print('Shape:', fluxes.shape)

#disc
print('Make temporary cornerplot for fluxes in 3 neighouring regions:')
labelz = ["$Region B$", "$Region C$", "$Region D$"]
fig = corner.corner(fluxes[:,1:4],labels=labelz,quantiles=None,plot_contours=False)
#fig = corner.corner(fluxes[:,1:4],labels=labelz,quantiles=None,plot_contours=False)
fig.savefig("disc_triangle.png")
close(fig)





stop



#print('Make temporary cornerplot for the bar')
#index = [8,9,10,11,12,13]
##labelz = ["$mag$", "$Rout$", "$a$","$b$","q","$\theta$"]
##fig = corner.corner(fluxes[:,1:4],labels=labelz,quantiles=None,plot_contours=False)
#fig = corner.corner(fluxes[:,1:4],quantiles=None,plot_contours=False)
#fig.savefig("bar_triangle.png")





index = [(x>regions[0,0]) & (x<regions[0,1]) & (y>regions[0,2]) & (y<regions[0,3])]
print(index)
print(index.shape)
stop




names = ['sigma', 'dim_components', 'max_num_components', 'hyper_location', 'hyper_scale', 'num_components', 'x', 'y', 'mag', 'Re', 'n', 'q', 'theta', 'boxi', 'mag-bar', 'Rout', 'a', 'b', 'q-bar', 'theta-bar', 'box-bar']

new_index = [names[:] == 'mag' or names[:]
 == 'mag-bar']
print(new_index)

                             



#['sigma', 'dim_components', 'max_num_components', 'hyper_location', 'hyper_scale', 'num_components', 'x', 'y', 'mag', 'Re', 'n', 'q', 'theta', 'boxi', 'mag-bar', 'Rout', 'a', 'b', 'q-bar', 'theta-bar', 'box-bar']


shortnames = list_to_array_str(Z)
xindex = find_index_str(shortnames, 'x')
for i in range(0,len(xindex)):
     if xindex[i] == True:
          print(shortnames[i],posterior_sample[0,i])

x = posterior_sample[:,xindex==True]
yindex = find_index_str(shortnames, 'y')
y = posterior_sample[:,yindex==True]


#set_regionA = [9,12, 40, 60]  # [ xmin, xmax, ymin, ymax ]


for i in range(0,number_of_regions):                                # loop over all regions
     if i == 0:  
       index = [(x>set_regions[0,0]) & (x<regions[0,1]) & (y>regions[0,2]) & (y<regions[0,3])]
count = len(index)



print(len(index))
imshow(data, cmap='jet', zorder=1, interpolation='nearest')
plt.scatter(x[index], y[index],  c='white', s=8, zorder=2)
plt.show()


stop


#set_regionB = [40,55, 50, 55]  # [ xmin, xmax, ymin, ymax ]	
index = [(x>set_regions[1,0]) & (x<regions[1,1]) & (y>regions[1,2]) & (y<regions[1,3])]
#index = [(x>set_regionB[0]) & (x<set_regionB[1]) & (y>set_regionB[2]) & (y<set_regionB[3])]
count = count + len(index) 
     
#set_regionC = [55,65, 50, 60]  # [ xmin, xmax, ymin, ymax ]	
#index = [(x>set_regionC[0]) & (x<set_regionC[1]) & (y>set_regionC[2]) & (y<set_regionC[3])]
index = [(x>set_regions[2,0]) & (x<regions[2,1]) & (y>regions[2,2]) & (y<regions[2,3])]
count = count + len(index) 

#set_regionD = [35,45, 35, 50]  # [ xmin, xmax, ymin, ymax ]	
#index = [(x>set_regionD[0]) & (x<set_regionD[1]) & (y>set_regionD[2]) & (y<set_regionD[3])]
index = [(x>set_regions[3,0]) & (x<regions[3,1]) & (y>regions[3,2]) & (y<regions[3,3])]
count = count + len(index) 


imshow(data, cmap='jet', zorder=1, interpolation='nearest')
plt.scatter(x[index], y[index],  c='white', s=8, zorder=2)
plt.show()

print(count)

stop










stop 
index =  (x!=0) & (x >9)  &  (x < 15)
print(index)
print(x)
stop 

#index = np.where(shortnames == shortnames[7])
#print(index)
print(shortnames[int(index)])
stop 
xindex  = Z == all_names[6]

index = [Z == 'x']
print(index)
print(all_names[6])
print(I[Z[6] ])
print(Z[index])


#index = data[:, ] > 11 < and all_names == 'x'


# print all the names of the parameters
print(np.where(Z[0: int(start_index)] == 'x') ) 


stop 

bins=10
# get the sum of weights and sum of occurrences (their division gives the mean) 
H, xs, ys =np.histogram2d(xsubset, ysubset, bins=bins) 
count, _, _ =np.histogram2d(xsubset, ysubset, bins=bins) 
#sub.axes.set_xlabel('x')
#sub.axes.set_ylabel('y')





#from math import *
#import pylab as p
#import matplotlib.pyplot as plt
##port numpy as np


#fig, axs = plt.subplots(1,1,figsize=(15,10))
#axs.imshow(data ,cmap='gray', interpolation='none', alpha=0.3)
#fig.canvas.draw()


#import Image


#background = imshow(data, cmap='gray', interpolation='none',alpha=0.3)


#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.85)
#imshow(data, cmap='gray', interpolation='none',alpha=0.3)

#fig.canvas.draw()

#x=xsubset
#y=ysubset

#H, xedges, yedges = np.histogram2d(x, y, bins=(xsize,ysize))
#H.shape, xedges.shape, yedges.shape

##extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
##plt.imshow(H, extent=extent, interpolation='nearest')

#overlay = plt.imshow(H, interpolation='nearest')

#ax = plt.gca()
#ax.set_ylim(ax.get_ylim()[::-1])

#plt.colorbar()
#plt.xlabel("x")
#plt.ylabel("y")



#background = background.convert("RGBA")
#overlay = overlay.convert("RGBA")

#new_img = Image.blend(background, overlay, 0.5)
#new_img.save("new.png","PNG")


#stop


#print( " Create 2D histogram GRAY SCALE")
#nbins = 10
#size_of_font = 2
#size_of_label = 1.6

#H, xedges, yedges = np.histogram2d(xsubset,ysubset,bins=nbins)

#H = np.rot90(H)
#H = np.flipud(H)
# 
#	# Mask zeros
##Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zer


#fig2 = plt.figure()
##fig2 = plt.figure(figsize=(8,6), dpi=300)
#ax = fig2.add_subplot(111)
##    plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.11)
#plt.subplots_adjust(left=0.11, right=0.94, top=0.97, bottom=0.12)
##fig2.subplots_adjust(top=0.95)
##plt.pcolormesh(xedges,yedges,Hmasked,vmin=0.00000001, vmax=(1e6))
#plt.axis([0, 0, 100, 100])

#plt.xlabel(r'$x_1$',fontsize=size_of_font)
#plt.ylabel(r'$x_2$',fontsize=size_of_font)
#plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
#plt.tick_params(axis='both', which='minor', labelsize=size_of_font) 
#ax.tick_params(axis='x', labelsize=size_of_label)
#ax.tick_params(axis='y', labelsize=size_of_label)  
##cbar = plt.colorbar()
##cbar.ax.set_ylabel('Density')
##    plt.set_cmap('gray_r')
##pylab.show()
#filename = 'counts_plot.png'
#plt.savefig(filename)
##filename = 'counts_plot.eps'
##plt.savefig(filename)
#plt.close(fig2)
#stop 


for i in range(0,len(all_names)):
      print(" \n ")
      print(all_names[i],  " ", Z.index(all_names[i]))

      this_index = Z.index(all_names[i])

      # if it is one of the basic statistics 
      if i > 5:
          subset = posterior_sample[:,this_index:this_index+NOO]                   # shape (samplesize, max_num_components)
          index = [subset != 0]
      else:
          subset = posterior_sample[:,i]                                           # shape (samplesize, max_num_components)
          index = [subset != 0]
      print(subset.shape)               

      # in case of magnitude rescale to prior-distribution
      if ((Z[i] == 'mag') or (Z[i]== 'mag-bar')):
         print("Rescale Magnitude")
         for k in range(0, S):
            location_index = posterior_sample[k,3]              # 3 
            scale_index =  posterior_sample[k,4]                # 4 
            for j in range(0,int(NOO)):    
                  if subset[k,j] < location_index: 
                       subset[k,j] =  (location_index-subset[k,j])/ scale_index
                  else:
                       subset[k,j] =  (subset[k,j]- location_index)/ scale_index
                       


#                  if subset[k,j] < location_index: 
#                       subset[k,j] = subset[k,j]*scale_index + location_index
#                  else:
#                       subset[k,j] = -subset[k,j]*scale_index + location_index
#         print(location_index.shape, scale_index.shape, subset.shape)

#         new_subset = help_subset[subset != 0]           
#      else:
#         new_subset = subset[subset != 0]           

     # params[i] = 5*rt(2) + 25;   // shape=2, location=25, scale=5

      new_subset = subset[index]   
      print (len(new_subset), " " , new_subset)

      if ((Z[i] == 'mag') or (Z[i]== 'mag-bar')):     
        title = 'Histogram ' +all_names[i] 
        plt.title(title) 
        plt.xlabel(all_names[i])

        plt.hist(new_subset, alpha=0.5, bins=50)
        filename = 'test_prior_histogram_' +all_names[i] +'.png'
        plt.savefig(filename)           
 
      else:
      
        title = 'Histogram ' +all_names[i] 
        plt.title(title) 
        plt.xlabel(all_names[i])

        plt.hist(new_subset, alpha=0.5, bins=20)
        filename = 'test_prior_histogram_' +all_names[i] +'.png'
        plt.savefig(filename)      





## Plot 2D histogram using pcolor
#print( " Create 2D color histogram")
##	fig2 = plt.figure()
#fig2 = plt.figure(figsize=(8,6), dpi=300)
#plt.subplots_adjust(left=0.13, right=0.95, top=0.96, bottom=0.13)
#ax = fig2.add_subplot(111)
#plt.pcolormesh(xedges,yedges,Hmasked/(nwalkers*run_length),vmin=0.0, vmax=0.005)    
#plt.axis([-10, 10, -2, 8])
#plt.xlabel(r'$x_1$',fontsize=size_of_font)
#plt.ylabel(r'$x_2$',fontsize=size_of_font)
#cbar = plt.colorbar()
#plt.tick_params(axis='both', which='major', labelsize=size_of_font)  
#plt.tick_params(axis='both', which='minor', labelsize=size_of_font)
#ax.tick_params(axis='x', labelsize=size_of_label)
#ax.tick_params(axis='y', labelsize=size_of_label)
#cbar.ax.set_ylabel('Density')
##
##cbar.formatter.set_powerlimits((0, 0))
##cbar.ax.yaxis.set_offset_position('right')                         
##cbar.update_ticks()

#plt.set_cmap('jet')
## s = str(alpha) 
##    s = s.replace('.','')
#filename = 'Gaussian_color_counts_plot_N_'+ str(ndim) +'.png'
#filename = os.path.join(img_path, filename)         
#plt.savefig(filename)
#filename = 'Gaussian_color_counts_plot_N_'+ str(ndim) +'.eps'
#filename = os.path.join(img_path, filename)         
#plt.savefig(filename)
#plt.show()
#plt.close(fig2)


#img = posterior_sample[0, start_index : final_index ].reshape(xsize, ysize)
#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.85)
#imshow(img, cmap='jet', interpolation='nearest')
#filename = 'first_data.png'

##fig, ax = plt.subplots()
##im = ax.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
#         #      vmin=0, vmax=1)
##fig.colorbar(im)
#plt.show()




#all_index = Z.index(all_names[0])
#y_index = Z.index('y')

#mag_index = Z.index('mag')
#Re_index = Z.index('Re')

#n_index = Z.index('n')
#q_index = Z.index('q')

#theta_index = Z.index('theta')
#boxi_index = Z.index('boxi')

#mag_bar_index = Z.index('mag-bar')
#Rout_index = Z.index('Rout')

#print(a_index = Z.index('a'))
#print(b_index = Z.index('b'))

#print(q_bar_index = Z.index('q-bar'))
#print(theta_index = Z.index('theta-bar'))

#print(box_bar_index = Z.index('box-bar'))
#print(sigma_index = Z.index('sigma'))


#print(hyper_scale_index = Z.index('hyper-scale'))
#print(hpyer_location_index = Z.index('hpyer-location_'))


#print(x_index, y_index)



## posterior_sample[:, I['']])
#print("Element 1", Z[0]," ")  # sigma
#print("Element 2", Z[1]," ")  # dim comp
#print("Element 3", Z[2]," ")  # max num comp

#print("Element 4", Z[3]," ")  # hyper location
#print("Element 5", Z[4]," ")  # hyper scale
#print("Element 6", Z[5]," ")  # num comp

#print("Element 7", Z[6]," ")  # x
#print("Element 8", Z[7]," ")  # x
#print("Element 9", Z[8]," ")  # x


#print("Element 10", Z[9]," ")  # x
#print("Element 11", Z[10]," ")  # x
#print("Element 12", Z[11]," ")  # x

#print(" \n ")

#print("print keys", indices.keys())

#print("print keys", I.keys())
#print("print keys", type(Z))
#print("print keys", type(I))  #

##print("print values", indices.values() )
#print( indices.get("x", "none"))
#print( Z[5])
#print( I.get(Z[5], "none"))



#print(" \n ")
#print("Element 1", Z[0]," ")  # sigma
#print("All the sigmas",  posterior_sample[:,0 ])           # non-flexible
#print("All the sigmas",  posterior_sample[:,I==Z[0] ])     # flexible
#print("All the sigmas",  posterior_sample[:,I=='sigma'])   # even more flexible

#print("Element 2", Z[1]," ")  # sigma
#print("All the dim_components",  posterior_sample[:,1 ])           # non-flexible
#print("All the dim_components",  posterior_sample[:,I[Z[1]] ])     # flexible
#print("All the dim_components",  posterior_sample[:,I['dim_components']])   # even more flexible


#print("Element 6", Z[5]," ")  # sigma
#print("All the dim_components",  posterior_sample[:,5 ])           # non-flexible
#print("All the dim_components",  posterior_sample[:,I[Z[5]] ])     # flexible
#print("All the dim_components",  posterior_sample[:,I['num_components']])   # even more flexible


#print("Element 7-9", Z[6]," ")  # sigma
#print("All the dim_components",  posterior_sample[:,6:9 ])           # non-flexible
##print("All the dim_components",  posterior_sample[:,I[Z[6:9]] ])     # flexible
##print("All the dim_components",  posterior_sample[:,I['x']])   # even more flexible



##print("Element 6", Z[5]," ")  # sigma
##print("All the num comp",  posterior_sample[:,6 ])           # non-flexible
##print("All the num comp",  posterior_sample[:,I==Z[5] ])     # flexible
##print("All the num comp",  posterior_sample[:,I=='num_components'])   # even more flexible



##print("Element ", Z[6]," ")  # x-vales
##print("All the x",  posterior_sample[:, ])           # non-flexible
##print("All the x",  posterior_sample[:,I[Z[6]] ])     # flexible
##print("All the x",  posterior_sample[:,I=='x'])   # even more flexible


##index = np.where(Z=='sigma')
##print("index: ", index," ")
##print("index: ", index," ",Z[index]," ", I[index] )




##starter_point = 6
##subset = posterior_sample[:,I[Z[7]] ]
##print("Subsetshape", subset)
##print("Element 7", Z[6:6+3]," ")  # x

##print("X coordinates",posterior_sample[:,0:51] ,"\n")




##index = I['num_components']
##print( "Index: ", I['num_components'])
##print(posterior_sample[:,int(index)])



#index = I[Z[6] ]
#temp = posterior_sample[0:,int(index) ]
#NOOx = temp[0]
#print("X", NOOx)
#print("X", NOOx+NOO)
##print(posterior_sample[:,int(index)+1:int(index)+1+NOO])


###img = posterior_sample[9, starting:starting+xsize*ysize].reshape((xsize, ysize))            # Need to edit this
####img = posterior_sample[0, starting:starting+xsize*ysize].reshape((ysize, xsize))            # Need to edit this  
###print("Shape of image is", type(img))

###subplot(1, 3, 2)

###fig = plt.figure(figsize=(8,6))
###ax = fig.add_subplot(111)
###fig.subplots_adjust(top=0.85)
###imshow(img, cmap='copper', interpolation='nearest')
###filename = 'first_data.png'
###plt.savefig(filename)
###filename = 'first_data.pdf'
###plt.savefig(filename)
###filename = 'first_data.eps'
###plt.savefig(filename)


### Extract column by name
###print("Indices")
###print(indices)

###plt.show()
###print(indices)
###print(type(indices))

###print("Length of the dictionanry : ", len(indices))


##I = indices.get('indices') 

###print("Items of the dictionanry : ", indices.items() )


##Z = indices.get('colnames')
###print("Colnames: ", Z)
##print("Type of Colnames: ", type(Z))
##print("Type of items : ", type(I))
##print("Type of Posterior  ", type(posterior_sample)  )


###index = 'log_a[2]'
###print( "Index: ", I['log_a[2]'])
##index = I['num_components']
##print( "Index: ", I['num_components'])
##print(posterior_sample[:,int(index)])



###print("Print #num_components:",  posterior_sample[:, I['num_components']])
###print("Print #log_box-bar[2]:",  posterior_sample[:, I['log_box-bar[2]']])
###<type 'dict'>
###('Colnames: ', ['sigma', 'dim_components', 'max_num_components', 'hpyer_location_', 'hyper_scale', 'num_components', 'x', 'x', 'x', 'y', 'y', 'y', 'mag', 'mag', 'mag', 'Re', 'Re', 'Re', 'n', 'n', 'n', 'q', 'q', 'q', 'theta', 'theta', 'theta', 'boxi', 'boxi', 'boxi', 'mag-bar', 'mag-bar', 'mag-bar', 'Rout', 'Rout', 'Rout', 'a', 'a', 'a', 'b', 'b', 'b', 'q-bar', 'q-bar', 'q-bar', 'theta-bar', 'theta-bar', 'theta-bar', 'box-bar', 'box-bar', 'box-bar'

###print("Print #log_box-bar[2]:",  posterior_sample[:, I['']])



