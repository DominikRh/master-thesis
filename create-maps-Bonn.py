import numpy as np
from scipy.integrate import quad
import scipy.ndimage as sc
from bisect import bisect_left
from astropy.io import fits
import os
from os import path
from shutil import copy
import subprocess

#this serves as a blackbox script that can simply be run from console
#this script first creates a distribution of sources on a pixel map of a given size
#then, that source list is printed to a file that can be read by Lenstool
#Lenstool lenses the source list with a specified galaxy cluster in the foreground
#the resulting list of images is read into this Python script and again placed onto a pixel map
#lastly, noise and smoothing are added

#to be included: redshift slices

n_maps = 10

Smin = 0 #use a sensible value for the telescope
N_sources=1000000 #this just has to be a really large number; don't use this script for maps that are more than around 1 square degree!
Nmin = 1
Nmax = 5.78
map_area = 1 #sq degree; again, don't use this script for larger maps!
axis_ratio = 1 #factor how much larger y axis of the camera is
n_pixels_x = 1000
n_pixels_y = n_pixels_x * axis_ratio

map_size_y = np.sqrt(map_area/axis_ratio) #map extent in y direction in degree
map_size_x = map_area/map_size_y #this should be adapted into the functions so they do not need to calculate this anymore

#from CCAT-p technical data sheet for this channel:
del_T_CMB = 42 * 10**-6 
channel_freq = 350 * 10**9
res = 35 #resolution in arcseconds

#constants and functions
h = 6.626 * 10**-34
G = 6.67 * 10**-11
c = 3 * 10**8
kB = 1.38 * 10**-23
T_CMB = 2.73

#function for dN/dS from https://arxiv.org/pdf/2002.07199.pdf
#dN/dS = N0/S0 * ((S/S0)**a + (S/S0)**b)**-1
#best fit parameters for differential number counts:
#N0 = 4.4 +- 0.6 * 10**4 mJy**-1 deg**-2
#S0 = 0.1 +- 0.02 mJy
#a = 2.5 +0.2 -0.1
#b = 0.0 + 0.6 - 0.2
#--> dN/dS = 4.4/0.1 * ((S/0.1)**2.5 + (S/0.1)**0.0)**-1
#--> N = integral dN/dS dS = 4.4/0.1 * ((S/0.1)**3.5/3.5 + (S/0.1)**1/1)
def dNdS(S):
	diff = 4.4/0.1 * ((S/0.1)**2.5 + (S/0.1)**0.0)**-1
	return diff
		
#this function creates a list of sources		
def draw_source(Smin, Nmin, Nmax, N_sources, map_area, axis_ratio, n_pixels_x):	#potentially cut down on the number of parameters
	
	testlog = np.logspace(0,2,num=101)
	testlog = (testlog/(max(testlog)-min(testlog))-min(testlog)/(max(testlog)-min(testlog)))*2 #flux limits in mJy

	N_log = []

	for i in range(0,len(testlog)):
		Smax = testlog[i]
		N_log.append(quad(dNdS, Smin, Smax))
	
	N_log = np.asarray(N_log)
	N_log = N_log[:,0]


	N_draw = np.random.random(N_sources)


	N_draw = N_draw * (Nmax - Nmin) + Nmin

	S_draw = []
	   	   
	for i in range(0, len(N_draw)):
		pos = bisect_left(N_log, N_draw[i])
		m = (N_log[pos]-N_log[pos-1])/(testlog[pos]-testlog[pos-1])
		b = N_log[pos]-m*testlog[pos]
		x = (N_draw[i]-b)/m
		S_draw.append(x)
	
	map_size_y = np.sqrt(map_area/axis_ratio)
	map_size_x = map_area/map_size_y

	n_pixels_y = n_pixels_x * axis_ratio

	#map_pos_x = np.random.randint(n_pixels_x, size=N_sources) 
	#map_pos_y = np.random.randint(n_pixels_y, size=N_sources)
	
	map_pos_x = np.random.random(N_sources) * n_pixels_x
	map_pos_y = np.random.random(N_sources) * n_pixels_y

	S_draw = np.asarray(S_draw)
	S_draw = 3.34 * S_draw / 1000 #convert into Jy and from 1.2mm to 850 um

	flux_list_mean = np.mean(S_draw)

	mean_flux = 5.84 * map_area * 3.34 #5.84 from Gonzalez-Lopez paper
	u = mean_flux - flux_list_mean
	map_normalization = np.random.normal(loc=u, scale=0.12*3.34) 

	S_sum = 0
	i = 0
	while S_sum < map_normalization:
		S_sum += S_draw[i]
		i = i+1

	S_draw = S_draw[0:i]
	map_pos_x = map_pos_x[0:i]
	map_pos_y = map_pos_y[0:i]

	map_pos = np.stack((map_pos_x, map_pos_y), axis=-1)
	
	fluxes = S_draw
	
	source_list = np.hstack((map_pos, np.reshape(fluxes, (len(fluxes), 1))))
	
#	grid = np.zeros((n_pixels_x, n_pixels_y))

#	i=0
#adding data to grid
#	for pixel in map_pos:
#		grid[pixel[0], pixel[1]] = grid[pixel[0], pixel[1]] + S_draw[i]
#		i=i+1
		
#	return grid
	return source_list


def smoothing(res, n_pixels_x, map_size_x, grid):
	#smoothing, factor 2.3548 stems from converting FWHM into std
	sigma_pixel = res / 3600 / 2.3548 * n_pixels_x / map_size_x
	#filter_boundary = np.mean(grid)
	grid = sc.gaussian_filter(grid, sigma=sigma_pixel, mode='wrap')#cval=filter_boundary
	return grid

def noise(del_T_CMB, res, channel_freq, map_area, n_pixels_x, grid):
	#implementation of noise in pixels

	beam = np.pi*(res/3600/2)**2/np.log(2)*(np.pi/180)**2

	x = h * channel_freq / (kB * T_CMB)
	I_0 = 2 * (kB * T_CMB)**3 / (c * h)**2

	#conversion from del T_CMB (uK arcmin) into intensity
	del_I = I_0 * del_T_CMB / T_CMB * x**4 * np.exp(x) / (np.exp(x) - 1)**2
	noise_w = del_I * beam * 10**26 * map_area / n_pixels_x * 60 #10**29 conversion into mJy, area/size is pixel size in degree, *60 in arcmin -> dependence on pixel size!


	#adding noise
	for a in range(0, len(grid[0])):
		for b in range(0, len(grid[1])):
			grid[a][b] = grid[a][b] + np.random.normal(loc=0, scale=noise_w)
			
	return grid

#for writing a line in front of the array (for create_source_file)

def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


#creating the source file for Lenstool

def create_source_file(source_list, n, map_area, axis_ratio, n_pixels_x, n_pixels_y):

	map_size_y = np.sqrt(map_area/axis_ratio) #map extent in y direction in degree
	map_size_x = map_area/map_size_y #x direction

	sources = np.asarray([[0.0 for i in range(0,8)] for j in range(len(source_list))], dtype = object)
	sources[:,3] = 0.5 * map_size_y / n_pixels_y * 3600 #for these parameters, look at page 55 f. of the Lenstool manual https://projets.lam.fr/attachments/download/1348/lenstool.pdf
	sources[:,4] = 0.5 * map_size_y / n_pixels_y * 3600 #major and minor axis of the ellipse representing the source, in arcseconds
	#setting radius of the sources to half the pixel size
	sources[:,6] = 2.0 #redshift		
	sources[:,0] = np.arange(0, len(source_list), 1)
	sources[:,2] = source_list[:,1] * map_size_y / n_pixels_y * 3600#convert from "pixel coordinates" to arcseconds, y direction
	sources[:,1] = source_list[:,0] * map_size_x / n_pixels_x * 3600#x direction
	sources[:,7] = -2.5 * np.log10(source_list[:,2]) #convert from monochromatic flux in Jy at 850 um into magnitudes

	np.savetxt('source-list-{}.cat'.format(n), sources, fmt='%i %f %f %f %f %f %f %f')
	prepend_line('source-list-{}.cat'.format(n), '#REFERENCE 3 0.0 0.0') #there is probably a better way to do this
	
#---------------------------------------------------------------------------------------------------------------------

#polygon area

def polygon_area(coords):
	#this expects "coordinates" to be a list with a sublist for each polygon that contains the x,y coordinates as tuples
	area = []
	for i in range(len(coords)):
		area_dummy = 0
		for j in range(len(coords[i])):
			area_dummy += 0.5 * (coords[i][j-1][0] * coords[i][j][1] - coords[i][j][0] * coords[i][j-1][1])
		area.append(area_dummy)
	
	return area

#----------------------------------------------------------------	

#finding out area coverage of ellipse

def ellipse_placement(ellipse_x, ellipse_y, ellipse_a, ellipse_b, ellipse_theta, ellipse_flux, grid):

	x_k = np.arange(-int(ellipse_a)-2, int(ellipse_a)+3, 1)
	y_k = np.arange(-int(ellipse_a)-2, int(ellipse_a)+3, 1)

#intersection points of formerly vertical grid lines

	t_v = np.array([0.0,0.0])
	x_k_list = []

	coeff_v1 = np.sin(ellipse_theta)**2 / ellipse_a**2 + np.cos(ellipse_theta)**2 / ellipse_b**2

	for i in range(0, len(x_k)):
		coeff_v2 = 2 * x_k[i] * np.sin(ellipse_theta) * np.cos(ellipse_theta) * (1/ellipse_a**2 - 1/ellipse_b**2)
		coeff_v3 = x_k[i]**2 * (np.cos(ellipse_theta)**2 / ellipse_a**2 + np.sin(ellipse_theta)**2 / ellipse_b**2) - 1
		coeff_v = [coeff_v1, coeff_v2, coeff_v3]
		roots = np.roots(coeff_v)
		if np.imag(roots[0]) == 0 and np.imag(roots[1]) == 0:
			t_v = np.vstack((t_v, roots))
			x_k_list.append(x_k[i])
	
	t_v = np.delete(t_v, 0, 0)
	x_k_list = np.asarray(x_k_list)
	x_k_list = np.vstack((x_k_list, x_k_list))
	x_k_list = np.transpose(x_k_list)

	v_x = x_k_list * np.cos(ellipse_theta) / ellipse_a + t_v * np.sin(ellipse_theta) / ellipse_a
	v_y = - x_k_list * np.sin(ellipse_theta) / ellipse_b + t_v * np.cos(ellipse_theta) / ellipse_b

#intersection points of formerly horizontal grid lines

	t_h = np.array([0.0,0.0])
	y_k_list = []

	coeff_h1 = np.cos(ellipse_theta)**2 / ellipse_a**2 + np.sin(ellipse_theta)**2 / ellipse_b**2

	for i in range(0, len(y_k)):
		coeff_h2 = 2 * y_k[i] * np.sin(ellipse_theta) * np.cos(ellipse_theta) * (1/ellipse_a**2 - 1/ellipse_b**2)
		coeff_h3 = y_k[i]**2 * (np.sin(ellipse_theta)**2 / ellipse_a**2 + np.cos(ellipse_theta)**2 / ellipse_b**2) - 1
		coeff_h = [coeff_h1, coeff_h2, coeff_h3]
		roots = np.roots(coeff_h)
		if np.imag(roots[0]) == 0 and np.imag(roots[1]) == 0:
			t_h = np.vstack((t_h, roots))
			y_k_list.append(y_k[i])
		
	t_h = np.delete(t_h, 0, 0)
	y_k_list = np.asarray(y_k_list)
	y_k_list = np.vstack((y_k_list, y_k_list))
	y_k_list = np.transpose(y_k_list)

	h_x = t_h * np.cos(ellipse_theta) / ellipse_a + y_k_list * np.sin(ellipse_theta) / ellipse_a
	h_y = - t_h * np.sin(ellipse_theta) / ellipse_b + y_k_list * np.cos(ellipse_theta) / ellipse_b

#determine number of points
	num_h = h_x.shape[0] * h_x.shape[1]
	num_v = v_x.shape[0] * v_x.shape[1]
	h_array = np.array([1 for i in range(num_h)]) 
	v_array = np.array([0 for i in range(num_v)])

#compile all intersection points (of cells with the circle) into one array of these points

#order of elements: x location, y location, x_k or y_k of the gridline, t, type of gridline (1 indicates horizontal, 0 vertical)
	coordinates = np.hstack((np.reshape(h_x, (num_h,1)), np.reshape(h_y, (num_h,1)), np.reshape(y_k_list, (num_h,1)), np.reshape(t_h, (num_h,1)), np.reshape(h_array, (num_h,1))))

	coordinates = np.vstack((coordinates, np.hstack((np.reshape(v_x, (num_v,1)), np.reshape(v_y, (num_v,1)), np.reshape(x_k_list, (num_v,1)), np.reshape(t_v, (num_v,1)), np.reshape(v_array, (num_v,1))))))

	coordinates = np.unique(coordinates, axis=0)

#index = np.reshape(np.array([i for i in range(len(coordinates))]), (len(coordinates), 1))

#coordinates = np.hstack((coordinates, index))

#split up into positive y-axis and negative y-axis location, calculate angle

	coordinates_p = coordinates[np.where(coordinates[:,1]>=0.)[0]]
	coordinates_n = coordinates[np.where(coordinates[:,1]<0.)[0]]

	intersection_angles_n = np.arccos(coordinates_n[:,0])*180/np.pi+90

	coordinates_pp = coordinates_p[np.where(coordinates_p[:,0]>=0.)[0]]
	coordinates_pn = coordinates_p[np.where(coordinates_p[:,0]<0.)[0]]

	intersection_angles_pp = -np.arccos(coordinates_pp[:,0])*180/np.pi+90
	intersection_angles_pn = -np.arccos(coordinates_pn[:,0])*180/np.pi+450

	intersection_angles_n = np.reshape(intersection_angles_n, (len(intersection_angles_n), 1))
	intersection_angles_pp = np.reshape(intersection_angles_pp, (len(intersection_angles_pp), 1))
	intersection_angles_pn = np.reshape(intersection_angles_pn, (len(intersection_angles_pn), 1))

#stack angles into coordinate array, compile into one big array, sort according to angle

	coordinates_n = np.hstack((coordinates_n, intersection_angles_n))
	coordinates_pp = np.hstack((coordinates_pp, intersection_angles_pp))
	coordinates_pn = np.hstack((coordinates_pn, intersection_angles_pn))

	coordinates = np.vstack((coordinates_n, coordinates_pp, coordinates_pn))
	coordinates = coordinates[coordinates[:,-1].argsort()]
#elements are still x, y, x_k/y_k, t, type (1 horizontal, 0 vertical), angular location (starting from (0,1) in clockwise direction)

#intersection_angles = np.sort(np.concatenate((intersection_angles_pp, intersection_angles_n, intersection_angles_pn)))

	angles_differences = np.diff(np.append(coordinates[:,-1], 360)) #more up-to-date version of numpy allows np.diff(array, append=360)

	slither_area = 0.5 * (angles_differences*np.pi/180 - np.sin(angles_differences*np.pi/180))


#find all grid points within the circle

	grid_coordinates = np.array([0.0, 0.0, 0.0, 0.0])
	for i in range(0, len(x_k)):
		for j in range(0, len(y_k)):
			g_x = x_k[i] * np.cos(ellipse_theta) / ellipse_a + y_k[j] * np.sin(ellipse_theta) / ellipse_a
			g_y = - x_k[i] * np.sin(ellipse_theta) / ellipse_b + y_k[j] * np.cos(ellipse_theta) / ellipse_b
			grid_coordinates = np.vstack((grid_coordinates, np.array([g_x, g_y, x_k[i], y_k[j]])))

	grid_coordinates = np.delete(grid_coordinates, 0, 0)
#only leave entries within the circle
	grid_coordinates = grid_coordinates[grid_coordinates[:,0]**2 + grid_coordinates[:,1]**2 <= 1]

	#print coordinates and grid_coordinates for debugging
	#grid_coordinates
	#coordinates


#identify polygons of intersection points and grid points

	polygons = []
	for i in range(0, len(coordinates)):
		if coordinates[i][4] == coordinates[(i+1) % len(coordinates)][4]: #check if type of neighbouring points are the same, i.e. both are from either horizontal or vertical gridlines intersecting the circle
			if coordinates[i][3] > 0.0 and coordinates[(i+1) % len(coordinates)][3] > 0.0: #check if t is larger than 0, important for determining whether to round up or down to next integer
				t_dummy = np.floor(coordinates[i][3]) #determine parameter t, rounding down since t is positive
				if coordinates[i][4] == 1.0: #if type is horizontal, create coordinates tupel of 1. intersection point, 2. intersection point, grid point on same y_k as first intersection point and grid point on same y_k as second intersection point
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i+1) % len(coordinates)][0], coordinates[(i+1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[i][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[i][2])][0][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[(i+1) % len(coordinates)][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[(i+1) % len(coordinates)][2])][0][1]]]
					polygons.append(tupel_dummy)
				else: #otherwise type is vertical
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i+1) % len(coordinates)][0], coordinates[(i+1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[i][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[i][2])][0][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[(i+1) % len(coordinates)][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[(i+1) % len(coordinates)][2])][0][1]]]
					polygons.append(tupel_dummy)
			else: #t is smaller than 0
				t_dummy = np.ceil(coordinates[i][3]) #round t up
				if coordinates[i][4] == 1.0: #if type is horizontal, create coordinates tupel of 1. intersection point, 2. intersection point, grid point on same y_k as first intersection point and grid point on same y_k as second intersection point
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i+1) % len(coordinates)][0], coordinates[(i+1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[i][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[i][2])][0][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[(i+1) % len(coordinates)][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == t_dummy, grid_coordinates[:,3] == coordinates[(i+1) % len(coordinates)][2])][0][1]]]
					polygons.append(tupel_dummy)
				else: #otherwise type is vertical
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i+1) % len(coordinates)][0], coordinates[(i+1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[i][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[i][2])][0][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[(i+1) % len(coordinates)][2])][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == t_dummy, grid_coordinates[:,2] == coordinates[(i+1) % len(coordinates)][2])][0][1]]]
					polygons.append(tupel_dummy)
		else: #type of point is not the same; checking t for both at the same time is not useful since they might have the same sign or not
			if coordinates[i][3] > 0.0:
				t_dummy_0 = np.floor(coordinates[i][3])
			else:
				t_dummy_0 = np.ceil(coordinates[i][3])
			if coordinates[(i+1) % len(coordinates)][3] > 0.0:
				t_dummy_1 = np.floor(coordinates[(i+1) % len(coordinates)][3])
			else:
				t_dummy_1 = np.ceil(coordinates[(i+1) % len(coordinates)][3]) #now the two parameters t are determined
			if (t_dummy_0, coordinates[i][2]) == (coordinates[(i + 1) % len(coordinates)][2], t_dummy_1): #in this case, the polygon is a triangle
				if coordinates[i][4] == 1.0: #horizontal gridline, meaning y_k is given in array "coordinates" and by "t" we determine the first parameter
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i + 1) % len(coordinates)][0], coordinates[(i + 1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2], grid_coordinates[:,2] == t_dummy_0)][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2], grid_coordinates[:,2] == t_dummy_0)][0][1]]]
					polygons.append(tupel_dummy)
				else: #vertical gridline, so x_k is given and t_dummy determines the second parameter
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i + 1) % len(coordinates)][0], coordinates[(i + 1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2], grid_coordinates[:,3] == t_dummy_0)][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2], grid_coordinates[:,3] == t_dummy_0)][0][1]]]
					polygons.append(tupel_dummy)
			else: #five sided polygon
				if coordinates[i][4] == 1.0: #horizontal gridline, determine last polygon point accordingly
					if coordinates[i][2] > 0.0: #this means we need to decrease y_k by 1!
						last_point_x = grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2] - 1, grid_coordinates[:,2] == t_dummy_0)][0][0]
						last_point_y = grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2] - 1, grid_coordinates[:,2] == t_dummy_0)][0][1]
					else: #increase y_k! what about coordinates[i][2] = 0?
						last_point_x = grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2] + 1, grid_coordinates[:,2] == t_dummy_0)][0][0]
						last_point_y = grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2] + 1, grid_coordinates[:,2] == t_dummy_0)][0][1]
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i + 1) % len(coordinates)][0], coordinates[(i + 1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2], grid_coordinates[:,2] == t_dummy_0)][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[i][2], grid_coordinates[:,2] == t_dummy_0)][0][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[(i+1) % len(coordinates)][2], grid_coordinates[:,3] == t_dummy_1)][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[(i+1) % len(coordinates)][2], grid_coordinates[:,3] == t_dummy_1)][0][1]], [last_point_x, last_point_y]]
					polygons.append(tupel_dummy)
				else: #vertical gridline, determine last polygon point accordingly
					if coordinates[i][2] > 0.0: #this means we need to decrease x_k by 1!
						last_point_x = grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2] - 1, grid_coordinates[:,3] == t_dummy_0)][0][0]
						last_point_y = grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2] - 1, grid_coordinates[:,3] == t_dummy_0)][0][1]
					else: #increase x_k! what about coordinates[i][2] = 0?
						last_point_x = grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2] + 1, grid_coordinates[:,3] == t_dummy_0)][0][0]
						last_point_y = grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2] + 1, grid_coordinates[:,3] == t_dummy_0)][0][1]
					tupel_dummy = [[coordinates[i][0], coordinates[i][1]], [coordinates[(i + 1) % len(coordinates)][0], coordinates[(i + 1) % len(coordinates)][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2], grid_coordinates[:,3] == t_dummy_0)][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,2] == coordinates[i][2], grid_coordinates[:,3] == t_dummy_0)][0][1]], [grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[(i+1) % len(coordinates)][2], grid_coordinates[:,2] == t_dummy_1)][0][0], grid_coordinates[np.logical_and(grid_coordinates[:,3] == coordinates[(i+1) % len(coordinates)][2], grid_coordinates[:,2] == t_dummy_1)][0][1]], [last_point_x, last_point_y]]
					polygons.append(tupel_dummy)


#order polygon coordinates in (counter-)clockwise direction

	center_x = []
	center_y = []
	for i in range(len(polygons)):
		center_x.append(np.mean([k[0] for k in polygons[i]]))
		center_y.append(np.mean([k[1] for k in polygons[i]]))
	
	for i in range(len(polygons)):
		angles = []
		for j in range(len(polygons[i])):
			angles.append(np.arctan2((polygons[i][j][1] - center_y[i]), (polygons[i][j][0]) - center_x[i]))
		inds = np.asarray(angles).argsort()
		polygons[i] = np.asarray(polygons[i])[inds]

#works, kinda ugly, but eh

	polygon_areas = polygon_area(polygons)

#add polygon and slither areas, subtract from circle

	leftover_area = np.pi - np.sum(slither_area + polygon_areas)
	coordinates = np.hstack((coordinates, np.reshape((polygon_areas + slither_area)/np.pi, (len(polygon_areas), 1)))) #percentual area

#calculate area of one full pixel
	pixel_0 = grid_coordinates[0]
	pixel_1 = grid_coordinates[np.logical_and(grid_coordinates[:,2] == pixel_0[2], grid_coordinates[:,3] == (pixel_0[3]+1))]
	pixel_2 = grid_coordinates[np.logical_and(grid_coordinates[:,2] == (pixel_0[2]+1), grid_coordinates[:,3] == (pixel_0[3]))]
	pixel_3 = grid_coordinates[np.logical_and(grid_coordinates[:,2] == (pixel_0[2]+1), grid_coordinates[:,3] == (pixel_0[3]+1))]

	pixel_coords = [np.vstack((pixel_0, pixel_1, pixel_2, pixel_3))] #outer list necessary to make it work with polygon_area

#sort this

	pixel_x_c = np.mean(pixel_coords[0][:,0])
	pixel_y_c = np.mean(pixel_coords[0][:,1])

	angles = []
	for j in range(len(pixel_coords[0])):
		angles.append(np.arctan2((pixel_coords[0][j][1] - pixel_y_c), (pixel_coords[0][j][0]) - pixel_x_c))
	inds = np.asarray(angles).argsort()
	pixel_coords = pixel_coords[0][inds]

#determine number of pixels and covered percentage of area

	pixel_area = polygon_area([pixel_coords])[0]
	n_pixels = int(np.round(leftover_area / pixel_area))

#area percentage per pixel
	area_ppx = pixel_area / np.pi


#transform back into normal coordinate system

	x_coordinates = []
	y_coordinates = []
	for i in range(len(coordinates)):
		if coordinates[i][4] == 1: #horizontal
			x_k = np.floor(coordinates[i][3])
			y_k = coordinates[i][2]
		else:
			x_k = coordinates[i][2]
			y_k = np.floor(coordinates[i][3])
		x_coordinates.append(int(x_k + ellipse_x))
		y_coordinates.append(int(y_k + ellipse_y))


#add data onto map

	for i in range(len(x_coordinates)):
		grid[x_coordinates[i]][y_coordinates[i]] += ellipse_flux * coordinates[i][6]


#determine full pixels on grid
	
	#all grid points within an x and y interval
	full_pixel_grid = [] 

	for i in range(min(x_coordinates), max(x_coordinates)+1):
		for j in range(min(y_coordinates), max(y_coordinates)+1):
			full_pixel_grid.append([i, j])

	#remove all gridpoints that are part of an intersected cell
	delete_pixel_grid = []

	for i in range(len(x_coordinates)):
		delete_pixel_grid.append([x_coordinates[i], y_coordinates[i]])

	#how much percentage of the total flux of the source is part of the full pixels?
	total_pixel_flux_percentage = 1 - np.sum(coordinates[:,6])

	pixel_locations = [i for i in full_pixel_grid if i not in delete_pixel_grid]

	#edge_pixels = [[min(x_coordinates), min(y_coordinates)], [min(x_coordinates), max(y_coordinates)], [max(x_coordinates), min(y_coordinates)], [max(x_coordinates), max(y_coordinates)]]

	#all "outer pixels" of the selected intervals
	#this should work
	outer_pixels = []
	for i in range(min(x_coordinates), max(x_coordinates)+1):
		outer_pixels.append([i, min(y_coordinates)])
		outer_pixels.append([i, max(y_coordinates)])
		
	for i in range(min(y_coordinates), max(y_coordinates)+1):
		outer_pixels.append([min(x_coordinates), i])
		outer_pixels.append([max(x_coordinates), i])
		

	pixel_locations = [i for i in pixel_locations if i not in outer_pixels]

	if len(pixel_locations) != n_pixels:
		print('Warning: pixels may not be correct')

	flux_percentage_ppx = total_pixel_flux_percentage / (len(pixel_locations))

	for i in range(len(pixel_locations)):
		grid[pixel_locations[i][0]][pixel_locations[i][1]] += ellipse_flux * flux_percentage_ppx	
		
	return grid

#---------------------------------------------------------------------------------------------------------------------

#calling the functions

if path.exists('lenstool-maps') == False:
	os.mkdir('lenstool-maps')
for q in range(n_maps):
	grid = np.zeros((n_pixels_x, n_pixels_y))
	list_of_sources = draw_source(Smin, Nmin, Nmax, N_sources, map_area, axis_ratio, n_pixels_x)
	create_source_file(list_of_sources, q, map_area, axis_ratio, n_pixels_x, n_pixels_y) #pay attention that the .cat file is in linux format
	os.mkdir('lenstool-maps/map-{}'.format(q)) #make folder for each map
	#os.rename(os.getcwd() + 'source-list-{}.cat'.format(q), os.getcwd() + 'lenstool-maps/map-{}'.format(q) + 'source-list-{}.cat'.format(q)) #move source list
	os.rename(os.getcwd() + '/source-list-{}.cat'.format(q), os.getcwd() + '/lenstool-maps/map-{}/source-list-{}.cat'.format(q,q)) #move source list
	#copy(os.getcwd() + 'parameter.par', os.getcwd() + 'lenstool-maps/map-{}'.format(q) + 'parameter.par') #copy parameter file, has to be in cwd!
	#copy(os.getcwd() + 'parameter.par', os.getcwd() + 'lenstool-maps/map-{}/parameter.par'.format(q)) #copy parameter file, has to be in cwd!
	with open('parameter.par', 'r') as file: #copy parameter file into appropriate directory and insert proper source file name
		filedata = file.read()
	filedata = filedata.replace('test.cat', 'lenstool-maps/map-{}/source-list-{}.cat'.format(q,q))
	with open(os.getcwd() + '/lenstool-maps/map-{}/parameter.par'.format(q), 'w') as file:
		file.write(filedata)
	
	#tell lenstool what to do
	subprocess.call(['lenstool %s -n' %(os.getcwd() + '/lenstool-maps/map-{}/parameter.par'.format(q))], shell=True)

	subprocess.call(['mv *.dat %s' %(os.getcwd() + '/lenstool-maps/map-{}/'.format(q))], shell=True)
	subprocess.call(['mv image.all %s' %(os.getcwd() + '/lenstool-maps/map-{}/'.format(q))], shell=True)
	subprocess.call(['mv para.out %s' %(os.getcwd() + '/lenstool-maps/map-{}/'.format(q))], shell=True)
	
	image_list = np.genfromtxt(os.getcwd() + '/lenstool-maps/map-{}/image.all'.format(q))
	for t in range(len(image_list)):
		ellipse_x = image_list[:,1][t] / (map_size_x / n_pixels_x * 3600) #conversion back into "pixel coordinates"
		ellipse_y = image_list[:,2][t] / (map_size_y / n_pixels_y * 3600)
		ellipse_a = image_list[:,3][t]
		ellipse_b = image_list[:,4][t]
		ellipse_theta = image_list[:,5][t]
		#ellipse_z = image_list[:,6] 
		ellipse_flux = -1/2.5 * 10**(image_list[:,7][t]) #convert into flux units!
		grid = ellipse_placement(ellipse_x, ellipse_y, ellipse_a, ellipse_b, ellipse_theta, ellipse_flux, grid)
		#print(t)
	#potentially noise and smoothing
	hdu = fits.PrimaryHDU(grid)
	hdu.writeto('pixelmap-{}.fits'.format(q))
	#could also save as a normal image file; in that case, include proper plotting code
	

#hdu = fits.PrimaryHDU(grid)
#hdu.writeto('test-pixelmap.fits')

#properly recognize the angle of rotation! Lenstool documentation: from horizontal to major axis, counter clock wise - same as in implementation!

grid_smooth = smoothing(res, n_pixels_x, map_area, grid)

grid_noise = noise(del_T_CMB, res, channel_freq, map_area, n_pixels_x, grid_smooth)
