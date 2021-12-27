'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data
from palettable import cartocolors
from palettable.colorbrewer.qualitative import Dark2_7
from matplotlib.colors import ListedColormap
import random
from matplotlib.patches import Ellipse


class Transformation(analysis.Analysis):

	'''Transformation object constructor'''
	def __init__(self, orig_dataset, data=None):
		
		super().__init__(data)
		self.orig = orig_dataset

	'''projects the original dataset onto the data in headers'''
	def project(self, headers):

		dict = {}										# create a dictionary
		selected = self.orig.select_data(headers)		# select the data with the given headers
		
		for i in range(len(headers)):					# add the data to the dictionary
			dict[headers[i]] = i
			
		self.data = data.Data(data = selected, headers=headers, header2col=dict)	# update data
		
	'''returns a projected data array with homogeneous coordinate'''
	def get_data_homogeneous(self):
		
		ones = np.ones((self.data.get_num_samples(),1))	# create an nparray of ones
		h = np.hstack((self.data.data,ones))			# add a column of ones to the data
		return h

	'''returns a homogenous transformation matrix to be used for translation'''
	def translation_matrix(self, magnitudes):
	
		size = self.data.get_num_dims()
		identity = np.eye(size+1, size+1)				# set up the identity array
		
		for i in range(size):							# insert the magnitudes in the appropriate location for translating
			identity[i,size] = magnitudes[i]
			
		return identity

	'''returns a homogenous transformation matrix to be used for scaling'''
	def scale_matrix(self, magnitudes):
		
		size = self.data.get_num_dims()
		identity = np.eye(size+1, size+1)				# set up the identity array
			
		for i in range(size):							# insert the magnitudes in the appropriate location for scaling
			identity[i,i] = magnitudes[i]
			
		return identity

	'''translates the headers variables by the given magnitudes'''
	def translate(self, magnitudes):
		
		trans = self.translation_matrix(magnitudes)
		h = self.get_data_homogeneous()
		h = h.T											# transpose the homogenous data array
		
		dataHeaders = self.data.get_headers()
		
		x = trans@h										# translate
		
		new = x.T
		new = new[:, 0:-1]						
		self.data = data.Data(headers=dataHeaders, data=new, header2col=self.data.get_mappings())	# update data

		return new

	'''scales the headers variables by the given magnitudes'''
	def scale(self, magnitudes):

		h = self.get_data_homogeneous()
		h = h.T											# transpose the homogenous data array
		
		dataHeaders = self.data.get_headers()

		scale = self.scale_matrix(magnitudes)
		
		x = scale@h										# scale
		
		new = x.T
		new = new[:, 0:-1]
		self.data = data.Data(headers=dataHeaders, data=new, header2col=self.data.get_mappings())	# update data
				
		return new

	'''transforms the projected dataset by applying the homogeneous transformation matrix C'''
	def transform(self, C):
		
		dataHeaders = self.data.get_headers()
		
		d = self.get_data_homogeneous()
		new = (C @ d.T).T
		final = new[:, 0:-1]
		
		self.data = data.Data(headers=dataHeaders, data=final, header2col=self.data.get_mappings())	# update data
		
		return final

	'''normalizes all of the variables in the projected dataset'''
	def normalize_together(self):
		
		headers = self.data.get_headers()
		m1 = self.min(headers)
		mins = min(m1)
		m2 = self.max(headers)
		maxs = max(m2)
		
		tmags = []	# the translation magnitudes
		smags = []	# the scaling magnitudes
		
		for i in range( len(self.data.get_headers()) ):
			tmags.append(-mins)
			smags.append(1/(maxs-mins))
		
		trans = self.translation_matrix(tmags)
		scale = self.scale_matrix(smags)
		
		result = scale@trans
		x = self.transform(result)
		
		self.data = data.Data(headers=headers, data=x, header2col=self.data.get_mappings())	# update data
		
		return x

	'''normalizes each of the variables separately'''
	def normalize_separately(self):
		
		headers = self.data.get_headers()
		min = self.min(headers)
		max = self.max(headers)
		
		tmags = []	# the translation magnitudes
		smags = []	# the scaling magnitudes
		
		for i in range( len(headers)):
			tmags.append(-min[i])
			smags.append((1/(max[i]-min[i])))
		
		trans = self.translation_matrix(tmags)
		scale = self.scale_matrix(smags)
		
		result = scale@trans
		
		x = self.transform(result)
		
		self.data = data.Data(headers=headers, data=x, header2col=self.data.get_mappings())	# update data
		
		return x

	'''returns a 3D homogeneous rotation matrix for rotating the projected data about the given header'''
	def rotation_matrix_3d(self, header, degrees):
		
		h = self.get_data_homogeneous()
		identity = np.eye(h.shape[1], h.shape[1])	# create an identity array
		
		radians = degrees * (np.pi/180)				# translate given degrees into radians for rotation
		
		headers = self.data.get_headers()
		
		if headers.index(header) == 0:				# rotate around x
			identity[1,1] = np.cos(radians)
			identity[1,2] = -(np.sin(radians))
			identity[2,1] = np.sin(radians)
			identity[2,2] = np.cos(radians)

		elif headers.index(header) == 1:			# rotate around y
			identity[0,0] = np.cos(radians)
			identity[0,2] = np.sin(radians)
			identity[2,0] = -(np.sin(radians))
			identity[2,2] = np.cos(radians)
		
		elif headers.index(header) == 2:			# rotate around z
			identity[0,0] = np.cos(radians)
			identity[0,1] = -(np.sin(radians))
			identity[1,0] = np.sin(radians)
			identity[1,1] = np.cos(radians)
		
		return identity
		
	'''rotates the projected data about the given header by the given angle (in degrees)'''
	def rotate_3d(self, header, degrees):
		
		h = self.get_data_homogeneous()
		identity = self.rotation_matrix_3d(header, degrees)
		
		h = h.T
		new = (identity @h).T
		rot = new[:,0:-1]
		
		self.data = data.Data(headers=self.data.get_headers(), data=rot, header2col=self.data.get_mappings())	# update data
		
		return rot

	'''creates a 2D scatterplot with the third dimension represented by a color scale'''
	def scatter_color(self, ind_var, dep_var, c_var, title=None, cmap="gray_r"):

		headers = [ind_var, dep_var, c_var]
		rows = []
		
		x = []
		y = []
		z = []
		arr = []
		
		dat = self.data.select_data(headers, rows)
		
		for i in range(self.data.get_num_samples()): # repeat for each row
			x.append(dat[:,0])
			y.append(dat[:,1])
			z.append(dat[:,2])

		arr.append(x)
		arr.append(y)
		arr.append(z)
		
		fig, ax = plt.subplots(1,1)
		
		ax.set_title(title, fontsize=15)
		ax.set_xlabel(ind_var)
		ax.set_ylabel(dep_var)
		
		im = ax.scatter(x, y, c=z, s=70, cmap=cmap)	# set up the scatterplot as a colormap
		cbar = fig.colorbar(im, ax=ax)				# add a colorbar to the right side
		cbar.set_label(c_var)						# label the colorbar with the third variable
		
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.5, hspace=.5)
		plt.show()
	
	'''creates a 2D scatterplot with the third dimension represented by a color scale and the fourth dimension represented by marker size'''
	def scatter_color_size(self, ind_var, dep_var, c_var, s_var, title=None, cmap="gray"):

		headers = [ind_var, dep_var, c_var, s_var]
		rows = []
		
		x = []
		y = []
		z = []
		a = []
		arr = []
		
		dat = self.data.select_data(headers, rows)
		print(data)
		
		for i in range(self.data.get_num_samples()): # repeat for each row
			x.append(dat[:,0])
			y.append(dat[:,1])
			z.append(dat[:,2])
			a.append(dat[:,3])

		arr.append(x)
		arr.append(y)
		arr.append(z)
		arr.append(a)
		
		fig, ax = plt.subplots(1,1)
		
		ax.set_title(title, fontsize=15)
		ax.set_xlabel(ind_var)
		ax.set_ylabel(dep_var)
		
		#size = 70
		size = dat[:,3]/10	# change the size of the marker depending on the fourth variable
		
		# represent the 4th dimension with varying marker shapes instead?
		#markershape = dat[:,3].replace("four","square").replace("two","circle")
		
		im = ax.scatter(x, y, c=z, cmap=cmap, s=size, alpha=0.1)	# set up the scatterplot as a colormap
		cbar = fig.colorbar(im, ax=ax)								# add a colorbar to the right side
		cbar.set_label(c_var)										# label the colorbar with the third variable
		
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.5, hspace=.5)
		plt.show()

	'''creates a heatmap of the given variables'''
	def heatmap(self, headers=None, title=None, cmap="gray"):

		# Create a doppelganger of this Transformation object so that self.data
		# remains unmodified when heatmap is done
		data_clone = data.Data(headers=self.data.get_headers(),
							   data=self.data.get_all_data(),
							   header2col=self.data.get_mappings())
		dopp = Transformation(self.data, data_clone)
		dopp.normalize_separately()

		fig, ax = plt.subplots(figsize=(15,15))
		if title is not None:
			ax.set_title(title)
		ax.set(xlabel="Individuals")

		# Select features to plot
		if headers is None:
			headers = dopp.data.headers
		m = dopp.data.select_data(headers)

		# Generate heatmap
		hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

		# Label the features (rows) along the Y axis
		y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
		ax.set_yticks(y_lbl_coords, minor=True)
		y_lbls = [""] + headers
		ax.set_yticklabels(y_lbls)
		ax.grid(linestyle='none')

		# Create and label the colorbar
		cbar = fig.colorbar(hmap)
		cbar.ax.set_ylabel("Normalized Features")

		return fig, ax