import mahotas as mt
import cv2 as cv
import os
import glob
import numpy as np
import count
import csv
import re


def extract_feature(image):
	
	##Color Feature
	(mean,std) = cv.meanStdDev(image)
	
	print(len(mean), type(mean))
	
	print(len(std), type(std))
	
	color_feature = np.array(mean)
	
	color_feature = np.concatenate([color_feature,std]).flatten()
	
	print(len(color_feature))
	
	##Texture Feature
	gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
	
	textures = mt.features.haralick(gray)
	
	ht_mean = textures.mean(axis = 0)
	
	print(len(ht_mean), type(ht_mean))
	
	
	## Shape Features
	ret,thresh = cv.threshold(gray,127,255,0)
	
	x,contours, hierarchy =   cv.findContours(thresh.copy(),1,2)
	
	cnt = contours[0]
	
	area = cv.contourArea(cnt)
	print(type(area))
	
	perimeter = cv.arcLength(cnt,True)
	print(type(perimeter))
	
	shape = np.array([])
	shape = np.append(shape,area)
	shape = np.append(shape,perimeter)
	print(len(shape))
	
	
	print(len(ht_mean) + len(std) + len(mean) + len(shape))
	
	ht_mean = np.concatenate([ht_mean,color_feature]).flatten()
	
	ht_mean = np.concatenate([ht_mean,shape]).flatten()
	
	print(len(ht_mean),ht_mean.shape)
	
	return(ht_mean)
	

	
	
	
def create_csv():	
	
	files = count.images()
	mydata = [['energy','contrast','correlation','variance','inverse difference moment','sum average','sum variance','sum entropy','entropy','difference variance','difference entropy','info_corr',
			   'maximal_corr_coeff','mean_B','mean_G','mean_R','std_B','std_G','std_R','area','perimeter','label']]
	
	path = '/home/ln-2/Desktop/Alok/SVM/alok1/results'
	for file in files:	
		print(path+ file)
		image = cv.imread(path + '/' + file)
		print(file)
		#gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
		#means = cv.mean(image)
		#print(len(means))
		print(image.shape)
		dim = (512,512)
		r_img = cv.resize(image,dim)
		print(r_img.shape)
		feature = extract_feature(r_img)
		label = 0
		
		## Healthy leaf are labeled as 0.
		if(re.search('test[1-9]+',file)):
			label = 0
		else:
			## Algal leaf spot is labeled as 1.
			if(re.search('algal[1-9]+',file)):
				label = 1
			## Blister Blight is labeled as 2.	
			elif(re.search('blister[1-9]+',file)):
				label = 2
			## Grey Spot is labled as 3.	
			elif(re.search('grey[1-9]+',file)):
				label = 3
		
		feature = np.append(feature,label)
		print()
		print(len(feature))
		feature = feature.tolist()
		mydata.append(feature)
		
	myfile = open('mycsv.csv','w')
	with myfile:
		writer = csv.writer(myfile)
		writer.writerows(mydata)


create_csv()
