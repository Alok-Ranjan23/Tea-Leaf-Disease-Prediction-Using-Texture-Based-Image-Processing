import os
import mahotas as mt
import cv2 as cv
import glob
import numpy as np
import csv
import re
import count_train
import count_test
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import tkinter as tk
from tkinter import filedialog




def extract_feature(image):
	
	##Color Feature
	(mean,std) = cv.meanStdDev(image)
	
	#print(len(mean), type(mean))
	
	#print(len(std), type(std))
	
	color_feature = np.array(mean)
	
	color_feature = np.concatenate([color_feature,std]).flatten()
	
	#print(len(color_feature))
	
	##Texture Feature
	gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
	
	textures = mt.features.haralick(gray)
	
	ht_mean = textures.mean(axis = 0)
	
	#print(len(ht_mean), type(ht_mean))
	
	
	## Shape Features
	ret,thresh = cv.threshold(gray,127,255,0)
	
	x,contours, hierarchy =   cv.findContours(thresh.copy(),1,2)
	
	cnt = contours[0]
	
	area = cv.contourArea(cnt)
	#print(type(area))
	
	perimeter = cv.arcLength(cnt,True)
	#print(type(perimeter))
	
	shape = np.array([])
	shape = np.append(shape,area)
	shape = np.append(shape,perimeter)
	#print(len(shape))
	
	
	#print(len(ht_mean) + len(std) + len(mean) + len(shape))
	
	ht_mean = np.concatenate([ht_mean,color_feature]).flatten()
	
	ht_mean = np.concatenate([ht_mean,shape]).flatten()
	
	#print(len(ht_mean),ht_mean.shape)
	
	return(ht_mean)





def recommendation_by_prediction(pic, model_svc,model_dtree,model_random,model_ada,min_element,max_element):
    
    ### Image Processing
    print(pic)
    pic = cv.imread(pic)
    dim = (512,512)
    r_img = cv.resize(pic,dim)
    
    ### Extract image features
    feature_list = extract_feature(pic).tolist()
    siz = len(feature_list)
    
    l1 = list(min_element)
    l2 = list(max_element)
    
    
    
    l1.pop(5)
    l2.pop(6)
    
    l1.pop(5)
    l2.pop(6)
    
    # Scaling of relevent feature and removal of irrelevent features
    j=0
    for i in range(siz):
        if(i == 2 or i == 6 or i == 8 or i == 11 or i == 12 or i == 19 or i == 20):
            pass
        else:
            feature_list[i] = (feature_list[i] - l1[j]) /(l2[j] - l1[j])    
            j = j + 1
    
    feature_list.pop(2)
    feature_list.pop(5)
    feature_list.pop(6)
    feature_list.pop(8)
    feature_list.pop(8)
    feature_list.pop(14)
    feature_list.pop(14)
    
    feature_list = [feature_list]
        
    ### Individual prediction result from each model
    pred_1 =  model_svc.predict(feature_list)
    pred_2 =  model_dtree.predict(feature_list)
    pred_3 =  model_random.predict(feature_list)
    pred_4 =  model_ada.predict(feature_list)
    
    
    ### Final class of the leaf
    final = np.array([])
    final = np.append(final,stats.mode([pred_1,pred_2,pred_3,pred_4]))
    final = final.tolist()
    #print(len(final))
    disease_label = {0:'Healthy Leaf',1:'Algal Leaf Spot',2:'Blister Blight',3:'Grey Spot'}
    #final = final[0:len(prediction)-1:2]
    
    return(disease_label[int(final[0])])




f = open('list_val.txt','r')
x = f.readlines()
f.close()

min_string = x[0]
max_string = x[1]

min_list = min_string.split(' ')
max_list = max_string.split(' ')



min_list = list(map(float,min_list[:-1]))
max_list = list(map(float,max_list[:-1]))
print()
print(" Min_list: ", min_list)
print()
print(" Max_list: ", max_list)


 	

model_svc = pickle.load(open('final_svm.sav','rb'))
model_dtree = pickle.load(open('final_dtree.sav','rb'))
model_random = pickle.load(open('final_random.sav','rb'))
model_ada = pickle.load(open('final_ada.sav','rb'))


root = tk.Tk()
file_path = filedialog.askopenfilename()


typo = recommendation_by_prediction(file_path,model_svc,model_dtree,model_random,model_ada,min_list,max_list)

img = ImageTk.PhotoImage(Image.open(file_path))
w1 = tk.Label(root, justify=tk.CENTER, width = 400, height = 400, image = img).pack(side="top")
explanation = "Tea Leaf Class: " + typo
w2 = tk.Label(root, justify=tk.CENTER, padx=10,fg = "Red", bg = "light green", text=explanation, font = "Helvetica 16 bold italic").pack(side="bottom")
root.mainloop()

#print("Disease for that leaf: ",typo)





