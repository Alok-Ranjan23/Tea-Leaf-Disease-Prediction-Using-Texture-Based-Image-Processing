"""
Below segment of code helps in finding initial count of tea leaf images in a particular folder('Dataset') and helps in to keep track 
of augmented images.
"""
import os


# To count number of files in a folder
def images():
	for dirpath,dirname,file in os.walk('/home/ln-2/Desktop/Project/disease-final/train'):
		print(len(file))
		return(file)
	"""
	for file in glob.glob('*.jpg'):
		c += 1
		
	return(c)
	"""
	
	


