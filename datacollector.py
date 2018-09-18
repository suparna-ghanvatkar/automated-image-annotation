import requests
import random
import os
from os import listdir
from os.path import isfile, join


 """The collector file to get the dataset items randomly using reservoir sampling"""

def random_line():
    line_num = 0
    selected_line = ''
    with open('NUS-WIDE-urls.txt') as f:
        while 1:
            line = f.readline()
            if not line: break
            line_num += 1
            if random.uniform(0, line_num) < 1:
                selected_line = line
    return selected_line.strip()
    
#making the requests and getting the data and saving into the dataset folder
for i in range(0,1000):
	link = random_line()
	links = []
	links = link.split(' ')
	print links[3], links[5]
	try:
		r = requests.get(links[5])
		if r.status_code == 200:
			with open('dataset/'+ links[3]+'.jpeg', 'wb') as f:
				f.write(r.content)
	except:
		pass

#removing the corrupted images from the dataset folder
path = 'C:\\Users\\NISHANT\\Desktop\\Sem 2\\MP\\Project\\Code\\dataset\\'
Images = [ f for f in listdir(path) if isfile(join(path,f)) ] 
print Images
for image in Images:
	if os.path.getsize(join(path,image)) < 100 * 1024:
		os.remove(join(path,image))
	