# !/usr/bin/python3

### Description    : Generate consistent and accurate labels from mask data.
### Version        : Python = 3.5; Tensorflow = 1.14.0
### Author         : Karthik Ramachandran
### Created        : 2019/10/06
### Last updated   : 2019/10/06

#loaddata.py must be called first
import pickle
import numpy as np

masks = pickle.load(open('masks_data.p', 'rb'))

labels = []


#image[x][y][(r, g, b)]
#966x1296x1
i=0
for image in masks:
    print(i)
    label = np.empty((966, 1296, 1))
    for x in range(966):
        for y in range(1296):
            red = image[x][y][0]
            green = image[x][y][1]
            blue = image[x][y][2]
            
            #yellow = 255, 255, 0
            #mask images highlighted in yellow, so all highlights should be close to 255, 255, 0 with some error
            if np.sqrt((red-255)**2 + (green-255)**2 + (blue-255)**2) <= 100:
                label[x][y] = 1
            else:
                label[x][y] = 0
    labels.append(label)
    i+=1
#convert to numpy array and save as pickle
labels = np.array(labels)
pickle.dump(labels, open('labels_data.p', 'wb'))