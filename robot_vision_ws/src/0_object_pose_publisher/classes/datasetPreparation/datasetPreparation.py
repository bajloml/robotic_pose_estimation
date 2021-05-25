import sys
import os
import json
import cv2
import colorsys

import tensorflow as tf
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi





"""
    Class which prepares the dataset from the images and the json files generated by the NDDS

    this class inherits a keras.utils.Sequence class because it will be used to load the custom dataset to the network.
    this enables the model to be fed dynamically in batches
"""


"""
Some simple vector math functions to find the angle
between two points, used by affinity fields.
"""

def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner


class datasetPreparation(tf.keras.utils.Sequence):
    
    def __init__(self, root, batch_size, datasetName, nb_vertex=8, keep_orientation=True, normal=None, test=False, target_transform=None,
                 objectsofinterest="", img_size=400, noise=2, sigma=16, debugFolderPath="",
                 random_translation=(25.0, 25.0), random_rotation=15.0, transform=None, shuffle=True, saveAffAndBelImages=False):
        '''
        initialize the instance of the class (constructor)
        '''
        self.objectsofinterest = objectsofinterest
        self.img_size = img_size
        self.target_transform = target_transform
        self.root = root
        self.imgs = []
        self.test = test
        self.normal = normal
        self.keep_orientation = keep_orientation
        self.noise = noise
        self.sigma = sigma
        self.debugFolderPath = debugFolderPath
        self.random_translation = random_translation
        self.random_rotation = random_rotation
        self.transform = transform  # dictionary of transforms to apply to the image
        self.batch_size = batch_size
        self.datasetName = datasetName
        self.shuffle = shuffle
        self.saveAffAndBelImages = saveAffAndBelImages

        # read all image names in the the dataset folder (root path)
        imgAndJsonNames = []
        for imgName in os.listdir(self.root):
            # check if it .png image file (find only images and their names)
            if imgName.split('.')[1] == 'png':
                imgAndJsonNames.append(imgName.split('.')[0])

        # set will remove duplicate names and sort the list
        imgAndJsonNames = list(sorted(set(imgAndJsonNames)))

        imgAndJsonNamesComplete = []
        for i in imgAndJsonNames:
            imgAndJsonNamesComplete.append(['{}.{}'.format(i, 'png'),
                                            '{}{}.{}'.format(self.root, i, 'png'),
                                            '{}{}.{}'.format(self.root, i, 'json')])

        self.imgs = imgAndJsonNamesComplete

        self.on_epoch_end()

        # Shuffle the data.
        # np.random.shuffle(self.imgs)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        '''
        __getitem__ is called for every iteration of the dataset in the model.fit or in the model.fit_generator,
        it is called while the dataset loader iterating through the Dataset
        returns the whole batch of the size self.batchsize
        '''
        # debug
        # print('print from markoDataset.__getitem__ number{}'.format(index))

        # batch_indexes is an array of indexes from a batch, use slicing
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.imgs[k] for k in batch_indexes]

        #tf.print('dataset: {}, batch number: {}, data from: {}-{}'.format(self.datasetName, index, str(batch_indexes[0]), str(batch_indexes[-1])))

        return self.__get_data(batch_data=batch)
        
    def __get_data(self, batch_data):

        imgs_list = []
        beliefs_list = []
        affinities_list = []

        for index, data in enumerate(batch_data):
            # fileName = data[index][0]      # read file name from the self.imgs list (for example 00000.png)
            # pathImg = data[index][1]       # read the path to the image from the self.imgs list (for example C:\dataset\00000.png)
            # pathJson = data[index][2]      # read the path to the json from the self.imgs list (for example C:\dataset\00000.json)

            fileName = data[0]      # read file name from the self.imgs list (for example 00000.png)
            pathImg = data[1]       # read the path to the image from the self.imgs list (for example C:\dataset\00000.png)
            pathJson = data[2]      # read the path to the json from the self.imgs list (for example C:\dataset\00000.json)

            if os.path.isfile(pathImg) and os.path.isfile(pathJson):

                img_PIL = self.loadImage(pathImg)   # image loaded from the pathImg
                img_json = self.loadJSON(pathJson, self.objectsofinterest, img_PIL)     # json loaded from the pathJSON

                # img_size = img_PIL.size
                img_size = (400, 400)

                pointsBelief = img_json['pointsBelief']
                objects_centroid = img_json['centroids']
                points_all = img_json['points']
                points_keypoints = img_json['keypoints_2d']
                translations = tf.keras.backend.constant(np.array(img_json['translations']))
                rotations = tf.keras.backend.constant(np.array(img_json['rotations']))

                # Camera intrinsics
                path_cam = pathImg.replace(fileName, '_camera_settings.json')
                with open(path_cam) as data_file:
                    cameraSettings = json.load(data_file)
                # Assumes one camera
                cam = cameraSettings['camera_settings'][0]['intrinsic_settings']

                matrix_camera = np.zeros((3, 3))
                matrix_camera[0, 0] = cam['fx']
                matrix_camera[1, 1] = cam['fy']
                matrix_camera[0, 2] = cam['cx']
                matrix_camera[1, 2] = cam['cy']
                matrix_camera[2, 2] = 1

                # Load the cuboid sizes
                path_set = pathImg.replace(fileName, '_object_settings.json')
                with open(path_set) as data_file:
                    objectSettings = json.load(data_file)

                if self.objectsofinterest is None:
                    cuboid = np.array(objectSettings['exported_objects'][0]['cuboid_dimensions'])
                else:
                    for info in objectSettings["exported_objects"]:
                        if self.objectsofinterest in info['class']:
                            cuboid = np.array(info['cuboid_dimensions'])

                # Debug show the image
                # cv_img_original = np.array(img_PIL)
                # cv_img_original = cv2.resize(cv_img_original, dsize=(800, 800))
                # cv2.imshow('cv_img_copy_{}_original'.format(fileName), cv_img_original)
                # cv2.waitKey(1)

                def Reproject(points, tm, rm):
                    """
                    Reprojection of points when rotating the image
                    """
                    proj_cuboid = np.array(points)

                    rmat = np.identity(3)
                    rmat[0:2] = rm
                    tmat = np.identity(3)
                    tmat[0:2] = tm

                    new_cuboid = np.matmul(
                        rmat, np.vstack((proj_cuboid.T, np.ones(len(points)))))
                    new_cuboid = np.matmul(tmat, new_cuboid)
                    new_cuboid = new_cuboid[0:2].T

                    return new_cuboid

                # Random image manipulation/augmentation, rotation and translation with zero padding
                ''' 
                dx = round(np.random.normal(0, 2) * float(self.random_translation[0]))
                dy = round(np.random.normal(0, 2) * float(self.random_translation[1]))
                angle = round(np.random.normal(0, 1) * float(self.random_rotation))

                tm = np.float32([[1, 0, dx], [0, 1, dy]])
                rm = cv2.getRotationMatrix2D((img_PIL.size[0]/2, img_PIL.size[1]/2), angle, 1)

                for i_objects in range(len(pointsBelief)):
                    points = pointsBelief[i_objects]
                    new_cuboid = Reproject(points, tm, rm)
                    pointsBelief[i_objects] = new_cuboid.tolist()
                    objects_centroid[i_objects] = tuple(new_cuboid.tolist()[-1])
                    pointsBelief[i_objects] = list(map(tuple, pointsBelief[i_objects]))

                for i_objects in range(len(points_keypoints)):
                    points = points_keypoints[i_objects]
                    new_cuboid = Reproject(points, tm, rm)
                    points_keypoints[i_objects] = new_cuboid.tolist()
                    points_keypoints[i_objects] = list(map(tuple, points_keypoints[i_objects]))

                image_r = cv2.warpAffine(np.array(img_PIL), rm, img_PIL.size)
                result = cv2.warpAffine(image_r, tm, img_PIL.size)
                img_PIL = Image.fromarray(result)                   '''

                # Resize the image or the tensors to be (9, 50, 50)
                # assumes image width and height are equals values and creates resizeTo tuple(50, 50)
                scaledSize = (min(img_size))/8.0  # 50
                if min(img_PIL.size) != scaledSize:
                    scaleFactor = int(min(img_PIL.size)/scaledSize)
                else:
                    scaleFactor = 8
                resizeTo = (int(img_PIL.size[0]/scaleFactor), int(img_PIL.size[1]/scaleFactor))

                # Create the belief map,
                # sigma is the size of the point shown on the belief maps,
                # pointBelief are the points from the 00000.json file projected cuboid and the projected cuboid centeroid, (projected =>pixels)
                # nbpoinst is 9 because that is cuboid projected points + center of the cuboid projected. this is comming from the .json file
                # finaly get the beliefs tensor of shape (9, 50, 50) for each image json pair.
                # create a tensor from the numpy array of PIL images
                nbpoints = 9
                tensorBeliefs = self.CreateBeliefMap(img=img_PIL,
                                                      pointsBelief=pointsBelief,
                                                      nbpoints=nbpoints,
                                                      sizeFactor=resizeTo,
                                                      sigma=self.sigma)

                # Create affinity maps tensor and the PIL image of the vector
                tensorAffinities, img_PIL_vector = self.CreateAffinityMap(img=img_PIL,
                                                                          nb_vertex=(nbpoints-1),
                                                                          pointsInterest=pointsBelief,
                                                                          objects_centroid=objects_centroid,
                                                                          scale=scaleFactor)

                if self.saveAffAndBelImages:
                    # save the result image(image on which the random transform has been applied) with the corresponding resized belief images
                    self.saveBeliefImageGrid(beliefsImgList=tensorBeliefs.numpy().tolist(), img=img_PIL,
                                             pathToSave='{}{}{}'.format(self.debugFolderPath, '\\beliefGridResized_', fileName))

                    # debug: save the result image(image on which the random transform has been applied) with the corresponding resized affnity images
                    self.saveAffinityImageGrid(affinitiesTensor=tensorAffinities, img=img_PIL, img_vector=img_PIL_vector,
                                               pathToSave='{}{}{}'.format(self.debugFolderPath, '\\affinityGridResized_', fileName))

                # image augmentation
                # apply random contrast, random brightness and resize
                # apply the normal probability density function, around the loc=1 for a value scale(gauss around loc for a scale limit)
                # enhancer = ImageEnhance.Contrast(img_PIL)
                # img_PIL = enhancer.enhance(np.random.normal(loc=1, scale=self.transform["contrast"]))
                # enhancer = ImageEnhance.Brightness(img_PIL)
                # img_PIL = enhancer.enhance(np.random.normal(loc=1, scale=self.transform["brightness"]))
                # img_PIL = img_PIL.resize((self.transform["imgSize"], self.transform["imgSize"]))

                # transform PIL image to tensor over the numpy array
                # to numpy array
                img_np = tf.keras.preprocessing.image.img_to_array(img_PIL)

                # apply the normalization on the image array
                # img_np *= (1.0/img_np.max())

                # to tensorflow tensor
                tensorImg = tf.keras.backend.constant(img_np)

                # switch axes from [9, 50, 50] and [16, 50, 50] to be of shape [50, 50, 9] and [50, 50, 16] (filters last)
                tensorBeliefs = tf.transpose(tensorBeliefs, [1, 2, 0])
                tensorAffinities = tf.transpose(tensorAffinities, [1, 2, 0])

                tensorAffinities = tf.abs(tensorAffinities) 
                tensorAffinities = tf.math.divide(tf.subtract(tensorAffinities, tf.reduce_min(tensorAffinities)),
                                                    tf.subtract(tf.reduce_max(tensorAffinities), tf.reduce_min(tensorAffinities))) *255

                # append to the list of tensors
                imgs_list.append(tensorImg)
                beliefs_list.append(tensorBeliefs)
                affinities_list.append(tensorAffinities)

            else:
              tf.print('{} is missing json or png'.format(fileName.split('.')[0]))

        batch_imgs = tf.keras.backend.stack(imgs_list)
        batch_beliefs = tf.keras.backend.stack(beliefs_list)
        batch_affinities = tf.keras.backend.stack(affinities_list)

        # apply the normalization of the tensors        (value − min_value) / (max_value − min_value)
        # batch_imgs = tf.math.divide(tf.subtract(batch_imgs, tf.reduce_min(batch_imgs)),
        #                              tf.subtract(tf.reduce_max(batch_imgs), tf.reduce_min(batch_imgs)))
        
        # batch_imgs = batch_imgs/255.0
                                     
        # batch_beliefs = tf.math.divide(tf.subtract(batch_beliefs, tf.reduce_min(batch_beliefs)),
        #                                 tf.subtract(tf.reduce_max(batch_beliefs), tf.reduce_min(batch_beliefs)))
        
        # batch affinities has negative values and that is why tf.abs is used
        # batch_affinities = tf.math.divide(tf.subtract(tf.abs(batch_affinities), tf.reduce_min(tf.abs(batch_affinities))),
        #                                    tf.subtract(tf.reduce_max(tf.abs(batch_affinities)), tf.reduce_min(tf.abs(batch_affinities))))


        # sequence must return a tuple, because later the fit_generator will expect a unambiguous type
        # of 2 or 3 members (x, y, (optional) sample weights)
        return({'images': batch_imgs},               # size = (batch_size, 400, 400, 3)
               {'beliefs': batch_beliefs,           # size = (batch_size, 50, 50, 16)
                'affinities': batch_affinities})    # size = (batch_size, 50, 50, 9)
  
    def __len__(self):
        '''
        overload of the __len__ method, returns the number of batches per epoch(in dataset)
        '''
        # length = int(len(self.imgs)//self.batch_size)
        if len(self.imgs) >= self.batch_size:
            length = int(np.floor(len(self.imgs)/self.batch_size))
        else:
            if (len(self.imgs) > 0) and (len(self.imgs) < self.batch_size):
                length = 1

        return length

    def getGenerator(self):
        """
        get the batch generator, tried to make it work with the fit_generator 
        """
        for batch_index, batch_tuple in enumerate(self):
            tf.print('dataset {}, batch_index: {}'.format(str(self.datasetName), str(batch_index)))
            yield batch_tuple

    def loadImage(self, path):
        img = Image.open(path).convert('RGB')
        return img.resize((self.img_size, self.img_size))

    def loadJSON(self, path, objectsofinterest, img):
        """
        Loads the data from a json file.
        If there are no objects of interest, then load all the objects.
        """
        with open(path) as data_file:
            data = json.load(data_file)
        # print (path)
        pointsBelief = []
        boxes = []
        points_keypoints_3d = []
        points_keypoints_2d = []
        pointsBoxes = []
        poses = []
        centroids = []

        translations = []
        rotations = []
        points = []

        for i_line in range(len(data['objects'])):
            info = data['objects'][i_line]
            if not objectsofinterest is None and not objectsofinterest.lower() in info['class'].lower():
                continue

            box = info['bounding_box']
            boxToAdd = []

            boxToAdd.append(float(box['top_left'][0]))
            boxToAdd.append(float(box['top_left'][1]))
            boxToAdd.append(float(box["bottom_right"][0]))
            boxToAdd.append(float(box['bottom_right'][1]))
            boxes.append(boxToAdd)

            boxpoint = [(boxToAdd[0], boxToAdd[1]), (boxToAdd[0], boxToAdd[3]),
                        (boxToAdd[2], boxToAdd[1]), (boxToAdd[2], boxToAdd[3])]

            pointsBoxes.append(boxpoint)

            # 3dbbox with belief maps
            points3d = []

            pointdata = info['projected_cuboid']
            for p in pointdata:
                points3d.append((p[0], p[1]))

            # Get the centroids
            pcenter = info['projected_cuboid_centroid']

            points3d.append((pcenter[0], pcenter[1]))
            pointsBelief.append(points3d)
            points.append(points3d + [(pcenter[0], pcenter[1])])
            centroids.append((pcenter[0], pcenter[1]))

            # load translations
            location = info['location']
            translations.append([location[0], location[1], location[2]])

            # quaternion
            rot = info["quaternion_xyzw"]
            rotations.append(rot)

        return {
            "pointsBelief": pointsBelief,
            "rotations": rotations,
            "translations": translations,
            "centroids": centroids,
            "points": points,
            "keypoints_2d": points_keypoints_2d,
            "keypoints_3d": points_keypoints_3d,
        }

    def saveBeliefImageGrid(self, beliefsImgList, img, pathToSave):
        '''
        beliefImg   --> list of PIL images
        img         --> image to append on the right side
        pathToSave  --> path on which to save the constructed image

        Constructs a grid of images from  the list beliefImg concats the img on the right side and save the whole image on the pathToSave
        '''
        # create a numpy array of PIL images from the list
        for i in range(len(beliefsImgList)):
            if i == 0:
                beliefsImg_np = np.array([np.array(beliefsImgList[i])])
            else:
                beliefsImg_np = np.append(beliefsImg_np, [np.array(beliefsImgList[i])], axis=0)

        numberOfColl = int(sqrt(len(beliefsImg_np)))
        numberOfRows = int(sqrt(len(beliefsImg_np)))
        # border color (white)
        borderColor = [255, 255, 255]

        for rowCounter in range(numberOfRows):
            rowImage_sliced = beliefsImg_np[rowCounter:(rowCounter + numberOfColl)]  # step for slicing is numberOfColl
            for image in range(len(rowImage_sliced)):
                # make a border on the edge of the image
                borderedImage = cv2.copyMakeBorder(rowImage_sliced[image], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=borderColor)
                # add these images to the numpy array
                if image == 0:
                    rowImageBordered_np = np.array([np.array(borderedImage)])
                else:
                    rowImageBordered_np = np.append(rowImageBordered_np, [np.array(borderedImage)], axis=0)
            # horizontal concationation of the whole row
            rowImage_cv = cv2.hconcat(rowImageBordered_np)
            # add it to the row numpy
            if rowCounter == 0:
                rows_cv = np.array([np.array(rowImage_cv)])
            else:
                rows_cv = np.append(rows_cv, [np.array(rowImage_cv)], axis=0)

        # make a grid of images(rows) vertical concat of the numpy array
        # change the type of the rows_cv and add a color to match the type and the shape of the 'img' to be able to concatonate
        gridBeliefImage_cv = cv2.cvtColor(cv2.vconcat(rows_cv.astype(np.uint8)), cv2.COLOR_BGR2RGB)
        # attach result image on the grid horizontaly
        gridBelief_cv = cv2.hconcat([gridBeliefImage_cv, cv2.resize(np.array(img), dsize=(gridBeliefImage_cv.shape[0], gridBeliefImage_cv.shape[1]))])
        # cv2.imshow('cv_grid_belief2', gridBelief_cv)
        # cv2.waitKey(0)
        cv2.imwrite('{}'.format(pathToSave), gridBelief_cv)

    def saveAffinityImageGrid(self, affinitiesTensor, img, img_vector, pathToSave):
        '''
        beliefImg   --> tf tensor of the vector
        img         --> image to append on the right side
        pathToSave  --> path on which to save the constructed image

        Constructs a grid of images from the tf tensor concats the img on the right side and save the whole image on the pathToSave
        '''
        # create a numpy array of tf tensor
        for i in range(affinitiesTensor.shape.dims[0]):
            if i == 0:
                # add the tensor[0] at the new axis 0
                affinityImg_np = np.expand_dims(tf.keras.backend.eval(affinitiesTensor[i]), axis=0)
            else:
                affinityImg_np = np.append(affinityImg_np, [tf.keras.backend.eval(affinitiesTensor[i])], axis=0)

        # change the type of the np array created from the tf tensor to match
        # the image uint8 to concatonate it later(hconcat and vconcat)
        affinityImg_np = (affinityImg_np*255).round().astype(np.uint8)

        numberOfColl = int(sqrt(len(affinityImg_np)))
        numberOfRows = int(sqrt(len(affinityImg_np)))
        # border color (white)
        borderColor = [255, 255, 255]

        for rowCounter in range(numberOfRows):
            rowImage_sliced = (affinityImg_np[rowCounter:(rowCounter + numberOfColl)])  # step for slicing is numberOfColl
            for image in range(len(rowImage_sliced)):
                # make a border on the edge of the image
                borderedImage = cv2.copyMakeBorder(rowImage_sliced[image], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=borderColor)
                # add these images to the numpy array
                if image == 0:
                    rowImageBordered_np = np.array([np.array(borderedImage)])
                else:
                    rowImageBordered_np = np.append(rowImageBordered_np, [np.array(borderedImage)], axis=0)
            # horizontal concationation of the whole row
            rowImage_cv = cv2.hconcat(rowImageBordered_np)
            # add it to the row numpy
            if rowCounter == 0:
                rows_cv = np.array([np.array(rowImage_cv)])
            else:
                rows_cv = np.append(rows_cv, [np.array(rowImage_cv)], axis=0)

        # make a grid of images(rows) vertical concat of the numpy array
        gridImage_cv = cv2.vconcat((rows_cv))
        gridImage_cv = cv2.cvtColor(gridImage_cv, cv2.COLOR_BGR2RGB)
        # cv2.imshow('gridImage_cv', gridImage_cv)
        # cv2.waitKey(0)

        # attach result image on the grid horizontaly
        grid_cv = cv2.hconcat([gridImage_cv,
                               cv2.resize(np.array(img_vector), dsize=(gridImage_cv.shape[0], gridImage_cv.shape[1])),
                               cv2.resize(np.array(img), dsize=(gridImage_cv.shape[0], gridImage_cv.shape[1]))])
        # cv2.imshow('cv_grid_affinity', grid_cv)
        # cv2.waitKey(0)
        cv2.imwrite('{}'.format(pathToSave), grid_cv)

    def CreateBeliefMap(self, img, pointsBelief, nbpoints, sizeFactor, sigma=16):
        """
        Args:
            img: image
            pointsBelief: list of points in the form of
                        [nb object, nb points, 2 (x,y)]
            nbpoints: (int) number of points, DOPE uses 8 points here
            sigma: (int) size of the belief map point
        return:
            return an tensor of array of PIL black and white images representing the
            belief maps
        """
        beliefsImg = []
        sigma = int(sigma)
        for numb_point in range(nbpoints):
            array = np.zeros(img.size)
            out = np.zeros(img.size)

            for point in pointsBelief:
                p = point[numb_point]
                w = int(sigma*2)
                if p[0]-w>=0 and p[0]+w<img.size[0] and p[1]-w>=0 and p[1]+w<img.size[1]:
                    for i in range(int(p[0])-w, int(p[0])+w):
                        for j in range(int(p[1])-w, int(p[1])+w):
                            array[i, j] = np.exp(-(((i - p[0])**2 + (j - p[1])**2)/(2*(sigma**2))))

            stack = np.stack([array, array, array],axis=0).transpose(2,1,0)
            imgBelief = Image.new(img.mode, img.size, "black")
            beliefsImg.append(Image.fromarray((stack*255).astype('uint8')))

        # create a numpy array of resized PIL images from the list to stack them into a tensor
        for i in range(len(beliefsImg)):
            if i == 0:
                beliefsImg_np = (np.array([np.array(beliefsImg[i].resize(sizeFactor))]))[:, :, :, 0]
            else:
                beliefsImg_np = np.append(beliefsImg_np, [(np.array(beliefsImg[i].resize(sizeFactor))[:, :, 0])], axis=0)

        return tf.keras.backend.constant(beliefsImg_np)

    def CreateAffinityMap(self, img, nb_vertex, pointsInterest, objects_centroid, scale):
        """
        Function to create the affinity maps,
        e.g., vector maps pointing toward the object center.

        Args:
            img: PIL image
            nb_vertex: (int) number of points. cuboid points?? value = 8
            pointsInterest: list of object points    points of the object pixels coordinate system given in the json file. value = list[number of objects classes]
                            there are 9 points per object (8 vertices points + 1 center of the object point)
            objects_centroid: (x,y) centroids for the objects value = list[number of objects classes]
            scale: (float) by how much you need to scale down the image.  number to scale the image to be 50x50 ??
        return:
            return a list of tensors for each point except centroid point
        """
        # Apply the downscale right now, so the vectors are correct.
        img_affinity = Image.new(img.mode, (int(img.size[0]/scale), int(img.size[1]/scale)), "black")
        # debug
        # should be completely black image
        # img_affinity.show("afterDownscale")

        affinities = []
        for i_points in range(nb_vertex):
            # affinities.append(torch.zeros(2, int(img.size[1]/scale), int(img.size[0]/scale)))
            affinities.append(tf.keras.backend.zeros((2, int(img.size[1]/scale), int(img.size[0]/scale))))

        # pointsInterest is a list of points for each object, points are also lists
        for i_pointsImage in range(len(pointsInterest)):
            pointsImage = pointsInterest[i_pointsImage]
            center = objects_centroid[i_pointsImage]
            # for each point of the cuboid get the vector to the center of the object
            for i_points in range(nb_vertex):
                affinity_pair, img_affinity = self.getAfinityCenter(width=int(img.size[0]/scale),
                                                                    height=int(img.size[1]/scale),
                                                                    point=tuple((np.array(pointsImage[i_points])/scale).tolist()),
                                                                    center=tuple((np.array(center)/scale).tolist()),
                                                                    radius=1,
                                                                    img_affinity=img_affinity)

                # debug
                # print('affinity: {}'.format(i_points))
                # img_affinity.show('after getAfinityCenter')

                affinities[i_points] = (affinities[i_points] + affinity_pair)/2

                # Normalizing, keras tensor to numpy array
                v = tf.keras.backend.eval(affinities[i_points])

                # debug
                # ImgAffinities0 = Image.fromarray((keras.backend.eval(affinities[i_points]))[0]*255)
                # ImgAffinities0.show('ImgAffinities0')
                # ImgAffinities1 = Image.fromarray((keras.backend.eval(affinities[i_points]))[1]*255)
                # ImgAffinities1.show('ImgAffinities1')

                xvec = v[0]
                yvec = v[1]

                norms = np.sqrt(xvec * xvec + yvec * yvec)
                nonzero = norms > 0

                xvec[nonzero] /= norms[nonzero]
                yvec[nonzero] /= norms[nonzero]

                # affinities[i_points] = torch.from_numpy(np.concatenate([[xvec],[yvec]]))
                # keras.backend.constant(beliefsImg_np[j])
                affinities[i_points] = tf.keras.backend.constant(np.concatenate([[xvec], [yvec]]))

                # debug
                # ImgAffinities0 = Image.fromarray((keras.backend.eval(affinities[i_points]))[0]*255)
                # ImgAffinities0.show('ImgAffinities0')
                # ImgAffinities1 = Image.fromarray((keras.backend.eval(affinities[i_points]))[1]*255)
                # ImgAffinities1.show('ImgAffinities1')

        # concatonate list of tensors along the "row" axis, since these tensors have shape(2,50,50)
        # and there are 8 of them in the list of tensors, the new tensor will be of shape(16,50,50)
        affinities = tf.keras.layers.Concatenate(axis=0)(affinities)

        # debug 
        # affinities_debug = tf.abs(tf.transpose(affinities, [1, 2, 0]))
        # affinitiesTensor = tf.reduce_sum(affinities_debug, axis=2)
        # affinitiesNumpy = np.divide(np.subtract(affinitiesTensor.numpy(), np.min(affinitiesTensor.numpy())), np.subtract(np.max(affinitiesTensor.numpy()), np.min(affinitiesTensor.numpy())))
        # img = Image.fromarray(affinitiesNumpy * 255)
        # img.show()

        return affinities, img_affinity

    def getAfinityCenter(self, width, height, point, center, radius=7, img_affinity=None):
        """
        Function to create the affinity maps,
        e.g., vector maps pointing toward the object center.

        Args:
            width: image wight
            height: image height
            point of the object in the pixel coordinates: (x,y)
            center of the obejct in pixel coordinates: (x,y) 
            radius of the ellipse around the image: pixel radius
            img_affinity: tensor to add to
        return:
            return a tensor
        """
        tensor = tf.keras.backend.zeros(shape=(2, width, height))

        # Create the canvas (background) for the afinity output
        imgAffinity = Image.new("RGB", (width, height), "black")

        # draw a white ellipse around the point in the radius on the imgAffinity
        draw = ImageDraw.Draw(imgAffinity)
        r1 = radius
        p = point
        draw.ellipse((p[0]-r1, p[1]-r1, p[0]+r1, p[1]+r1), (255, 255, 255))
        del draw

        # debugMarko show drawed image
        # imgAffinity.show('with elipse')

        # the array from the imgAffinity's first color collumn.
        # it has only the white ellipse on it
        # Normalize by /255
        array = (np.array(imgAffinity)/255)[:, :, 0]

        # debugMarko
        # Image.fromarray(array*255).show()

        # pixel axies angle from center to the one vertex point
        angle_vector = np.array(center) - np.array(point)   # angle by the coordinate system axises
        angle_vector = normalize(angle_vector)              # normalize with the np.linalg.norm(angle_vector, ord=1)
        affinity = np.concatenate([[array*angle_vector[0]], [array*angle_vector[1]]])   # concatenate along the axis=0

        # debugMarko
        # Image.fromarray(np.abs(affinity[0])*255).show()
        # Image.fromarray(np.abs(affinity[1])*255).show()

        # print (tensor)
        if not img_affinity is None:
            # Find the angle vector
            # print (angle_vector)
            if length(angle_vector) > 0:
                angle = py_ang(angle_vector)
            else:
                angle = 0
            # print(angle)
            c = np.array(colorsys.hsv_to_rgb(angle/360, 1, 1)) * 255
            draw = ImageDraw.Draw(img_affinity)
            draw.ellipse((p[0]-r1, p[1]-r1, p[0]+r1, p[1]+r1), fill=(int(c[0]), int(c[1]), int(c[2])))
            del draw
        # re = torch.from_numpy(affinity).float() + tensor
        re = tf.keras.backend.constant(affinity) + tensor

        # debugMarko
        # re_debug = tf.transpose(re, [1, 2, 0])
        # reTensor = tf.reduce_sum(re_debug, axis=2) * 255
        # reNumpy = reTensor.numpy() #.astype(np.uint8)
        # img = Image.fromarray(reNumpy)
        # img.show()

        return re, img_affinity