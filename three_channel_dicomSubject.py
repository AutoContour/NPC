import os
import dicom
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
import scipy
import random
import medpy

import shutil
import scipy.interpolate as si
import skimage
import codecs


class dicomSubject(object):
    def __init__(self, subject_folder_path):

        print 'subject folder path: ', subject_folder_path

        self.name = os.path.basename(subject_folder_path)
        self.image = {}
        self.image_CLAHE = {}
        self.contour = None
        self.temp_contours = None
        self.segmentation = None
        self.temp_segmentation = None
        self.origin = {}
        self.pixel_spacing = {}
        self.size = {}
        self.slice_num = {}

        self.plane_idx = []


        modalities = ['T1', 'T1C', 'T2']
        self.modalities = ['T1', 'T1C', 'T2']

        for modality in modalities:
            slices = []
            dicom_series = glob.glob(os.path.join(subject_folder_path, modality, '*.DCM'))
            for s in dicom_series:
                slices.append(dicom.read_file(s, force=True))
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

            self.origin[modality] = slices[0].ImagePositionPatient
            self.pixel_spacing[modality] = [slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]]

            raw_img = np.stack([s.pixel_array for s in slices], axis=-1)
            max_value = np.max(raw_img)
            min_value = np.min(raw_img)
            raw_img[raw_img>max_value*0.9] = max_value * 0.9
            raw_img = raw_img/(max_value*0.9)

            self.image[modality] = raw_img
            self.size[modality] = self.image[modality].shape


        # # # resize image for two modalities: T2 and T2DIXONG
        modalities.remove('T1C')
        modalities.remove('T1')
        for modality in modalities:
            image_tmp = np.zeros(self.size['T1'])
            for sle_idx in xrange(self.size[modality][2]):
                image_tmp[:,:,sle_idx] = scipy.misc.imresize(self.image[modality][:,:,sle_idx], self.size['T1'][0:2])
            self.image[modality] = image_tmp / 255.0


        self.image_CLAHE = self.image
        # # # load the label
        modality = 'T1'
        self.contour = np.zeros(self.size[modality])
        self.segmentation = np.zeros(self.size[modality])
        structure_set_file = glob.glob(os.path.join(subject_folder_path, modality, '*StrctrSets.dcm'))
        structure = dicom.read_file(structure_set_file[0], force=True)
        
        plane_idx = []

        for roi in structure.ROIContours: # loop over different anatomy type
            number = roi.ReferencedROINumber    
            for plane_contour in roi.Contours: # loop over different planes of one anatomy
                contour_points = zip(*[iter(plane_contour.ContourData)]*3)
                z_voxel = int(round((contour_points[0][2] - self.origin[modality][2]) / self.pixel_spacing[modality][2]))

                test_aa = []
                for point in contour_points:
                    x_voxel = int(round((point[0] - self.origin[modality][0]) / self.pixel_spacing[modality][0]))
                    y_voxel = int(round((point[1] - self.origin[modality][1]) / self.pixel_spacing[modality][1]))

                    test_aa.append([x_voxel,y_voxel])


                test_aa.append(test_aa[0])
                temp_contour = interplote(test_aa)
                temp_contour = np.array(temp_contour)
                self.contour[temp_contour[:,1],temp_contour[:,0],z_voxel] = 1 # mind the dimension matching
                seg = ndimage.binary_fill_holes(self.contour[:,:,z_voxel]) # fill the inside of the contour


                self.segmentation[:,:,z_voxel] = seg
                plane_idx.append(z_voxel)
                self.plane_idx = plane_idx






def load_train_negative(batch_size, n_epochs, patchSize):
    # # # # Description: Positive + Negative samples, No Overlap
    patchX, patchY, patchZ = patchSize

    data_folder = '../data'
    files = os.listdir(data_folder)
    case_num = len(files)

    for epoch in xrange(n_epochs*case_num):

        random.shuffle(files)

        train_x = np.zeros([batch_size, 3, 2*patchX, 2*patchY, 2*patchZ]) 
        train_y = np.zeros([batch_size, 2*patchX, 2*patchY, 2*patchZ])

        folder = files[epoch%case_num]

        case = dicomSubject(subject_folder_path=os.path.join(data_folder, folder))
        seg = case.segmentation

        positive = np.where(seg==1)
        positive = zip(*positive)
        random.shuffle(positive)
        sizeX, sizeY, sizeZ = seg.shape


        dummy_id = 0
        i = 0
        i_counted = 0
        while i < batch_size * 3 / 4: # and dummy_id < 30:
            x, y, z = positive[i_counted]
            if x < patchX or x > sizeX - patchX or y < patchY or y > sizeY - patchY or z < patchZ or z > sizeZ - patchZ:
                dummy_id = dummy_id + 1
                i_counted = i_counted + 1
                continue
            for idx, modality in enumerate(case.modalities):
                img = case.image[modality] - 0.2  
                train_x[i, idx, :, :, :] = img[x - patchX:x + patchX, y - patchY:y + patchY, z - patchZ:z + patchZ]
            train_y[i, :, :, :] = seg[x - patchX:x + patchX, y - patchY:y + patchY, z - patchZ:z + patchZ]
            i = i + 1
            i_counted = i_counted + 1
            print 'positive sample',i

        # crop random region
        j = 0
        while i+j < batch_size:
            x,y,z = [random.randint(patchX,sizeX-patchX),
                     random.randint(patchY,sizeY-patchY),
                     random.randint(patchZ,sizeZ-patchZ)]
            
            for idx, modality in enumerate(case.modalities):
                img = case.image[modality] - 0.2
                train_x[i+j,idx,:,:,:] = img[x-patchX:x+patchX,y-patchY:y+patchY,z-patchZ:z+patchZ]

            train_y[i+j,:,:,:] = seg[x-patchX:x+patchX,y-patchY:y+patchY,z-patchZ:z+patchZ]
            j = j + 1
            print 'negative sample',j

        print 'positive rate: ', np.mean(train_y)

        yield train_x.astype('float32'), train_y.astype('int32')








def interplote(points):
    added = []
    for i in xrange(len(points)-1):
        dist = np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))
        if dist > 1.4:
            pair = [points[i], points[i+1]]

            if np.abs(points[i][0]-points[i+1][0]) > np.abs(points[i][1]-points[i+1][1]):

                min_idx = np.argmin([points[i][0],points[i+1][0]])
                xx = np.linspace(start=pair[min_idx][0], stop=pair[1-min_idx][0], num=pair[1-min_idx][0]-pair[min_idx][0]+2, dtype='int32')
                interp = np.interp(xx, [pair[min_idx][0],pair[1-min_idx][0]], [pair[min_idx][1],pair[1-min_idx][1]])
                for dummy in zip(xx, interp):
                    added.append([int(dummy[0]),int(dummy[1])])
                
            else:
                min_idx = np.argmin([points[i][1],points[i+1][1]])
                yy = np.linspace(start=pair[min_idx][1], stop=pair[1-min_idx][1], num=pair[1-min_idx][1]-pair[min_idx][1]+2, dtype='int32')
                interp = np.interp(yy, [pair[min_idx][1],pair[1-min_idx][1]], [pair[min_idx][0],pair[1-min_idx][0]])
                for dummy in zip(interp,yy):
                    added.append([int(dummy[0]),int(dummy[1])])
                

    return [list(x) for x in set(tuple(x) for x in added+points)]

