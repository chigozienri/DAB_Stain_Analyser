#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:21:34 2023

Class related to analysis of DAB stained images

@author: jbeckwith
"""
import numpy as np
import skimage as ski

class DAB():
    def __init__(self):
        self = self
        return
        
    def imread(self, file):
        """ imread function
        takes RGB image and corrects openCV's ordering
        ================INPUTS============= 
        file is file path
        ================OUTPUT============= 
        img is array """
        img = ski.io.imread(file) # read in image
        return img

    def im2double(self, img):
        """ im2double function
        takes image and normalises to double
        ================INPUTS============= 
        img is image object
        ================OUTPUT============= 
        imdouble is numpy array """
        info = np.iinfo(img.dtype) # Get the data type of the input image
        imdouble = img.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype
        return imdouble
    
    def getMeanchannelvalues(self, image, mask):
        """ getMeanLABvalues function
        takes image, image mask and gets mean channel components 
        assumes Lab colour space
        ================INPUTS============= 
        image is numpy array
        mask is numpy logical array
        ================OUTPUT============= 
        imdouble is numpy array """
        LMean = np.nanmean(np.multiply(image[:, :, 0], mask)) # mean of only the pixels within the masked area.
        aMean = np.nanmean(np.multiply(image[:, :, 1], mask)) # mean of only the pixels within the masked area.
        bMean = np.nanmean(np.multiply(image[:, :, 2], mask)) # mean of only the pixels within the masked area.
        return LMean, aMean, bMean
    
    def pseudo_circularity(self, MajorAxisLength, MinorAxisLength):
        """ pseudo_circularity function
        takes major and minor axis length and computes pseudo-circularity
        ================INPUTS============= 
        MajorAxisLength is major axis length in pixels
        MinorAxisLength is minor axis length in pixels
        ================OUTPUT============= 
        p_circ is pseudo-circularity (runs from 0--1) """
        p_circ = np.divide(np.multiply(2, MinorAxisLength), np.add(MinorAxisLength, MajorAxisLength))
        return p_circ

    
    def clean_nuclear_mask(self, mask):
        """ clean_nuclear_mask function
        takes mask, and cleans up nuclei
        removes 3*3 (i.e. diffraction limited) objects
        clears border, connects larger aggregates if small "holes" inside, etc
        ================INPUTS============= 
        mask is logical array of image mask
        ================OUTPUT============= 
        cleaned_mask is cleaned up mask """
        mask_disk = 1*ski.morphology.binary_closing(mask, ski.morphology.disk(3))
        seed = np.copy(mask_disk)
        seed[1:-1, 1:-1] = mask_disk.max()
        
        mask_filled = ski.morphology.reconstruction(seed, mask_disk, method='erosion')
        cleaned_mask = 1*ski.morphology.binary_closing(mask_filled, ski.morphology.disk(2))
       
        from skimage.measure import label, regionprops_table
        label_img = label(cleaned_mask)
        props = regionprops_table(label_img, properties=('area',
                                                         'axis_minor_length'))
        Area = props['area']
        indices_toremove = np.unique(np.unique(label_img)[1:]*(Area < 60))[1:]
        mask=np.isin(label_img,indices_toremove)
        cleaned_mask[mask] = 0
        return cleaned_mask

    
    def clean_protein_mask(self, mask):
        """ clean_protein_mask function
        takes mask, and cleans up protein aggregates
        removes 3*3 (i.e. diffraction limited) objects
        clears border, connects larger aggregates if small "holes" inside, etc
        ================INPUTS============= 
        mask is logical array of image mask
        ================OUTPUT============= 
        cleaned_mask is cleaned up mask """
        mask_disk = 1*ski.morphology.binary_closing(mask, ski.morphology.disk(1))
        seed = np.copy(mask_disk)
        seed[1:-1, 1:-1] = mask_disk.max()
        
        mask_filled = ski.morphology.reconstruction(seed, mask_disk, method='erosion')
        cleaned_mask = ski.segmentation.clear_border(mask_filled)
        
        from skimage.measure import label, regionprops_table
        label_img = label(cleaned_mask)
        props = regionprops_table(label_img, properties=('area',
                                                         'axis_minor_length'))
        minorA = props['axis_minor_length']
        Area = props['area']
        indices_toremove = np.unique(np.hstack([np.unique(label_img)[1:]*(minorA < 3), np.unique(label_img)[1:]*(Area < 9)]))[1:]
        mask=np.isin(label_img,indices_toremove)
        cleaned_mask[mask] = 0
        return cleaned_mask
    
    def colourFilterLab(self, image, initial_params, rate=[0.75, 4], percentage=0.075, maxiter=30):
        """ colourFilterLab function
        takes image, and uses initial parameters and rate to separate out
        coloured objects
        ================INPUTS============= 
        image is numpy array
        initial_params are initial Lmean, aMean, bMean and threshold parameters
        rate is how fast to change the parameters per iteration
        percentage is the percentage of pixels we worry about in the mask
        maxiter is how many iterations of optimisation to do (default 30)
        ================OUTPUT============= 
        image_mask is logical array of image mask
        current_params are image analysis final params """
        mask_current = np.zeros(image[:,:,0].shape);
        LMean = initial_params[0];
        aMean = initial_params[1];
        bMean = initial_params[2];
        thres = initial_params[3];
        ratep_1 = rate[0];
        ratep_2 = rate[1]; 
        
        iteration = 1
        while (np.nansum(mask_current) < np.multiply(percentage, image[:,:,0].size)) and (iteration <= maxiter):
            deltaL = np.subtract(image[:,:,0], LMean) # get mean channel 1
            deltaa = np.subtract(image[:,:,1], aMean) # get mean channel 2
            deltab = np.subtract(image[:,:,2], bMean) # get mean channel 3
            deltaE = np.sqrt((deltaL**2 + deltaa**2 + deltab**2)) # get change versus iteration
            mask_previous = mask_current # store previous mask
            mask_current = 1*(deltaE <= thres) # update mask
            mask_current = np.where(mask_current==0, np.nan, mask_current)
            LMean, aMean, bMean = self.getMeanchannelvalues(image, mask_current) # get new means
            LMean = np.multiply(LMean, ratep_1) # adjust means with rate of change
            aMean = np.multiply(aMean, ratep_1) # adjust means with rate of change
            bMean = np.multiply(bMean, ratep_1) # adjust means with rate of change
            
            dEmask = np.multiply(deltaE, mask_current)
            meanMaskedDeltaE  = np.nanmean(dEmask)
            stDevMaskedDeltaE = np.nanstd(dEmask)
            thres = np.add(meanMaskedDeltaE, np.multiply(ratep_2, stDevMaskedDeltaE))
            iteration += 1
            
        image_mask = mask_previous # update mask
        image_mask[np.isnan(image_mask)] = 0
        current_params = np.array([LMean, aMean, bMean, thres])
        return image_mask, current_params
    
    def analyse_DAB(self, file, asyn_params=np.array([27, 6, 5, 15]), use_defaults=1, check_mask=1):
        """ analyse_DAB function
        takes file, and uses initial parameters and rate to separate out
        coloured protein objects
        then returns table with object information
        ================INPUTS============= 
        file is filename
        asyn_params are initial default Lmean, aMean, bMean and threshold parameters
        use_defaults is boolean that is 1 (use the default parameters to separate colours) or 0 (get user selection)
        check_mask outputs a figure that shows the mask---if you don't like it, you can re-run
        ================OUTPUT============= 
        asyn_params is parameters used to get particular mask
        table_asyn is pandas array of asyn data """
        img = ski.io.imread(file)
        lab_Image = ski.color.rgb2lab(self.im2double(img))
        if use_defaults == 0:
            init_guess = self.get_guess(self, img, lab_Image)
            asyn_params = np.hstack([init_guess], asyn_params[-1])
            
        image_mask_asyn, asyn_params = self.colourFilterLab(lab_Image, asyn_params)
        image_mask_asyn = self.clean_protein_mask(image_mask_asyn)
        if check_mask == 1:
            masks = image_mask_asyn
            self.plot_masks(img, masks)
        from skimage.measure import label, regionprops_table
        label_img_asyn = label(image_mask_asyn)
        props_asyn = regionprops_table(label_img_asyn, properties=('area',
                                                         'centroid',
                                                         'axis_major_length',
                                                         'axis_minor_length'))
        import pandas as pd
        table_asyn = pd.DataFrame(props_asyn)
        table_asyn['pseudo_circularity'] = self.pseudo_circularity(props_asyn['axis_major_length'], props_asyn['axis_minor_length'])
        return table_asyn, asyn_params
    
    def analyse_DAB_and_cells(self, file, asyn_params=np.array([27, 6, 5, 15]), nuclei_params=np.array([70, 1, -5, 4]), use_defaults=1, check_mask=1):
        """ analyse_DAB_and_cells function
        takes file, and uses initial parameters and rate to separate out
        coloured objects
        then returns table with object information
        ================INPUTS============= 
        file is filename
        asyn_params are initial default Lmean, aMean, bMean and threshold parameters
        nuclei_params are initial default Lmean, aMean, bMean and threshold parameters
        ================OUTPUT============= 
        table_asyn is pandas array of asyn data
        table_nuclei is pandas array of nuclei data
        asyn_params is parameters used to gets asyn mask
        nuclei_params is parameters used to get nuclear mask """
        img = ski.io.imread(file)
        lab_Image = ski.color.rgb2lab(self.im2double(img))
        if use_defaults == 0:
            init_guess = self.get_guess(self, img, lab_Image)
            asyn_params = np.hstack([init_guess], asyn_params[-1])
            init_guess_n = self.get_guess(self, img, lab_Image, "select area that is just nuclear staining; press enter when complete")
            nuclei_params = np.hstack([init_guess_n], nuclei_params[-1])
            
        image_mask_asyn, asyn_params = self.colourFilterLab(lab_Image, asyn_params)
        image_mask_asyn = self.clean_protein_mask(image_mask_asyn)
        image_mask_nuclei, nucl_params = self.colourFilterLab(lab_Image, nuclei_params, rate=[1,2])
        image_mask_nuclei = self.clean_nuclear_mask(image_mask_nuclei)
        if check_mask == 1:
            masks = np.dstack([image_mask_asyn, image_mask_nuclei])
            self.plot_masks(img, masks)
        from skimage.measure import label, regionprops_table
        label_img_asyn = label(image_mask_asyn)
        props_asyn = regionprops_table(label_img_asyn, properties=('area',
                                                         'centroid',
                                                         'axis_major_length',
                                                         'axis_minor_length'))
        import pandas as pd
        table_asyn = pd.DataFrame(props_asyn)
        table_asyn['pseudo_circularity'] = self.pseudo_circularity(props_asyn['axis_major_length'], props_asyn['axis_minor_length'])
        
        label_img_nucl = label(image_mask_nuclei)
        props_nuclei = regionprops_table(label_img_nucl, properties=('area',
                                                         'centroid',
                                                         'axis_major_length',
                                                         'axis_minor_length'))
        table_nuclei = pd.DataFrame(props_nuclei)
        table_nuclei['pseudo_circularity'] = self.pseudo_circularity(props_nuclei['axis_major_length'], props_nuclei['axis_minor_length'])
        return table_asyn, table_nuclei, asyn_params, nuclei_params
    
    def get_guess(self, img, lab_image, message="select area that is just DAB-stained protein aggregate; press enter when complete"):
        import cv2
        r = cv2.selectROI(message, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

        cv2.destroyAllWindows()
        area = lab_image[int(r[1]):int(r[1]+r[3]),  
                              int(r[0]):int(r[0]+r[2])] 
        init_guess = np.mean(np.mean(area, axis=1), axis=0)
        return init_guess
    
    def plot_masks(self, img, masks):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        axes[0].imshow(img)
        
        axes[1].imshow(img)
        if len(masks.shape) > 2: # if multiple masks
            colors = ['darkred', 'darkblue']
            for i in np.arange(masks.shape[2]): # plot multiple masks
                axes[1].contour(masks[:, :, i], [0.5], linewidths=0.5, colors=colors[i])
        else:
            axes[1].contour(masks, [0.5], linewidths=0.5, colors='darkred')

        
        for a in axes:
            a.axis('off')
        
        plt.tight_layout()
        
        plt.show()
        return