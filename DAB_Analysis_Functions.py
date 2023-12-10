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
        img is opencv image """
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
    
    def analyse_DAB_and_cells(self, file):
        return