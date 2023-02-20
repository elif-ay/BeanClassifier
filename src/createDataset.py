'''
    "images" klasörü altındaki tüm tohum resimlerinden özellik veri setini çıkarır.
    Çıkarılan özellikleri csv dosyasına yazar.
'''

import cv2
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2
import skimage.feature as feature
from skimage import io
import os
from PIL import Image
from numpy import array, zeros, amax, amin, clip
from math import pi, sqrt
from timeit import itertools
from matplotlib import pyplot as plt
import sys
import csv
import pandas as pd

def GLCM(imageName):
    img = cv2.imread(imageName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    graycom = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

    contrast = feature.graycoprops(graycom, 'contrast')
    dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = feature.graycoprops(graycom, 'homogeneity')
    energy = feature.graycoprops(graycom, 'energy')
    correlation = feature.graycoprops(graycom, 'correlation')
    ASM = feature.graycoprops(graycom, 'ASM')

    return contrast, dissimilarity, homogeneity, energy, correlation, ASM

def ORB(imageName):
    orbImg = cv2.imread(imageName, 0)
    orb = cv2.ORB_create()
    orbkp = orb.detect(orbImg,None)
    orbkp, orbdes = orb.compute(orbImg, orbkp)
    orbNum = orbdes.shape[0]

    return orbNum

def SIFT(imageName):
    siftImg = cv2.imread(imageName)
    siftgray = cv2.cvtColor(siftImg, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    siftkp, siftdes = sift.detectAndCompute(siftgray,None)
    siftNum = siftdes.shape[0]

    return siftNum


def WriteToCsv():
    dataHeader = ['Contrast1', 'Contrast2', 'Contrast3', 'Contrast4', 'Dissimilarity1', 'Dissimilarity2', 'Dissimilarity3', 'Dissimilarity4', 'Homogeneity1', 'Homogeneity2', 'Homogeneity3',
                  'Homogeneity4', 'Energy1', 'Energy2', 'Energy3', 'Energy4', 'Correlation1', 'Correlation2', 'Correlation3', 'Correlation4', 'ASM1', 'ASM2', 'ASM3', 'ASM4', 'ORB', 'SIFT']


    fileName = "C:\\Users\\Elif\\Desktop\\bitirmeProjesi\\images\\"
    i = 1
    with open('dataset.csv', 'w') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(dataHeader)

        for r, d, f in os.walk(fileName):
            for file in f:
                if file.endswith(".JPG"):
                    #print(file)
                    imageName = os.path.join(r, file)

                    contrast, dissimilarity, homogeneity, energy, correlation, ASM = GLCM(imageName)

                    orbNum = ORB(imageName)

                    siftNum = SIFT(imageName)

                    dataRow = []

                    dataRow = [contrast[0][0], contrast[0][1], contrast[0][2], contrast[0][3], dissimilarity[0][0], dissimilarity[0][1], dissimilarity[0][2], dissimilarity[0][3],
                               homogeneity[0][0], homogeneity[0][1], homogeneity[0][2], homogeneity[0][3], energy[0][0], energy[0][1], energy[0][2], energy[0][3],
                               correlation[0][0], correlation[0][1], correlation[0][2], correlation[0][3], ASM[0][0], ASM[0][1], ASM[0][2], ASM[0][3], orbNum, siftNum]

                    writer.writerow(dataRow)

                    '''

                    print("-----------" + str(i) + ".)" + file)
                    i = i + 1
                    print("Contrast: {}".format(contrast))
                    print("Dissimilarity: {}".format(dissimilarity))
                    print("Homogeneity: {}".format(homogeneity))
                    print("Energy: {}".format(energy))
                    print("Correlation: {}".format(correlation))
                    print("ASM: {}".format(ASM))
                    print("ORB: {}".format(orbNum))
                    print("SIFT: {}".format(siftNum))

                    print(contrast[0][0])

                    # 1. Open a new CSV file
                    with open('students.csv', 'w') as file:
                        # 2. Create a CSV writer
                        writer = csv.writer(outputFile)

                        # 3. Write data to the file
                        writer.writerow(student_header)
                        writer.writerows(student_data)
                        file.close()

                    data = pd.read_csv('students.csv')
                    df = pd.DataFrame(data)

                    print(df)
                '''
        outputFile.close()

WriteToCsv()
