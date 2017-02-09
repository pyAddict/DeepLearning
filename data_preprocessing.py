from __future__ import print_function

modelStringList = [
    ['iphone', 'samsung']]
wayBillDescFile = 'prod_desc_3.txt'
fakeBillsFile = 'tracker-dgw.txt'
maxLen = 24000
impurity_level = 0.5

import cv2
import numpy as np
from sklearn.cross_validation import train_test_split

from scipy import misc
import os
import glob
from random import shuffle, randrange, sample

wayBillDescriptions = {}
fakeBills = []
files = glob.glob("./new-processed/*.jpg")

def read_way_bill_desc(path):
    with open(path, 'r') as myFile:
        for line in myFile:
            line = line.strip()
            data = line.split(
                '\t')
            if len(data) >= 2:
                wayBillDescriptions[
                    str(data[0])] = str(data[1])
    # return
    # wayBillDescriptions

def isModelString(desc, modelStrings):
    '''
      Given a description, checks whether it is in accordance with the modelStringList
    '''
    for stringList in modelStrings:
        isModelDesc = True
        for string in stringList:
            if string not in desc.lower():
                isModelDesc = False
                break
        if isModelDesc:
            return True
    return False

def isModelDescription(desc, modelStrings):
    if not isModelString(desc, modelStrings):
        return False
    return ("clone" not in desc.lower()) and ("average" not in desc.lower()) and ("refurbished" not in desc.lower()) \
    and ("badimage" not in desc.lower())

def isModelFile(fileName, modelStrings):
    '''
       Given a filename like 6158012_<waybill>_SERVICE2.jpg,
       extracts the waybill number and checks whether the modelStrings are present in
       the wayBill's description.
    '''
    # Extract
    # Waybill Number
    fileNameParts = fileName.split(
        '_')
    if len(fileNameParts) < 3:
        return False
    wayBill = str(
        fileNameParts[1])

    # Get the
    # description
    if wayBill not in wayBillDescriptions:
        return False
    desc = wayBillDescriptions[
        wayBill]

    if not isModelString(desc, modelStrings):
        return False

    return ("clone" not in desc.lower()) and ("average" not in desc.lower()) and ("refurbished" not in desc.lower()) and ("badimage" not in desc.lower())



def create_fake_bills(path):
	with open(fakeBillsFile, 'r') as myFile:
	    for line in myFile:
	        line = line.strip()
	        row = line.split(
	            '|')
	        if len(row) > 3 and (str(row[2]) in wayBillDescriptions) and isModelDescription(str(row[3]), modelStringList):
	            fakeBills.append(
	                str(row[2]))


def isFakeFile(fileName):
    '''
       Given a filename, checks whether the extracted waybill is in the fakeBills List
    '''
    fileNameParts = fileName.split(
        '_')
    wayBill = str(
        fileNameParts[1])
    if wayBill in fakeBills:
        return True
    return False


def create_index():
	read_way_bill_desc(
	    wayBillDescFile)
	create_fake_bills(
	    fakeBillsFile)
	ind_iphone = []
	ind_samsung = []
	ind_others = []
	for f in range(len(files)):
		if isFakeFile(files[f]):
			continue
		bil_no = str(files[f]).split('_')[1]
		if bil_no not in wayBillDescriptions.keys():
			continue
		desc = wayBillDescriptions[bil_no]
		if isModelDescription(desc,[['iphone']]):
			ind_iphone.append(f)
		elif isModelDescription(desc,[['samsung']]):
			ind_samsung.append(f)
		else:
			ind_others.append(f)
	return np.array(ind_iphone),np.array(ind_samsung),np.array(ind_others)

def sampling():
	ind_iphone,ind_samsung,ind_others = create_index()
	i,dc = train_test_split(ind_iphone,test_size=impurity_level)
	j,dc = train_test_split(ind_samsung,test_size=impurity_level)
	dataInd = np.concatenate([i,j,ind_others])
	dataInd = dataInd[:maxLen]
	return dataInd


def process_data(typ=None):
	dataInd = sampling()
	file = np.array(files)
	netFiles = file[dataInd]
	np.random.shuffle(netFiles)
	part = int(maxLen/3)
	return netFiles,(typ-1)*part, (typ*part)


def check_code(data):
	cnt1 = 0
	cnt2 = 0
	cnt3 = 0
	for d in data:
		bil = d.split('_')[1]
		desc = wayBillDescriptions[bil]
		if isModelString(desc,[['iphone']]):
			cnt1+=1
		if isModelString(desc,[['samsung']]):
			cnt2+=1
		else:
			cnt3+=1
	print('iphone',cnt1,'samsung',cnt2,'others',cnt3)




data,ind1,ind2 = process_data(typ=1)
check_code(data)




