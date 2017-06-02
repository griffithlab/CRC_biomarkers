#!/usr/bin/python
import sys
import argparse
import numpy as np

def parse_args(argv):
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-p', '--patientinfo', dest='patientinfo')
	parser.add_argument('-d', '--chipdata', dest='chipdata')
	parsed = parser.parse_args(argv[1:])
	return parsed

def main(argv):
	parsed = parse_args(argv)

	# read data
	patientinfo = np.loadtxt(parsed.patientinfo, dtype=str, delimiter="\t")
	chipdata = np.loadtxt(parsed.chipdata, dtype=str, delimiter="\t")
	
	chipdata_chipids = chipdata[1:,0]
	patient_chipids = [x for x in patientinfo[1:,0] if len(x)==5]
	if len(np.intersect1d(chipdata_chipids, patient_chipids)) != len(chipdata_chipids):
		sys.exit("Error: Patient ID mismatch")

	# parse patient info into dictionary
	patient_dict = {}
	for p in patient_chipids:
		indx = np.where(patientinfo[1:,0] == p)[0][0]		
		if patientinfo[indx, 1].lower() == "m":
			sex = 0
		elif patientinfo[indx, 1].lower() == "f":
			sex = 1
		else:
			sex = 2
		if patientinfo[indx, 2] != "":
			age = int(patientinfo[indx, 2])
		else:
			age = 0
		if patientinfo[indx, 3].lower() == 'cauc./white':
			background = 1
		else:
			background= 0
		if patientinfo[indx, 4].lower() == "yes":
			smoking = 1
		else:
			smoking = 0
		if patientinfo[indx, 5].lower() == "yes":
			family_hist = 1
		else:
			family_hist = 0
		patient_dict[p] = [sex, age, background, smoking, family_hist]

	# append patient info to data matrix
	features_added = ['Sex', 'Age', 'Background', 'Smoking', 'FamilyHistory']
	for i in range(len(features_added)):
		tmp_entry = [features_added[i]]
		for j in range(len(chipdata_chipids)):
			tmp_entry.append(patient_dict[chipdata_chipids[j]][i])
		chipdata = np.hstack((chipdata, np.array(tmp_entry)[np.newaxis].T))

	# save data
	np.savetxt(parsed.chipdata, chipdata, fmt="%s", delimiter="\t")

if __name__ == "__main__":
    main(sys.argv)
