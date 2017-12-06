import os
import re

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np

def getDistance(v0, v1):
	ret = 0
	for i in range(len(v0)):
		ret += (v0[i]-v1[i])**2
	return ret**0.5

def closest(attr, name):
	distance = [ [0]*len(name) for _ in range(len(name)) ]
	shortest = 1000000
	shortestI = -1
	shortestJ = -1
	for i in range(len(name)):
		for j in range(len(name)):
			if i == j:
				continue
			distance[i][j] = getDistance(attr[i], attr[j])
			if distance[i][j] < shortest:
				shortest = distance[i][j]
				shortestI = i
				shortestJ = j
	print("["+name[shortestI]+"], ["+name[shortestJ]+"] =>","%.2f"%(shortest))
	return max(shortestI, shortestJ), min(shortestI, shortestJ)

def hierarchical(name, attr):
	for i in range(len(name)):
		if len(name) == 1:
			break
		x,y = closest(attr, name)
		name.append(name[x] + "||" + name[y])
		name.pop(x)
		name.pop(y)

		tmp = [ (attr[x][j] + attr[y][j])/2 for j in range(len(attr[x]))]
		attr.append(tmp)
		attr.pop(x)
		attr.pop(y)
		#for n in name:
		#	print(n, end="  ")
		#print()

''' 
attr has following features
	0. atom count
	1. aa count
	2. atom C %
	3. atom O %
	4. atom N %
	5. atom S %
	6 - 25. aa %
	26. volume
'''
def main():
	name = []
	attr = []
	for filename in os.listdir("res"):
		with open("res/" + filename) as f:
			name.append(filename[:-4])

			feature = [0] * 27
			old_resSeq = ""
			AMINOACID = {"ALA":6, "ARG":7, "ASN":8, "ASP":9, "CYS":10, 
						"GLU":11, "GLN":12, "GLY":13, "HIS":14, "ILE":15, 
						"LEU":16, "LYS":17, "MET":18, "PHE":19, "PRO":20, 
						"SER":21, "THR":22, "TRP":23, "TYR":24, "VAL":25}
			grid = []
			for line in f:
				if re.match("^ATOM.*", line):
					if line[21] == "B":
						continue
					feature[0] += 1
					resSeq = line[22:26]
					if resSeq != old_resSeq:
						old_resSeq = resSeq
						feature[1] += 1
					if line[76:78].strip() == "C":
						feature[2] += 1
					if line[76:78].strip() == "O":
						feature[3] += 1
					if line[76:78].strip() == "N":
						feature[4] += 1
					if line[76:78].strip() == "S":
						feature[5] += 1
					if len(line[17:20].strip()) == 3:
						feature[AMINOACID[line[17:20].strip()]] += 1
					grid.append(str(int(float(line[30:38].strip())/10)) +","+str(int(float(line[38:46].strip())/10))+","+str(int(float(line[46:54].strip())/10)))
			grid = list(set(grid))
			feature[26] = len(grid)

			attr.append(feature)
	
	attr = np.array(attr)
	col_max = attr.max(axis=0)
	#print(attr)
	#print(col_max)
	#print(attr/col_max)
	
	attr = attr/col_max
	#hierarchical(name, attr)
	ytdist = attr
	Z = hierarchy.linkage(ytdist, 'single')
	#plt.figure()
	#dn = hierarchy.dendrogram(Z, labels=name,leaf_font_size=20, orientation='left')
	#plt.show()
	#plt.savefig("task2.png")
	print(Z)
if __name__ == "__main__":
	main()
