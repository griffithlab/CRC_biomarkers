#/usr/bin/python
import sys
import argparse
import numpy as np

def parse_args(argv):
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-t', '--top_genes', dest='top_genes')
	parser.add_argument('-i', '--input_expr', dest='input_expr')
	parser.add_argument('-o', '--output_expr', dest='output_expr')
	parsed = parser.parse_args(argv[1:])
	return parsed

def main(argv):
	parsed = parse_args(argv)
	input_expr = np.loadtxt(parsed.input_expr, dtype=str, delimiter="\t")
	samples = input_expr[0,1:]
	genes = input_expr[1:,0]
	if parsed.top_genes != None:
		top_genes = np.loadtxt(parsed.top_genes, dtype=str, skiprows=1, usecols=[0])
	
	sample_ids = []
	labels = []
	for i in range(len(samples)):
		tmp = samples[i].split(".")
		sample_ids.append(tmp[0])
		labels.append(tmp[1])

	input_expr = input_expr[1:,1:]
	top_indx = []
	if parsed.top_genes != None:
		for i in range(len(top_genes)):
			top_indx.append(np.where(genes == top_genes[i])[0][0])
		output_expr = np.vstack((np.append(['sample','label'], top_genes)[np.newaxis], np.hstack((np.array(sample_ids)[np.newaxis].T, np.array(labels)[np.newaxis].T, input_expr[top_indx,:].T))))
	else:
		output_expr = np.vstack((np.append(['sample','label'], genes)[np.newaxis], np.hstack((np.array(sample_ids)[np.newaxis].T, np.array(labels)[np.newaxis].T, input_expr.T))))
	
	np.savetxt(parsed.output_expr, output_expr, fmt="%s", delimiter="\t")

if __name__ == "__main__":
    main(sys.argv)