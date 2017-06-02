#!/usr/bin/python
import sys
import argparse
import numpy as np

def parse_args(argv):
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-i', '--input_expr', dest='input_expr')
	parser.add_argument('-l', '--transcript_cluster_list', dest='transcript_cluster_list')
	parser.add_argument('-c', '--column_number', dest='column_number', type=int)
	parser.add_argument('-o1', '--output_expr_filtered', dest='output_expr_filtered')
	parser.add_argument('-o2', '--output_expr_full', dest='output_expr_full')
	parsed = parser.parse_args(argv[1:])
	return parsed

def main(argv):
	parsed = parse_args(argv)

	##load gene filtering dictionary
	f = open(parsed.transcript_cluster_list, "r")
	lines = f.readlines()
	tcs = set()
	for line in lines:
		tmp_arr = line.strip().split("\t")
		if len(tmp_arr) > parsed.column_number:
			for tmp_tc in tmp_arr[parsed.column_number].split(","):
				tcs.add(tmp_tc)
	f.close()
	tcs = list(tcs)
	print "Transript cluster count:", len(tcs)

	##load input expression data 
	expr = np.loadtxt(parsed.input_expr, dtype=str, delimiter="\t")
	
	##modify header (sample_id.label)
	for i in range(1,expr.shape[1]):
		tmp = expr[0,i].split(" ")
		if len(tmp) == 1:
			sample = tmp[0]
		else:
			sample = tmp[0] + "." + tmp[1].split(".")[0]
		expr[0,i] = sample

	##filter genes
	tc_indx_filtered = [0]
	tc_indx_full = [0]
	for i in range(1,expr.shape[0]):
		if expr[i,0] in tcs:
			tc_indx_filtered.append(i)
		if expr[i,0].startswith("TC"):
			tc_indx_full.append(i)

	##save output expr
	np.savetxt(parsed.output_expr_filtered, expr[tc_indx_filtered,:], fmt="%s", delimiter="\t")
	np.savetxt(parsed.output_expr_full, expr[tc_indx_full,:], fmt="%s", delimiter="\t")

if __name__ == "__main__":
    main(sys.argv)
