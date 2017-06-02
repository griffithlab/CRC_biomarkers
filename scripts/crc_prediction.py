#/usr/bin/python
import sys
import argparse
import numpy as np
from sklearn.externals import joblib


def parse_args(argv):
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('-a', '--learning_algorithm', dest='learning_algorithm', default='nu_svm')
	parser.add_argument('-i', '--input_expr', dest='input_expr')
	parser.add_argument('-p', '--outlier_predictors', dest='outlier_predictors')
	parser.add_argument('-s', '--normal_stats', dest='normal_stats')
	parser.add_argument('-m', '--model_filename', dest='model_filename')
	parser.add_argument('-o', '--output_tbl', dest='output_tbl', default=False)
	parsed = parser.parse_args(argv[1:])
	return parsed


def parse_data(filename, label_col, data_col_start):
	data = np.loadtxt(filename, dtype=str, delimiter='\t')
	gene_id = data[0, data_col_start:]
	sample_id = data[1:, 0]
	expr = np.array(data[1:, data_col_start:], dtype=np.float32)
	label = data[1:, label_col]
	return [gene_id, sample_id, expr, label]


def get_predictor_expr(filename, expr, gene_id):
	print expr.shape
	f = open(filename, "r")
	lines = f.readlines()
	tc_predictors_indx = []
	tc_predictors = []
	for line in lines:
		tmp_arr = line.strip().split("\t")
		if len(tmp_arr) > 1:
			for tmp_tc in tmp_arr[1].split(","):
				if tmp_tc in gene_id:
					tmp_indx = np.where(gene_id == tmp_tc)[0][0]
					tc_predictors_indx.append(tmp_indx)
					tc_predictors.append(tmp_tc)
	return (tc_predictors, expr[:, tc_predictors_indx])


def parse_normal_stats(filename):
	stats = np.loadtxt(filename, skiprows=1, dtype=str)
	stats_dict = {}
	for i in range(len(stats)):
		stats_dict[stats[i,0]] = [float(stats[i,3]), float(stats[i,4])]
	return stats_dict


def convert_labels(labels):
	conv_labels = -1 * np.ones(len(labels))
	for i in range(len(labels)):
		if labels[i] == 'N':
			conv_labels[i] = 0
		elif labels[i] == 'P':
			conv_labels[i] = 1
		elif labels[i] == 'C':
			conv_labels[i] = 2
	return conv_labels


def convert_reversed_labels(labels):
	conv_labels = np.chararray(len(labels))
	for i in range(len(labels)):
		if labels[i] < .5:
			conv_labels[i] = "N"
		elif labels[i] >= .5 and labels[i] < 1.5:
			conv_labels[i] = "P"
		elif labels[i] >= 1.5:
			conv_labels[i] = "C"
	return conv_labels


def calculate_confusion_matrix(label_te, label_pred):
	pred_pos = np.append(np.where(label_pred == "C")[0], np.where(label_pred == "P")[0])
	pred_neg = np.where(label_pred == "N")[0]
	te_pos = np.append(np.where(label_te == "C")[0], np.where(label_te == "P")[0])
	te_neg = np.where(label_te == "N")[0]
	tps = len(np.intersect1d(pred_pos, te_pos)) 
	fps = len(np.intersect1d(pred_pos, te_neg)) 
	fns = len(np.intersect1d(pred_neg, te_pos))
	tns = len(np.intersect1d(pred_neg, te_neg))
	print 'TPs', tps, 'FPs', fps, 'FNs', fns, 'TNs', tns
	sens = tps/float(len(te_pos))
	spec = tns/float(len(te_neg))
	accu = (tps+tns)/float(len(label_te))
	return [sens, spec, accu]


def print_accuracies(sens, spec, accu):
	print 'Sens %.1f%% Spec %.1f%% Accu %.1f%%' % (sens*100, spec*100, accu*100)


def print_output(filename, summary):
	writer = open(filename, 'w')
	writer.write("#sample_id\ttrue_label\tpredict_label\n")
	for i in range(len(summary)):
		writer.write("%s\t%s\t%s\n" % (summary[i,0], summary[i,1], 
			'Negative' if summary[i,3]=='N' else 'Positive'))
	writer.close()


def main(argv):
	# parse data
	parsed = parse_args(argv)

	[gene_id, sample_id, expr_te, label_te] = parse_data(parsed.input_expr, 1, 2)

	label_unique= np.unique(label_te)
	label_count = np.array([len(np.where(label_te == l)[0]) for l in label_unique])

	print "Testing set dimension:", expr_te.shape[0], "samples x", expr_te.shape[1], "features"
	print "True labels", label_unique, "| Counts", label_count 

	print "----------------------------------------------"
	print "Testing results:"


	##### Random Forest #####
	if parsed.learning_algorithm.lower() ==	'random_forest':
		from sklearn.ensemble import RandomForestClassifier
		
		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T))
			
		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### SVM #####
	elif parsed.learning_algorithm.lower() == 'svm':
		from sklearn.svm import SVC

		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T, probability_predicted))

		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### Nu-SVM #####
	elif parsed.learning_algorithm.lower() == 'nu_svm':
		from sklearn.svm import NuSVC

		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T, probability_predicted))

		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### SVR #####
	elif parsed.learning_algorithm.lower() == 'svr':
		from sklearn.svm import SVR

		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		accuracy_predicted = clf.score(expr_te, convert_labels(label_te))
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, convert_reversed_labels(label_predicted)[np.newaxis].T))

		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, convert_reversed_labels(label_predicted))
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### Neural Network #####
	elif parsed.learning_algorithm.lower() == 'neural_net':
		from sklearn.neural_network import BernoulliRBM

		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		accuracy_predicted = len([label_predicted[i] for i in range(len(label_predicted)) if (label_predicted[i] == label_te[i])]) / float(len(label_predicted))

		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### Naive Bayes #####
	elif parsed.learning_algorithm.lower() == 'naive_bayes':
		from sklearn.naive_bayes import GaussianNB	

		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T, probability_predicted))
		
		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### Gradient Boosting #####
	elif parsed.learning_algorithm.lower() == 'grad_boosting':
		from sklearn.ensemble import GradientBoostingClassifier
		
		# ## convert to two class
		# label_te = np.array([1 if x=='P' or x=='C' else 0 for x in label_te])

		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T, probability_predicted))

		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### AdaBoost #####
	elif parsed.learning_algorithm.lower() == "adaboost":
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier

		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T, probability_predicted))
		
		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)


	##### Gaussian Process #####
	elif parsed.learning_algorithm.lower() == 'gauss_process':
		from sklearn.gaussian_process import GaussianProcessClassifier
		from sklearn.gaussian_process.kernels import RBF
		
		# predict on validation set
		clf = joblib.load(parsed.model_filename)
		label_predicted = clf.predict(expr_te)
		probability_predicted = clf.predict_proba(expr_te)
		accuracy_predicted = clf.score(expr_te, label_te)
		summary = np.hstack((sample_id[np.newaxis].T, label_te[np.newaxis].T, label_predicted[np.newaxis].T, probability_predicted))

		# print output message
		[sens, spec, accu] = calculate_confusion_matrix(label_te, label_predicted)
		print_accuracies(sens, spec, accu)
		print_output(parsed.output_tbl,  summary)

	else:
		sys.exit('Improper learning algorithm option given.')

	print "----------------------------------------------"

if __name__ == "__main__":
    main(sys.argv)

