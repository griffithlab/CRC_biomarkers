#!/usr/bin/python
import sys
import os
import argparse
import shutil
import numpy as np
import time
from sklearn.externals import joblib
import random


def parse_args(argv):
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('-a', '--learning_algorithm', dest='learning_algorithm', default='nu_svm')
	parser.add_argument('-i', '--input_expr', dest='input_expr')
	parser.add_argument('-p', '--outlier_predictors', dest='outlier_predictors')
	parser.add_argument('-s', '--normal_stats', dest='normal_stats')
	parser.add_argument('-o', '--output_directory', dest='output_directory')
	parser.add_argument('-cv', '--cross_valid', dest='cross_valid', default=None, type=int)
	parsed = parser.parse_args(argv[1:])
	return parsed


def parse_data(filename, label_col, data_col_start):
	data = np.loadtxt(filename, dtype=str, delimiter='\t')
	gene_id = data[0, data_col_start:]
	sample_id = data[1:, 0]
	expr = np.array(data[1:, data_col_start:], dtype=np.float32)
	label = data[1:, label_col]
	return [gene_id, sample_id, expr, label]


def generate_cross_validation(expr_tr, label_tr, n_folds=10):
	# make sure at least one Normal, at least one CRC in each fold
	indx_N = np.where(label_tr == "N")[0]
	indx_C = np.setdiff1d(range(len(label_tr)), indx_N)
	random.shuffle(indx_N)
	random.shuffle(indx_C)
	indx_N_rand_chunks = np.array_split(indx_N, n_folds)
	indx_C_rand_chunks = np.array_split(indx_C, n_folds)
	expr_tr_rand_chuks = []
	label_tr_rand_chuks = []
	for i in range(len(indx_N_rand_chunks)):
		indx_combined = list(indx_N_rand_chunks[i]) + list(indx_C_rand_chunks[i])
		expr_tr_rand_chuks.append(expr_tr[indx_combined,:])
		label_tr_rand_chuks.append(label_tr[indx_combined])
	return (np.array(expr_tr_rand_chuks), np.array(label_tr_rand_chuks))


def get_predictor_expr(filename, expr, gene_id):
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


def parse_predictor_stats(expr):
	return (np.median(expr, axis=0), np.median(expr, axis=0), np.percentile(expr, 25, axis=0),np.percentile(expr, 75, axis=0))


def boostrap_label_group(sample_id, expr, labels):
	## boostrap label groups that have smaller number of samples to match the largest
	unique_labels = np.unique(labels)
	label_indx_dict = {}
	largest_sample_num = 0
	for ul in unique_labels:
		tmp_arr = np.where(labels == ul)[0]
		label_indx_dict[ul] = tmp_arr
		largest_sample_num = len(tmp_arr) if len(tmp_arr)>largest_sample_num else largest_sample_num
	for ul in unique_labels:
		tmp_arr = label_indx_dict[ul]
		if len(tmp_arr) != largest_sample_num:
			new_arr = np.random.choice(tmp_arr, largest_sample_num-len(tmp_arr), replace=True)
			sample_id = np.append(sample_id, sample_id[new_arr])
			expr = np.vstack((expr, expr[new_arr,]))
			labels = np.append(labels, labels[new_arr])
	return (sample_id, expr, labels)


def subsample_label_group(sample_id, expr, labels):
	## subsample label groups that have larger number of samples to match the smallest
	unique_labels = np.unique(labels)
	label_indx_dict = {}
	smallest_sample_num = 10000000000
	for ul in unique_labels:
		tmp_arr = np.where(labels == ul)[0]
		label_indx_dict[ul] = tmp_arr
		smallest_sample_num = len(tmp_arr) if len(tmp_arr)<smallest_sample_num else smallest_sample_num
	indx_to_remove = np.array([])
	for ul in unique_labels:
		tmp_arr = label_indx_dict[ul]
		if len(tmp_arr) != smallest_sample_num:
			new_arr = np.random.choice(tmp_arr, len(tmp_arr)-smallest_sample_num, replace=False)
			indx_to_remove = np.append(indx_to_remove, new_arr)
	sample_id = np.delete(sample_id, indx_to_remove)
	expr = np.delete(expr, indx_to_remove, axis=0)
	labels = np.delete(labels, indx_to_remove)
	return (sample_id, expr, labels)


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


def calculate_confusion_matrix(label_te, label_pred):
	pred_pos = np.append(np.where(label_pred == "C")[0], np.where(label_pred == "P")[0])
	pred_neg = np.where(label_pred == "N")[0]
	te_pos = np.append(np.where(label_te == "C")[0], np.where(label_te == "P")[0])
	te_neg = np.where(label_te == "N")[0]
	tps = len(np.intersect1d(pred_pos, te_pos)) 
	fps = len(np.intersect1d(pred_pos, te_neg)) 
	fns = len(np.intersect1d(pred_neg, te_pos))
	tns = len(np.intersect1d(pred_neg, te_neg))
	print tps, fps, fns, tns
	sens = tps/float(len(te_pos))
	spec = tns/float(len(te_neg))
	accu = (tps+tns)/float(len(label_te))
	return [sens, spec, accu]


def parse_cv_result(clf):
	print('CV training', np.max(clf.cv_results_['mean_train_score']))
	print('CV testing', np.max(clf.cv_results_['mean_test_score']))
	# print("")
	# print(clf.cv_results_['mean_test_score'])
	# print("")
	print(clf.cv_results_['params'][np.argmax(clf.cv_results_['mean_train_score'])])
	print(clf.cv_results_['params'][np.argmax(clf.cv_results_['mean_test_score'])])
	opt_params = clf.cv_results_['params'][np.argmax(clf.cv_results_['mean_test_score'])]
	return opt_params


def main(argv):
	# parse data
	parsed = parse_args(argv)
	if parsed.output_directory != None:
		parsed.output_directory += '/' if (not parsed.output_directory.endswith('/')) else ''
		if (os.path.exists(parsed.output_directory)):
			shutil.rmtree(parsed.output_directory)
		os.makedirs(parsed.output_directory)
	
	[gene_id, sample_id, expr_tr, label_tr] = parse_data(parsed.input_expr, 1, 2)

	label_unique= np.unique(label_tr)
	label_count = np.array([len(np.where(label_tr == l)[0]) for l in label_unique])

	print "Training set dimension:", expr_tr.shape[0], "samples x", expr_tr.shape[1], "features"
	print "True labels", label_unique, "| Counts", label_count

	time_start = time.clock()
	
	##### Random Forest #####
	if parsed.learning_algorithm.lower() ==	'random_forest':
		from sklearn.ensemble import RandomForestClassifier

		if parsed.cross_valid:
			## sklearn model selection
			from sklearn.model_selection import GridSearchCV
			rf = RandomForestClassifier()
			hyperparams = {'n_estimators': [250, 500, 1000],
							'criterion': ['gini', 'entropy'],
							'class_weight': [None, 'balanced']}
			clf = GridSearchCV(rf, hyperparams, cv=parsed.cross_valid, n_jobs=4)
			clf.fit(expr_tr, label_tr)
			params = parse_cv_result(clf)
		else:
			params = {'n_estimators': 1000,
							'criterion': 'gini',
							'class_weight': None}
			
		## train the model
		clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
										criterion=params['criterion'],
										class_weight=params['class_weight'],
										oob_score=True,
										n_jobs=4, 
										verbose=False)
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, label_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')
		
		## sort genes by importance
		num_most_important_gene = 25
		gene_score = clf.feature_importances_
		gene_index = gene_score.argsort()[-num_most_important_gene:][::-1]
		num_most_important_gene = min(num_most_important_gene, len(gene_score))


	##### C-SVM #####
	elif parsed.learning_algorithm.lower() == 'svm':
		from sklearn.svm import SVC

		if parsed.cross_valid:
			## sklearn model selection
			from sklearn.model_selection import GridSearchCV
			svm = SVC()

			from sklearn.model_selection import RandomizedSearchCV
			import scipy.stats as ss
			hyperparams = {'C': ss.expon(scale=10), #randomized parameters
							'kernel':['rbf'],
							# 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
							'class_weight': [None]}
			clf = RandomizedSearchCV(svm, hyperparams, n_iter=500, cv=parsed.cross_valid, n_jobs=4)
			
			clf.fit(expr_tr, label_tr)
			params = parse_cv_result(clf)
		else:
			params = {'C': 1.2, #1.1 for 330 samples, 4.5 for 273 samples
						'kernel': 'rbf',
						'class_weight': None}

		## train the model
		clf = SVC(C=params['C'], 
					kernel=params['kernel'], 
					class_weight=params['class_weight'],
					probability=True, 
					verbose=False)
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, label_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### Nu-SVM #####
	elif parsed.learning_algorithm.lower() == 'nu_svm':
		from sklearn.svm import NuSVC

		if parsed.cross_valid:
			## sklearn model selection
			from sklearn.model_selection import GridSearchCV
			svm = NuSVC()

			from sklearn.model_selection import RandomizedSearchCV
			import scipy.stats as ss
			hyperparams = {'nu': ss.expon(scale=10), #randomized parameters
							'kernel':['rbf'],
							# 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
							'class_weight': [None]}
			clf = RandomizedSearchCV(svm, hyperparams, n_iter=500, cv=parsed.cross_valid, n_jobs=4)
			
			clf.fit(expr_tr, label_tr)
			params = parse_cv_result(clf)
		else:
			params = {'nu': 0.82, 
						'kernel': 'rbf',
						'class_weight': 'balanced'}

		## train the model
		clf = NuSVC(nu=params['nu'], 
					kernel=params['kernel'], 
					class_weight=params['class_weight'],
					probability=True, 
					verbose=False)
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, label_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### SVR #####
	elif parsed.learning_algorithm.lower() == 'svr':
		from sklearn.svm import SVR

		if parsed.cross_valid:
			## sklearn model selection
			svr = SVR()

			from sklearn.model_selection import RandomizedSearchCV
			import scipy.stats as ss
			hyperparams = {'C': ss.expon(scale=10), #randomized parameters
							'kernel':['rbf'], # 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
							}
			clf = RandomizedSearchCV(svr, hyperparams, n_iter=500, cv=parsed.cross_valid, n_jobs=4)
			
			clf.fit(expr_tr, convert_labels(label_tr))
			params = parse_cv_result(clf)
		else:
			params = {'C': 1.1, #1.1 for 330 samples, 4.5 for 273 samples
						'kernel': 'rbf'}

		## train the model
		clf = SVR(C=params['C'], 
					kernel=params['kernel'], 
					verbose=False)
		clf.fit(expr_tr, convert_labels(label_tr))
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, convert_labels(label_tr)) #coefficient of determination R^2 of the prediction

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### Neural Network #####
	elif parsed.learning_algorithm.lower() == 'neural_net':
		from sklearn.linear_model import LogisticRegression
		from sklearn.neural_network import BernoulliRBM
		from sklearn.pipeline import Pipeline

		# train the model
		logistic = LogisticRegression(C=10)
		rbm = BernoulliRBM(n_components=256, learning_rate=.001, n_iter=100, verbose=False)
		clf = Pipeline(steps=[('rmb', rbm), ('logistic', logistic)])
		clf.fit(expr_tr, label_tr)
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### Naive Bayes #####
	elif parsed.learning_algorithm.lower() == 'naive_bayes':
		from sklearn.naive_bayes import GaussianNB	
		clf = GaussianNB()
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, label_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### Gradient Boosting #####		
	elif parsed.learning_algorithm.lower() == 'grad_boosting':
		from sklearn.ensemble import GradientBoostingClassifier

		# ## convert to two class 
		# label_tr = [1 if x=='P' or x=='C' else 0 for x in label_tr]

		if parsed.cross_valid:
			## sklearn model selection
			from sklearn.model_selection import GridSearchCV
			gb = GradientBoostingClassifier()
			hyperparams = {'learning_rate': [.01, .0075, .005, .001, .0005], 
							'max_depth': [3],
							'subsample': [1, .8, .5],
							'n_estimators': [1000]}
			clf = GridSearchCV(gb, hyperparams, cv=parsed.cross_valid, n_jobs=4)
			clf.fit(expr_tr, label_tr)
			params = parse_cv_result(clf)
		else:
			params = {'learning_rate': .0025, 
						'max_depth': 3,
						'subsample': .8,
						'n_estimators': 1000}

		## train the model
		clf = GradientBoostingClassifier(learning_rate=params['learning_rate'], 
											n_estimators=params['n_estimators'], 
											max_depth=params['max_depth'], 
											subsample=params['subsample'], 
											verbose=False)
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, label_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### AdaBoost #####
	elif parsed.learning_algorithm.lower() == "adaboost":
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.tree import DecisionTreeClassifier

		if parsed.cross_valid:
			## sklearn model selection
			from sklearn.model_selection import GridSearchCV
			ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
			hyperparams = {'learning_rate': [.01, .0075, .005, .001, .0005], 
							'n_estimators': [1000]}
			clf = GridSearchCV(ab, hyperparams, cv=parsed.cross_valid, n_jobs=4)
			clf.fit(expr_tr, label_tr)
			params = parse_cv_result(clf)
		else:
			params = {'learning_rate': .0025, 
						'n_estimators': 1000}

		## train the model
		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), 
									learning_rate=params['learning_rate'],
									n_estimators=params['n_estimators'])
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)
		accuracy_pred = clf.score(expr_tr, label_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')


	##### Gaussian Process #####
	elif parsed.learning_algorithm.lower() == 'gauss_process':
		from sklearn.gaussian_process import GaussianProcessClassifier
		from sklearn.gaussian_process.kernels import RBF

		if parsed.cross_valid:
			## sklearn model selection
			from sklearn.model_selection import GridSearchCV
			gb = GaussianProcessClassifier()
			hyperparams = {}
			clf = GridSearchCV(gb, hyperparams, cv=parsed.cross_valid, n_jobs=4)
			clf.fit(expr_tr, label_tr)
			params = parse_cv_result(clf)
		else:
			params = {}

		## train the model
		clf = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), 
										optimizer="fmin_l_bfgs_b")
		clf.fit(expr_tr, label_tr)
		label_pred = clf.predict(expr_tr)

		## save the model
		if parsed.output_directory != None:
			joblib.dump(clf, parsed.output_directory + 
				parsed.learning_algorithm.lower() + '_model.pkl')

	else:
		sys.exit('Improper learning algorithm option given.')


	## print timer messages
	time_end = time.clock()
	# print "Training time elapsed:", time_end-time_start, "sec"


if __name__ == "__main__":
    main(sys.argv)

