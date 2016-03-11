# Cell cycle classification and feature selection - perform feature selection and prediction with internal and external cross-validation; results are saved as figures and in a hdf5 file.
# 

########
######## PACKAGE LOADING
########
from __future__ import print_function
import h5py
import sys
import os
from os.path import exists
sys.path.append('./')
import scipy as SP
import numpy
from numpy import inf
import pdb
import matplotlib as mpl
if 0:
	mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as PL
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, svm, ensemble, linear_model, preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC, l1_min_c
from sklearn.feature_selection import RFECV, SelectKBest

from sklearn import metrics, ensemble
from sklearn.cross_validation import StratifiedKFold, LeaveOneOut
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')
# hacky hack we need for Windows users
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
	
class cyclone:
	"""
	cyclone
	This class takes care of assigning cell cycle stage based on cell cycle annotated genes in the  transcriptome.  
	"""
	def __init__(self,Y,
				row_namesY=None,
				cc_geneNames=None,
				labels=None, 
				Y_tst=None, 
				row_namesY_tst=None, 
				labels_tst=None, 
				learn_reverse=False, 
				norm='rank'):
			
		print("Initialise model...")
		assert row_namesY != None, 'cyclone: provide gene names for Y'
		assert cc_geneNames != None, 'cyclone: provide cell cycle genes'
		assert labels != None, 'cyclone: provide labels for training'
		assert norm in ['rank', 'tc', 'median', 'none'] , 'normalisation has to be either "rank" '
		'(rank normalisation) or "tc" (total count) or "median" (log2 fold change to median) o "none" (already normalized)' 

		cc_ens = cc_geneNames
		genes = row_namesY 
		genesT = row_namesY_tst
		# Take from the data frame the genes that are in the intersection 
		# and normalize them
		ind_cc_tr = SP.zeros((len(cc_ens),))
		if Y_tst != None and row_namesY_tst != None:
			ind_cc_tr = SP.zeros((len(cc_ens),))
			ind_cc_ts = SP.zeros((len(cc_ens),))
			for i in range(len(cc_ens)):
				ind_cc_ts[i] = SP.where(map(lambda x:x==cc_ens[i], genesT))[0][0]
				ind_cc_tr[i] = SP.where(map(lambda x:x==cc_ens[i], genes))[0][0]
			inds_tr = ind_cc_tr.astype('int32')
			inds_ts = ind_cc_ts.astype('int32')
			Y_tst = Y_tst[:,inds_ts]
			Y = Y[:,inds_tr]
			if norm == 'rank':
				for ir in range(Y_tst.shape[0]):
					Y_tst[ir,:] = SP.stats.rankdata(Y_tst[ir,:], method='average')		
				for ir in range(Y.shape[0]):
					Y[ir,:] = SP.stats.rankdata(Y[ir,:], method='average')
			elif norm == 'tc':
		   		Y_tst = SP.transpose(SP.transpose(Y_tst)/SP.sum(Y_tst,1))	 
				Y = SP.transpose(SP.transpose(Y)/SP.sum(Y,1))	
			elif norm == 'median':	
				for ir in range(Y_tst.shape[0]):
					med_ = SP.median(Y_tst[ir,:])
					Y_tst[ir,:] = SP.log2(Y_tst[ir,:]/med_)		
				for ir in range(Y.shape[0]):
					med_ = SP.median(Y[ir,:])
					Y[ir,:] = SP.log2(Y[ir,:]/med_)		
				Y[Y==-inf] = min(Y[Y>-inf])-1 
				Y_tst[Y_tst==-inf] = min(Y_tst[Y_tst>-inf])-1
			else:
				print("Not normalizing data ...")
			
			# Swap train and test
			if learn_reverse == True:	
				labels_ = labels_tst
				Y_ = Y_tst
				Y_tst = Y
				labels_tst = labels
				Y = Y_
				labels = labels_
				
			# Assign variables for the classifier
			self.Y = Y
			self.Y_tst = Y_tst
			self.scores = None
			self.labels = labels
			self.labels_tst = labels_tst
			self.numClasses_tst = len(SP.unique((labels_tst)))
			self.inds_tr = inds_tr
			self.inds_tst = inds_ts
		else:
			ind_cc_tr = SP.zeros((len(cc_ens),))
			for i in range(len(cc_ens)):
				ind_cc_tr[i] = SP.where(map(lambda x:x==cc_ens[i], cc_genes))[0][0]
			inds_tr = ind_cc_tr.astype('int32')
			for ir in range(Y.shape[0]):
				Y[ir,:] = SP.stats.rankdata(Y[ir,:])
			self.Y = Y[:,inds_tr]
			slef.Y_tst = None
			
		self.geneNames = cc_geneNames
		self.numClasses = len(SP.unique((labels)))

	def trainModel(self, 
				do_pca=False, 
				out_dir='./cache', 
				rftop=40, 
				cv=10, 
				npc=3):
		
		if not os.path.exists(out_dir): os.makedirs(out_dir)	
		
		Y = self.Y
		Y_tst = self.Y_tst if self.Y_tst is not None else self.Y
		labels = self.labels
		labels_test = self.labels_tst
		var_names = self.geneNames
		numClasses = self.numClasses
		numElements = len(labels)
		predRF = SP.zeros((numElements, numClasses))
		predGNB = SP.zeros((numElements, numClasses))
		predLR = SP.zeros((numElements, numClasses))
		predLRall = SP.zeros((numElements, numClasses))
		
		# Compute CV list
		cv_object = LeaveOneOut(len(labels)) if cv == "LOOCV" else StratifiedKFold(labels, n_folds=cv)
		CV_list = list(iter(cv_object))
		
		print("Performing cross validation ...")
		num_iterations = len(CV_list) + 1
		for i in range(num_iterations):
			last_iteration = (i == num_iterations - 1)
			if not last_iteration:
				print("Fold %s of %s" % (str(i+1),str(num_iterations-1)))
			else:
				print("Final model")
				
			if not last_iteration:
				# get data of a CV run
				cv_tr = CV_list[i][0]
				cv_tst = CV_list[i][1]
				lab_tr = labels[cv_tr]
				lab_tst = labels[cv_tst]
				Ytr = Y[cv_tr,:]
				Ytst = Y[cv_tst,:]
			else:
				lab_tr = labels
				lab_tst = labels_test
				Ytr = Y
				Ytst = Y_tst
				
			if do_pca:
				print("  Computing PCA ...")
				# do PCA to get features
				pcaCC = PCA(n_components=npc, whiten=False)
				pcaCC.fit(Ytr)
				pcaTst = pcaCC.transform(Ytst)
				pcaTr = pcaCC.transform(Ytr)
				combined_features = FeatureUnion([("pca", pcaCC)])
				gnb = GaussianNB()
				y_pred = gnb.fit(pcaTr, lab_tr).predict_proba(pcaTst)
				if not last_iteration:
					predGNB[cv_tst,:] = y_pred
				else:
					predGNB_ts = y_pred		
		
			# Do lasso with regularisation path
			cs = l1_min_c(Ytr, lab_tr, loss='log') * SP.logspace(0, 3)
			print("  Computing Linear Regression ...")
			# Linear Regression
			lasso = linear_model.LogisticRegression(C=cs[0]*10.0, penalty='l1', tol=1e-6)
			param_grid = dict(C=cs)
			clf_lr = GridSearchCV(lasso, param_grid=param_grid, cv=5, scoring='f1')
			clf_lr.fit(Ytr, lab_tr)
			clf_lr.best_estimator_.fit(Ytr, lab_tr)
			predicted = clf_lr.best_estimator_.predict_proba(Ytst)
			if not last_iteration:
				predLR[cv_tst,:] = predicted
			else:
				predLR_ts = predicted
			# Linear Regression All
			clfAll = linear_model.LogisticRegression(C=1e5, penalty='l2', tol=1e-6)
			clfAll.fit(Ytr, lab_tr)
			predicted = clfAll.predict_proba(Ytst)
			if not last_iteration:
				predLRall[cv_tst,:] = predicted
			else:
				predLRall_ts = predicted
				
			print("  Computing random forest ...")
			forest = ExtraTreesClassifier(n_estimators=500,
										  random_state=0, criterion="entropy", bootstrap=False)
			forest.fit(Ytr, lab_tr)
			pred = forest.predict_proba(Ytst)
			if not last_iteration:
				predRF[cv_tst,:] = pred
			else:
				predRF_ts = pred
			importances = forest.feature_importances_
			std = SP.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
			topfeat = min(Ytr.shape[1], rftop)
			indices = SP.argsort(importances)[::-1][0:topfeat]
			# store full feature ranking
			featrank_rf = SP.argsort(importances)[::-1]
	
		# Output and write results
		f2 = open(os.path.join(out_dir, 'classification_reportCV.txt'),'w')
		
		# Random Forest (Argmax chose the column (class) with the higesh score)
		predRFv = SP.argmax(predRF, axis=1) + 1
		predRF_trv = SP.argmax(predRF_ts, axis=1) + 1
		self.scores = predRF
		self.scores_tst = predRF_ts
		self.ranking = var_names[indices]
		numpy.savetxt(os.path.join(out_dir, 'RF_scores.txt'), predRF)
		numpy.savetxt(os.path.join(out_dir, 'RF_scores_test.txt'), predRF_ts)
		
		# Logistic Regression
		predLRv = SP.argmax(predLR, axis=1) + 1
		predLR_trv = SP.argmax(predLR_ts, axis=1) + 1
		self.scoresLR = predLR
		self.scoresLR_tst = predLR_ts
		numpy.savetxt(os.path.join(out_dir, 'LR_scores.txt'), predLR)
		numpy.savetxt(os.path.join(out_dir, 'LR_scores_test.txt'), predLR_ts)
		
		# Logistic Regression All
		predLRallv = SP.argmax(predLRall, axis=1) + 1
		predLRall_trv = SP.argmax(predLRall_ts, axis=1) + 1
		self.scoresLRall = predLRall
		self.scoresLRall_tst = predLRall_ts
		numpy.savetxt(os.path.join(out_dir, 'LRAll_scores.txt'), predLRall)
		numpy.savetxt(os.path.join(out_dir, 'LRAll_scores_test.txt'), predLRall_ts)
		
		# PCA + Gaussian Naive Bayes
		predGNBv = SP.argmax(predGNB, axis=1) + 1
		predGNB_trv = SP.argmax(predGNB_ts, axis=1) + 1
		self.scoresGNB = predGNB
		self.scoresGNB_tst = predGNB_ts
		numpy.savetxt(os.path.join(out_dir, 'Gausian_scores.txt'), predGNB)
		numpy.savetxt(os.path.join(out_dir, 'Gausian_scores_test.txt'), predGNB_ts)
		
		print("Classification report for classifier %s:\n%s\n" 
			% ('Gaussian Naive Bayes', metrics.classification_report(labels, predGNBv)))
		print("Classification report for classifier %s:\n%s\n" 
			% ('Random Forest', metrics.classification_report(labels, predRFv)))
		print("Classification report for classifier %s:\n%s\n" 
			% ('LR', metrics.classification_report(labels, predLRv)))
		print("Classification report for classifier %s:\n%s\n" 
			% ('LRall', metrics.classification_report(labels, predLRallv)))
		f2.close()

	def writePrediction(self, original_test_labels, output, out_dir="./", method="RF"):
		if method=='RF':
			scores = self.scores_tst
		elif method=='LR':
			scores = self.scoresLR_tst
		elif method=='LRall':
			scores = self.scoresLRall_tst
		elif method=='GNB':
			scores = self.scoresGNB_tst
		with open(os.path.join(out_dir,output), "w") as filehandler:
			for i,class_index in enumerate(SP.argmax(scores, axis=1) + 1):
				filehandler.write(str(class_index) + "\t" 
								+ str(original_test_labels[i] + "\t") 
								+ str(self.labels_tst[i]) + "\n")
			
	def plotScatter(self, 
				file_name='scatter_ts',
				out_dir='./cache',
				plot_test=False, 
				xaxis=0,
				yaxis=2, 
				xlab='G1 score', 
				ylab='G2M score', 
				class_labels=['G1 phase','S phase','G2M phase'], 
				method='RF', 
				decision_lines=True):	
		plparams = {'backend': 'pdf',
		  'axes.labelsize': 14,
		  'text.fontsize': 14,
		  'legend.fontsize': 13,
		  'xtick.labelsize': 14,
		  'ytick.labelsize': 14,
		  'text.usetex': False}
		PL.rcParams.update(plparams)
		assert self.scores != None, 'cyclone: first train the model before attempting to plot'
		if not os.path.exists(out_dir): os.makedirs(out_dir)
		file_name = file_name + method + '.pdf'
		if not plot_test: file_name = file_name + method + '_cv.pdf'
		
		if plot_test:
			labs = self.labels_tst
			if method=='RF':
				scores = self.scores_tst
			elif method=='LR':
				scores = self.scoresLR_tst
			elif method=='LRall':
				scores = self.scoresLRall_tst
			elif method=='GNB':
				scores = self.scoresGNB_tst
		else:	
			labs = self.labels
			if method == 'RF':
				scores = self.scores
			elif method == 'LR':
				scores = self.scoresLR
			elif method == 'LRall':
				scores = self.scoresLRall
			elif method == 'GNB':
				scores = self.scoresGNB
		cols = ['r', 'b', 'g', 'y', 'Crimson', 'DeepPink','LightSalmon','Lime', 'Olive']
		cols_d = {}
		for i,class_name in enumerate(class_labels):
			cols_d[class_name] = cols[i]
		labs = labs.astype('int')
		lab_col = list()
		fig = PL.figure(figsize=(6,6))
		ax = fig.add_subplot(111)
		ax.set_position([0.1,0.1,0.7,0.7])
		hList = list()
		for iplot in range(len(labs)):
			if class_labels[labs[iplot]-1] in cols_d.keys():
				hList.append(PL.plot(scores[iplot,xaxis],scores[iplot,yaxis],'.',markersize=15,
									c=cols_d[class_labels[labs[iplot]-1]], alpha=0.75))
			else:
				hList.append(PL.plot(scores[iplot,xaxis],
									scores[iplot,yaxis],'.',markersize=15,c='#8c510a', alpha=0.75))
		PL.xlabel(xlab)
		PL.ylabel(ylab)
		x_max = scores[:,xaxis].max() #+ 0.05
		PL.xlim(xmax = x_max+0.05)
		y_max = scores[:,yaxis].max() #+ 0.05
		PL.ylim(ymax = y_max+0.05)

		if decision_lines:
			x_min = 0.0
			y_min = 0.0
			h = 0.001
			xx, yy = SP.meshgrid(SP.arange(x_min, x_max, h), SP.arange(y_min, y_max, h))
			zz = 1 - (xx + yy)
			Z = SP.argmax(SP.dstack((xx,yy,zz)), 2)
			PL.contour(xx, yy, Z, levels = [0,1])

		legH = list()
		u_classes = SP.unique(labs)
		for ileg in u_classes:
			legH.append(hList[SP.where(labs==ileg)[0][0]][0])
			
		lh = PL.legend(legH,class_labels,loc='upper center',
					bbox_to_anchor=(0.5, 1.15),ncol=3, numpoints=1,scatterpoints=1)
		lh.set_frame_on(False)
		ax.spines["right"].set_visible(False)
		ax.spines["top"].set_visible(False)
		ax.get_xaxis().tick_bottom()
		PL.savefig(out_dir + '/' + file_name, bbox_extra_artists=[lh])#,bbox_inches='tight')

	def plotPerformance(self,
					perfType='ROC',
					plot_test=False, 
					out_dir = './cache',
					class_labels = ['G1 phase','S phase','G2M phase'], 
					method='RF'):
		plparams = {'backend': 'pdf',
		  'axes.labelsize': 14,
		  'text.fontsize': 14,
		  'legend.fontsize': 13,
		  'xtick.labelsize': 14,
		  'ytick.labelsize': 14,
		  'text.usetex': False}
		PL.rcParams.update(plparams)
		assert(perfType in ['ROC', 'PR'])
		if plot_test:
			labs = self.labels_tst
			if method == 'RF':			
				scores = self.scores_tst
			elif method == 'LR':
				scores = self.scoresLR_tst
			elif method == 'LRall':
				scores = self.scoresLRall_tst
			elif method == 'GNB':
				scores = self.scoresGNB_tst
		else:
			labs = self.labels
			if method == 'RF':
				scores = self.scores
			elif method == 'LR':
				scores = self.scoresLR
			elif method == 'LRall':
				scores = self.scoresLRall
			elif method == 'GNB':
				scores = self.scoresGNB
		PL.figure()
		col = ['r', 'b', 'g', 'y', 'Crimson', 'DeepPink','LightSalmon','Lime', 'Olive']
		aucList = SP.zeros((scores.shape[1],))
		for ind in range(scores.shape[1]):
			labels_i = labs.copy()
			scores_i = scores[:,ind]
			labels_i[labs==ind + 1] = 1
			labels_i[labs!=ind + 1] = 0
			if perfType == 'ROC':
				fpr, tpr, thresholds = metrics.roc_curve(labels_i, scores_i)
				aucList[ind] = metrics.auc(fpr, tpr)
			elif perfType == 'PR':
				fpr, tpr, thresholds = metrics.precision_recall_curve(labels_i, scores_i)
			PL.plot(fpr, tpr, '-', c=col[ind])
		if perfType == 'ROC':
			PL.title("ROC curve ccClassify")
			PL.xlabel('FPR')
			PL.ylabel('TPR')
			leg_str = list()
			for i in range(len(class_labels)):
				leg_str.append(class_labels[i]+', AUC = '+str(SP.round_(aucList[i],3)))
			PL.legend(leg_str, loc='lower-right')
			ax = plt.gca()
			ax.spines["right"].set_visible(False)
			ax.spines["top"].set_visible(False)
			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()
			PL.savefig(out_dir + '/ROC_test' + str(plot_test) + '_' + method + '.pdf', bbox_inches='tight')
		else:
			PL.title("PR curve ccClassify")
			PL.xlabel('Precision')
			PL.ylabel('Recall')
			leg_str = list()
			for i in range(len(class_labels)):
				leg_str.append(class_labels[i])#+', AUC = '+str(SP.round_(aucList[i],3)))
			PL.legend(leg_str, loc='lower-right')
			ax = plt.gca()
			ax.spines["right"].set_visible(False)
			ax.spines["top"].set_visible(False)
			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()
			PL.savefig(out_dir + '/ROC_test' + str(plot_test) + '_' + method + '.pdf', bbox_inches='tight')

	def plotF1(self,
			plot_test=False, 
			out_dir='./cache',
			class_labels = ['G1','S','G2M']):
		plparams = {'backend': 'pdf',
		  'axes.labelsize': 14,
		  'text.fontsize': 14,
		  'legend.fontsize': 13,
		  'xtick.labelsize': 14,
		  'ytick.labelsize': 14,
		  'text.usetex': False}
		PL.rcParams.update(plparams)
		fig = PL.figure(figsize=(6,6))
		ax = fig.add_subplot(111)
		ax.set_position([0.1,0.1,0.7,0.7])
		hList = list()
		cols = ['r', 'b', 'g', 'y', 'Crimson', 'DeepPink','LightSalmon','Lime', 'Olive']
		cols_d = {}
		for i,class_name in enumerate(class_labels):
			cols_d[class_name] = cols[i]
		m_i = 0
		all_meth = ['GNB','RF', 'LR', 'LRall']
		for method in all_meth: 
			if plot_test:
				labs = self.labels_tst
				if method == 'RF':
					scores = self.scores_tst
				elif method == 'LR':
					scores = self.scoresLR_tst
				elif method == 'LRall':
					scores = self.scoresLRall_tst
				elif method == 'GNB':
					scores = self.scoresGNB_tst
			else:
				labs = self.labels
				if method == 'RF':
					scores = self.scores
				elif method == 'LR':
					scores = self.scoresLR
				elif method == 'LRall':
					scores = self.scoresLRall
				elif method == 'GNB':
					scores = self.scoresGNB
			pred = SP.argmax(scores,axis=1) + 1
			f1_scores = metrics.f1_score(labs, pred, average=None)
			f1_score_av = metrics.f1_score(labs, pred, average="macro")
			for iplot in range(len(f1_scores)):
				if class_labels[iplot] in cols_d.keys():
					hList.append(PL.plot(m_i,f1_scores[iplot],'^',
										markersize=15,c=cols_d[class_labels[iplot]], alpha=0.65))		
			PL.plot([m_i-0.25, m_i+0.25], [f1_score_av,f1_score_av], linewidth=2, color='r', hold=True)
			m_i += 1.0
		PL.xticks(range(len(all_meth)), all_meth, rotation=46)
		PL.ylabel('F1 score')
		PL.ylim(ymin = -0.05, ymax = 1.05)
		ax = plt.gca()
		ax.spines["right"].set_visible(False)
		ax.spines["top"].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()
		PL.savefig(out_dir + '/F1_test' + str(plot_test) + '_' + method + '.pdf', bbox_inches='tight')
