import numpy as np
import scipy.io as sio
import time
import datetime

# pyRiemann
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import TSclassifier

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# This class is for managing the trial orders and calibration and classification phases.

class trial_manager(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		
		
	# the initialize method reads settings and outputs the first header
	def initialize(self):
		print 'Initializing trial_manager'
		global state_dict
		self.ntTrial = int(self.setting['Number of test trials'])
		self.ncTrial = int(self.setting['Number of calibration trials'])
		self.prepTime = float(self.setting['Preparation time'])
		self._rand_trial_gen() #generates the order of trials
		self.trialLength = int(self.setting['Trial Length'])
		self.itt = int(self.setting['Inter trial time'])
		self.nextEvent = self.prepTime
		self.TrialOrder = self.calibTrialOrder
		self.nChan = self.setting['Number of Channels']
		state_dict['nChan'] = self.nChan

		self.stimLabel = self.setting['stateLabel']
		self.stimCode = OpenViBE_stimulation[self.stimLabel]
		self.output[0].append(OVStimulationHeader(0., 0.))
		
	def process(self):
		global state_dict
		if (self.getCurrentTime()< self.nextEvent):
			#print 'Wait!'
			return
		self._update_next_event()
		print 'time:', self.getCurrentTime(), 'State', state_dict
		global cov_mat
		print 'cov_mat shape:', cov_mat.shape
		if (state_dict['Label']==1) and (state_dict['State']=='Trial'):
			self.stimCode = 1
		elif (state_dict['Label']==-1) and (state_dict['State']=='Trial'):
			self.stimCode = 2
		else:
			self.stimCode = 3

		stimSet = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime()+1./self.getClock())
		stimSet.append(OVStimulation(self.stimCode, self.getCurrentTime(), 0.))
		self.output[0].append(stimSet)	
		

	def uninitialize(self):
		print 'groundtruth shape:',groundtruth.shape
		print 'online_pred shape:',online_pred.shape
		print 'Saving groundtruth and predictions...'
		now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")

		np.save('C:\\Users\\Chris\\Documents\\3_Uni\\1_MSc Neuroengineering\\Semester 4\\NISE\\OpenVibe scenarios\\Moh\\functions\\groundtruth_'+now+'.npy',groundtruth)
		np.save('C:\\Users\\Chris\\Documents\\3_Uni\\1_MSc Neuroengineering\\Semester 4\\NISE\\OpenVibe scenarios\\Moh\\functions\\prediction_'+now+'.npy',online_pred)
        np.save('C:\\Users\\Chris\\Documents\\3_Uni\\1_MSc Neuroengineering\\Semester 4\\NISE\\OpenVibe scenarios\\Moh\\functions\\onlineAcc_.npy',online_pred_arr)
        				

	def _rand_trial_gen(self):
		self.testTrialOrder = np.ones(self.ntTrial, dtype=np.int32)
		self.calibTrialOrder = np.ones(self.ncTrial, dtype=np.int32)
		self.testTrialOrder[0:int(self.ntTrial/2)] = -1;
		self.calibTrialOrder[0:int(self.ncTrial/2)] = -1;
		
		np.random.shuffle(self.calibTrialOrder)
		np.random.shuffle(self.calibTrialOrder)
		np.random.shuffle(self.testTrialOrder)
		np.random.shuffle(self.testTrialOrder)
		
		print 'Number of calibration trials:',self.ncTrial, ', Number of test trials:', self.ntTrial
		#print 'Calib trial Order:', self.calibTrialOrder
		#print 'Test trial Order:', self.testTrialOrder
		print 'Number of samples in positive class (calib):', np.sum(self.calibTrialOrder==1)
		print 'Number of samples in positive class:(test)', np.sum(self.testTrialOrder==1)
	
	def _update_next_event(self):
		global state_dict
		if (state_dict['State'] == 'ITP') & (state_dict['Phase'] == 'Calib')&((state_dict['CurrentTrial']%self.ncTrial)==0.0):
			state_dict['Phase'] = 'Test'
			self.TrialOrder  = self.testTrialOrder
			state_dict['CurrentTrial']=0
			state_dict['State'] = 'Rest'
			print 'Resting period...'
			self.nextEvent = np.ceil(self.getCurrentTime()) + self.prepTime
		elif (state_dict['State'] == 'ITP') & ((state_dict['CurrentTrial']%20)==0.0):
			state_dict['State'] = 'Rest'
			self.nextEvent = np.ceil(self.getCurrentTime()) + self.prepTime
		elif (state_dict['State'] == 'Trial'):
			state_dict['State'] = 'ITP'
			self.nextEvent = np.ceil(self.getCurrentTime()) + self.itt
		else:
			state_dict['State'] = 'Trial'
			self.nextEvent = np.ceil(self.getCurrentTime()) + self.trialLength
			state_dict['Label'] = self.TrialOrder[state_dict['CurrentTrial']]
			state_dict['CurrentTrial'] = state_dict['CurrentTrial'] + 1
		
		return
	

state_dict = {'Phase':'Calib', 'State':'Rest', 'Label':1,'CurrentTrial': 0, 'nChan':13, 'isTrained':False}	
box = trial_manager()

cov_mat = np.zeros([0,state_dict['nChan'], state_dict['nChan']],dtype=np.float16)
groundtruth = np.zeros(0,dtype=np.int16)
online_pred = np.zeros(0,dtype=np.int16)
online_pred_arr = np.zeros(0,dtype=np.int16)


# make classification pipeline
ts = TangentSpace()
classifier = SVC(kernel='linear')
#classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#		  intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#		  penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#		  verbose=0, warm_start=False)
model = make_pipeline(ts,classifier)

































































