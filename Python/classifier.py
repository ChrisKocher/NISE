import numpy as np

class classifier(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		
	# the initialize methodeads settings and outputs the first header
	def initialize(self):
		print 'initializing classifier'
		self.stimLabel = self.setting['ClassLabel']
		self.stimCode = OpenViBE_stimulation[self.stimLabel]
		self.winCount = 0
		self.nRejection = int(self.setting['nRejection'])
		#print self.stimLabel
		self.output[0].append(OVStimulationHeader(0., 0.))
		
	def process(self):
		# TO DO: first pop out the data then check the state_dict
		
		for chunkIndex in range( len(self.input[0]) ):
			if(type(self.input[0][chunkIndex]) == OVSignalHeader):
				self.signalHeader = self.input[0].pop()
			elif(type(self.input[0][chunkIndex]) == OVSignalBuffer):
				chunk = self.input[0].pop()
				#print(type(chunk))
				Buffer = None
				Buffer = np.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
				print('Buffer: ',np.shape(Buffer))

                global state_dict
				
                if state_dict['State']=='Trial':
                    global cov_mat
                    global groundtruth
                    global online_pred
                    global model
                    global online_pred_arr
                    # classification - saving cov matrices 
                    #(Note: reject windows at the beginning of the trials which are not related to the task
                    if (self.winCount < self.nRejection):
                        self.winCount +=1
						#print 'rejected!'
                    else:
                        Buffer =np.transpose(np.expand_dims(Buffer, axis=2),(2, 0,1))
						#print 'Shape of buffer', np.shape(Buffer)
                        curr_cov =Covariances('lwf').transform(Buffer[:,:,:])# 64:(511-64)Throwing away beginning and end of the window to remove edge artifact due to filtering
						#print 'curr_cov matrix shape:', np.shape(curr_cov)
						
                        cov_mat = np.concatenate((cov_mat, curr_cov),axis=0)
                        groundtruth = np.concatenate((groundtruth, [state_dict['Label']]))
                        if state_dict['Phase']== 'Calib':
                            self.stimCode = state_dict['Label']
                            stimSet = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime()+1./self.getClock())
                            stimSet.append(OVStimulation(self.stimCode, self.getCurrentTime(), 0.))
                            self.output[0].append(stimSet)
                            print 'Predefined output generated:',state_dict['Label']
                        else:	
                            pred = model.predict(curr_cov)
                            #print 'pred shape:', pred.shap
                            online_pred = np.concatenate((online_pred,pred))
                            self.stimCode = pred
                            stimSet = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime()+1./self.getClock())
                            stimSet.append(OVStimulation(self.stimCode, self.getCurrentTime(), 0.))
                            self.output[0].append(stimSet)
                            print 'Output was generated, Prediction:', self.stimCode, '  Groundtruth:', state_dict['Label']
                else:
                    self.winCount = 0
                    return

	def uninitialize(self):
		x=1				

box = classifier()

#state_dict = {'Phase':'Calib'|'Test'|'ITP', 'State':'Rest', 'Label':1,'CurrentTrial': 1}