import threading
import time
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool

class trainer(OVBox):
    def __init__(self):
        OVBox.__init__(self)

		
	# the initialize method reads settings and outputs the first header
    def initialize(self): 
        print 'initializing trainer'
        self.isTrained = False
        #self.thrd = threading.Thread(target = self.trainer_thread)
    def process(self):
        global state_dict
		#print self.isTrained
        if ((state_dict['State']=='ITP') or (state_dict['State']=='Rest')) and (not self.isTrained) and (state_dict['Phase']=='Test'):
			# Trainin
            global model
            global cov_mat
            global groundtruth
            global online_pred
            global online_pred_arr
            print 'Training...'
			#thrd = threading.Thread(target = self.trainer_thread)
			#thrd.start()
			#print 'Training finished at:', self.getCurrentTime()
			#pool = Pool(processes=1)
			#async_result = pool.apply_async(self.trainer_thread, args = (model, cov_mat, groundtruth))
			#model = async_result.get()
			#MyThread(self.on_thread_finished, (model, cov_mat, groundtruth)).start(
            t = time.time()
            model.fit(cov_mat,groundtruth)
            elapsed = time.time() - t
            print 'Training time in main thread:', elapsed, 's'
            self.isTrained = True
            if (online_pred.shape[0]!= 0):
                print 'Online accuracy:', np.mean(online_pred==groundtruth[-online_pred.shape[0]:])
                online_pred_arr = np.concatenate(online_pred_arr,np.mean(online_pred==groundtruth[-online_pred.shape[0]:]))
        elif (state_dict['State']=='Trial'):
			self.isTrained = False

	# def trainer_thread(self,model,cov_mat,groundtruth):
	# 	print 'inside the thread'
	# 	t = time.time()
	# 	# global model
	# 	# global cov_mat
	# 	# global groundtruth
	# 	model.fit(cov_mat, groundtruth)
	# 	elapsed = time.time() - t
	# 	print 'Training time in thread:', elapsed, 's'
	# 	return (model, elapsed)

	# def on_thread_finished(self, data):
	# 	trained_model, elapsed = data
	# 	print 'on_thread_finished in: ',elapsed,'s'
	# 	global model
	# 	model = data

	def uninitialize(self):
		x=1	

# class MyThread(threading.Thread):
#     def __init__(self, callback, args):
#         threading.Thread.__init__(self)
#         self.callback = callback
#         self.model, self.cov_mat, self.groundtruth = args

#     def run(self):
#         t = time.time()
#         self.model.fit(self.cov_mat, self.groundtruth)
#         elapsed = time.time() - t
#         print 'Training time in thread:', elapsed, 's'
#         self.callback(model,elapsed)


box = trainer()



#state_dict = {'Phase':'Calib'|'Test'|'ITP', 'State':'Rest', 'Label':1,'CurrentTrial': 1}