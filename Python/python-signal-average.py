import numpy
import matplotlib.pyplot as plt

class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
        self.signalHeader = None
	
            
    def process(self):
        global numpyBuffer1
        global numpyBuffer2
        global numpyBuffer3
        global numpyBuffer1_list
        #numpyBuffer1 = []
        #numpyBuffer2 = []
        #numpyBuffer3 = []
        
        for chunkIndex in range( len(self.input[0]) ):
            if(type(self.input[0][chunkIndex]) == OVSignalHeader):
                self.signalHeader = self.input[0].pop()
				
                outputHeader = OVSignalHeader(
                self.signalHeader.startTime, 
                self.signalHeader.endTime,
                [1, self.signalHeader.dimensionSizes[1]], 
                ['Mean']+self.signalHeader.dimensionSizes[1]*[''],
                self.signalHeader.samplingRate)
			
                self.output[0].append(outputHeader)
				
                numpyBuffer1 = numpy.zeros(tuple(self.signalHeader.dimensionSizes))
                numpyBuffer2 = numpy.zeros(tuple(self.signalHeader.dimensionSizes))
                numpyBuffer3 = numpy.zeros(tuple(self.signalHeader.dimensionSizes))
                numpyBuffer1_list = []
                
            elif(type(self.input[0][chunkIndex]) == OVSignalBuffer):
                chunk = self.input[0].pop()
                numpyBuffer = numpy.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
                #numpyBuffer = numpyBuffer.mean(axis=0)
                
                if((chunk.startTime > 0) &(chunk.endTime <= 20)):
                    numpyBuffer1 = numpy.hstack((numpyBuffer1, numpyBuffer))
                    print('1')
                if((chunk.startTime > 20) &(chunk.endTime <= 40)):
                    numpyBuffer2 = numpy.hstack((numpyBuffer2, numpyBuffer))
                    print('2')
                if((chunk.startTime > 40) &(chunk.endTime <= 55)):
                    numpyBuffer3 = numpy.hstack((numpyBuffer3, numpyBuffer))
                    print('3')
                
                if((chunk.startTime > 56)):
                    print('baseline: ', numpy.mean(numpyBuffer1))
                    print('relaxed: ',numpy.mean(numpyBuffer2))
                    print('attentive: ', numpy.mean(numpyBuffer3))
                    numpyBuffer2 = numpyBuffer2 + 0.5
                    numpyBuffer3 = numpyBuffer3 + 1
                chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, numpyBuffer.tolist())
                self.output[0].append(chunk)
				
            elif(type(self.input[0][chunkIndex]) == OVSignalEnd):
                self.output[0].append(self.input[0].pop())	 		
                
    def uninitialize(self):
        
        plt.hist([numpyBuffer1[0,:], numpyBuffer2[0,:], numpyBuffer3[0,:]], density=True)
        #numpy.save('C:\\Users\\Chris\\Documents\\3_Uni\\1_MSc Neuroengineering\\Semester 4\\NISE\\NISE\\Python\\buffer.npy', numpyBuffer1)
        plt.show()
        

box = MyOVBox()
