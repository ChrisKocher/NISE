import numpy
import matplotlib.pyplot as plt

class MyOVBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
        self.signalHeader = None
	
    def process(self):
        for chunkIndex in range( len(self.input[0]) ):
            if(type(self.input[0][chunkIndex]) == OVSignalHeader):
                self.signalHeader = self.input[0].pop()
				
                outputHeader = OVSignalHeader(
                self.signalHeader.startTime, 
                self.signalHeader.endTime,
                [1, self.signalHeader.dimensionSizes[1]], 
                ['Mean']+self.signalHeader.dimensionSizes[1]*[''],
                self.signalHeader.samplingRate)
                print('1: ',self.signalHeader.startTime)
				
                self.output[0].append(outputHeader)
				
            elif(type(self.input[0][chunkIndex]) == OVSignalBuffer):
                chunk = self.input[0].pop()
                print('2: ',chunk.startTime)
                numpyBuffer = numpy.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
                numpyBuffer = numpyBuffer.mean(axis=0)
                print(numpyBuffer.mean())

                chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, numpyBuffer.tolist())
                self.output[0].append(chunk)
				
            elif(type(self.input[0][chunkIndex]) == OVSignalEnd):
                self.output[0].append(self.input[0].pop())	 		
                
    def uninitialize(self):
        plt.plot([1,2,3],[1,2,3])
        plt.show()

box = MyOVBox()
