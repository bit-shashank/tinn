from tinn.layer import Layer
import tinn.activations as activations
import numpy as np
import pickle


class NeuralNet:
    """
        Core class of the neural network
    """

    def __init__(self):
        self.layers=[]


    def add(self,layer):
        """
            Add a layer to the model
            Parameters:
                layer (Layer): An object of class Layer
        """
        self.layers.append(layer) 

    def feed_forward(self,inputs):
        """
            Feed forward the inputs through the neural network
            Parameters:
                inputs (array): An array of inputs
            Returns:
                out_layer (array): Array of outputs from neural network
        """
        inp=inputs
        for layer in self.layers:
            out_layer=layer.feed(inp)
            inp=out_layer
        return out_layer


    def backpropogate(self,inputs,t_outputs,learning_rate=0.01):
        """
            Backpropogate through neural network and adjust the weights using gradient descent
            Parameters:
                inputs (array)          : List of inputs
                t_outputs (array)       : List of correct outputs
                learning_rate (float)   : Learning rate , default is 0.01
            Returns:
                mse (float)             : Mean square error of the model
        """
        p_outputs=self.feed_forward(inputs)
        errors=t_outputs-p_outputs
        #Calulating Mean Square Error
        mse= (errors**2).sum()/len(errors)

        for i in reversed(range(len(self.layers))):
            curr_layer=self.layers[i]
            #Calculating gradient
            p_outputs=curr_layer.output 
            gradients=activations.apply(p_outputs,curr_layer.activation,deriv=True)
            gradients*=(errors * learning_rate)
            
            #Calculating deltas weights
            if i-1>=0:
                transMat=self.layers[i-1].output.T
            else:
                transMat=inputs.T
            deltaWeights=np.dot(gradients,transMat)

            #Adjusting weights and biases
            curr_layer.weights+=deltaWeights
            curr_layer.bias+=gradients

            #Updating errors for previous layers
            weightsT=curr_layer.weights.T 
            errors=np.dot(weightsT,errors) 
        
        return mse
            

    
    def train(self,inputData,outputData,learning_rate=0.01,epocs=100,suffle=True):
        """
            Trains the model on given training data
            Parameters:
                inputData (ndarray)     : Array of all input data
                outputData (ndarray)    : Array of all output data
                learning_rate(float)    : Learning rate of the model
                epocs (int)             : Number of iterations over training data 
                suffle (boolean)        : Set to false to prevent shuffling between epocs
        """
        len_inp=len(inputData)
        printProgressBar(0,len_inp,prefix='Epoc:', suffix='Error:',length=50)
        for epoc in range(epocs):
            if suffle:
                #Shuffle all data before each epoc
                inputData,outputData=self.unison_suffle(inputData,outputData)
            #Train network on all input data
            for i in range(len(inputData)):
                error=self.backpropogate(inputData[i],outputData[i],learning_rate)
                printProgressBar(i + 1, len_inp, prefix = 'Epoc:'+str(epoc), suffix = 'Error:'+str(error), length = 50)
        

    def validate(self,test_inputs,test_outputs):
        """
            Validates the trained model over given test_inputs and test_outputs
            Parameters:
                test_inputs (array): List of all testing inputs
                test_outputs(array): List of all testing outputs
            Returns :
                accuracy (float)   : Accuracy of the model, a float between [0,1] inclusive
        """
        total_correct=0
        for i in range(len(test_inputs)):
            p_outputs=self.feed_forward(test_inputs[i])
            a_outputs=test_outputs[i]

            if(self.argmax(p_outputs)==self.argmax(a_outputs)):
                total_correct+=1

        accuracy=total_correct/len(test_inputs)
        return accuracy

    def save(self,file):
        """
            Saves the model to a file
            Parameters:
                file (str): Name of the file
        """
        output_file=open(file,'wb')
        pickle.dump(self,output_file)
        output_file.close()

    @staticmethod
    def load(file):
        """
            Loads the model from the given file
            Parameters:
                file (str)      : Name of the file 
            Returns:
                nn (NeuralNet)  : Return the saved model
        """
        model_file=open(file,'rb')  
        nn= pickle.load(model_file)
        model_file.close()
        return nn
 
    def unison_suffle(self,a, b):
        """
            Utility method to randomly shuffle two arrays in unison
            Parameters:
                a (array): first array
                b (array): second array
        """
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]


    def predict(self,inputs):
        """
            Predict the outputs of the neural netwrok on given inputs
            Parameters:
                inputs (array) : List of inputs
            Returns:
                outputs (array): List of outputs predicted by model
        """
        outputs=self.feed_forward(inputs) 
        return outputs

    def predict_classes(self,inputs):
        outputs=self.predict(inputs) 
        out=outputs.reshape(len(outputs))
        classes=np.where(out==max(out))
        return classes

    def argmax(self,outputs):
        """
            Returns the elements with max values
            Parameters:
                outputs(list)
            Returns:
                classes(list): Classes with maximum values in outputs
        """
        out=outputs.reshape(len(outputs))
        classes=np.where(out==max(out))
        return classes


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
