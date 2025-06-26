
from math import acos, pi
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel , linear_kernel,sigmoid_kernel, laplacian_kernel, chi2_kernel

class BaseTSVMClassifier(BaseEstimator):
    
    def __init__(self,kernel_name='rbf',mu=1.0) -> None:
        self.kernel_name = kernel_name
        self.mu = mu
        self.classes = []
        self.reading_time = 0.0
        self.training_time = 0.0
        self.wc_time = 'Not available'
    
    
    def compute_angles(self):
        """
            Returns the angle of the Norm vectors in digree
        """
        if self.kernel_name == '' or self.kernel_name == None:
           # calculate angle
           length = np.linalg.norm(self.u1)*np.linalg.norm(self.u2)
           self.norm_angle = 180*acos((self.u1.T @ self.u2)/length)/pi 
           self.planes_angle = 380-180-self.norm_angle

        else: 
            self.norm_angle   = 'The angles are avilable in linear mode.'
            self.planes_angle = 'The angles are avilable in linear mode.'
        
        return self.norm_angle, self.planes_angle


    @property
    def positive_label(self): return self.classes[0] if np.any(self.classes) else None
    
    def Kernel(self,A,B):
        
        if self.kernel_name == 'linear': return linear_kernel(A,B)
        # K(x, y) = exp(-gamma ||x-y||^2)
        elif self.kernel_name == 'rbf': return rbf_kernel(A,B,gamma=self.mu)
        # K(X, Y) = tanh(gamma <X, Y> + coef0)
        elif self.kernel_name == 'sigmoid': return sigmoid_kernel(A,B,gamma=self.mu)
        # K(x, y) = exp(-gamma ||x-y||_1)
        elif self.kernel_name == 'lap': return laplacian_kernel(A,B,self.mu)
        # k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])        
        elif self.kernel_name == 'chi2': return chi2_kernel(A,B,self.mu)
        else: return A @ B.T
    
    # This function splits the data to the positive and negetive class
    # Inputs: X as data samples, y the labels of the data
    # Returns:
    # Matrix A, the samples of minority class or positive samples
    # Matrix B, the samples of majority class or negetive samples
    # IR, Imbalance rate(is number of samples in B devided by number of samples in A)
    # classes as 2 elements array, first element as label of 'A', second element as lebel of 'B'
    def extract_data(X,y:np.array):

        classes = np.unique(y)
    
        m1 = len(y[y == classes[0]])
        m2 = len(y[y == classes[1]])

        # Changeing m1 and m2 if m1 > m2, (We assume that the majority class is the negative class)
        if m1 > m2 :
            m2 = m1 + m2
            m1 = m2 - m1
            m2 = m2 - m1
            tmp = classes[0]
            classes[0] = classes[1]
            classes[1] = tmp
    
        A = np.asarray(X[y == classes[0]]) # minority class - positive class
        
        B = np.asarray(X[y == classes[1]])#,dtype=np.float64) # majority class - negative class
        # returns:
        #    A:Positive data, 
        #    B:negative data, 
        #    m2/m1:Imbalance rate,
        #    classes:labels(binary class)
        return A, B, m2/m1 , classes
    
    def split(self,X,y:np.array,dtype:np.dtype='float64'):
        
        classes = np.asarray(np.unique(y))
        
        n_features = int(X.shape[1])

        m1 = len(y[y == classes[0]])
        m2 = len(y[y == classes[1]])
        # Changeing m1 and m2 if m1 > m2, (We assume that the majority class is the negative class)
        if m1 > m2 :
            m2 = m1 + m2
            m1 = m2 - m1
            m2 = m2 - m1
            tmp = classes[0]
            classes[0] = classes[1]
            classes[1] = tmp

        self.IR = np.round(m2/m1,3)
        
        A = np.asarray(X[y == classes[0]],dtype= dtype) # Positive class
        
        B = np.asarray(X[y == classes[1]],dtype= dtype) # Negative class
        # returns:
        #    A:Positive data, 
        #    B:negative data, 
        #    m1,m2: number of samples in each class,
        #    classes:labels(binary class),
        #    n_features: number of features
        return A.reshape((m1,n_features)), m1, B.reshape((m2,n_features)), m2, classes, n_features
    
    def compute_scores(self,y_true,y_pred):
      
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        L = len(y_true)
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(L):

            if y_true[i] == y_pred[i] == self.classes[0] : TP +=1
            
            if y_true[i] == self.classes[1] and y_pred[i] == self.classes[0] : FP +=1
            
            if y_true[i] == y_pred[i] == self.classes[1] : TN +=1
            
            if y_true[i] == self.classes[0] and y_pred[i] == self.classes[1] : FN +=1
        # Accuracy
        A = (TP + TN)/(TP + FP +TN + FN)
        # Precision
        P = TP/(TP + FP) if TP + FP >0 else 0.0
        # Recall
        R = TP/(TP + FN)  if TP + FN >0 else 0.0
        # F1-Score
        F1= 2*(P*R)/(P + R)  if R + P >0 else 0.0
        # g-mean
        G = np.sqrt(P*R)

        return A, P, R, F1, G
