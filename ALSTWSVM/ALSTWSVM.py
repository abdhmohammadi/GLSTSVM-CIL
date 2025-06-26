
import math
from math import pi,acos 
import sys
from time import perf_counter
import numpy as np
sys.path.append('/base-tools')
from BaseTSVMClassifier import BaseTSVMClassifier

class ALSTWSVM(BaseTSVMClassifier):
    """
    Least Squares Angle based Twin SVM
    """
    def __init__(self,c1=0.1,c2 = 0.5, c3 = 0.1,c5=0.1, kernel_name = 'rbf',mu = 1.0,compute_errors_and_angles=True) -> None:
        """
        c1 : 1e-05 to 1
        c2 : (0,1)
        c3 in (0,1]
        c4 : 1e-05 to 1
        """    
        super().__init__(kernel_name=kernel_name,mu=mu)
        
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = 1- self.c2
        self.c5 = c5
        self.wc_time = 0.0
        self.compute_errors_and_angles = compute_errors_and_angles

    def fit(self,X_train,y_train):

        A, m1, B, m2 , self.classes,self.n_features = self.split(X = X_train, y=y_train)
        
        start = perf_counter()

        if self.kernel_name == None or self.kernel_name == '':
            H = np.hstack((A,np.ones((m1,1))))
            G = np.hstack((B,np.ones((m2,1))))
        else:
            self.C = np.vstack((A,B))
            H = np.hstack((self.Kernel(A,self.C) , np.ones((m1,1))))
            
            G = np.hstack((self.Kernel(B,self.C) , np.ones((m2,1))))
        
        GTG = G.T @ G
        self.u1 = - self.c3 * np.linalg.inv(H.T @ H + self.c3 * GTG + self.c1 * np.identity(GTG.shape[1])) @ G.T @ np.ones((m2,1))
        self.u2 = - 0.5 * (self.c4/self.c2) * np.linalg.inv(GTG + 0.5*(self.c5/self.c2) * np.identity(GTG.shape[1])) @ self.u1

        self.training_time = perf_counter() - start
        
        if self.compute_errors_and_angles == True:
           if self.kernel_name == '' or self.kernel_name == None:
             
              v1 = np.asarray(self.u1[:-1])
              v2 = np.asarray(self.u2[:-1])
              length = np.linalg.norm(v1) * np.linalg.norm(v2)

              self.angle = acos((v1.T @ v2)/length)*180/pi 

              ssd1 = H @ self.u1
              ssd1 = ssd1.T @ ssd1
              ssd2 = G @ self.u2 + np.ones((m2,1))
              ssd2 = ssd2.T @ ssd2
              self.SSD = math.sqrt(ssd1 +ssd2)
        else: 
            self.angle   = 'The angles are avilable in linear mode.'
        

        
        
    def predict(self, X_test):

        predicted = []
        
        for x in X_test:
            x = np.asarray(x)

            if self.kernel_name == None or  self.kernel_name == '':
                x = np.hstack((x,[1]))
            else:
               
                x = self.Kernel(x.reshape(1,-1),self.C)[0]
                x = np.vstack((x.reshape(len(x),1),[1]))

            f1 = (x.T @ self.u1)/np.linalg.norm(self.u1)**2
            f2 = (x.T @ self.u2)/np.linalg.norm(self.u2)**2
            
            predicted.append(self.classes[0] if abs(f1) < abs(f2) else self.classes[1])

        return np.asarray(predicted)
    
    def classification_report(self):
        print()
        print('------------------------ LS-ATWSVM REPORT ---------------------')
        print('Hyperplane parameters(c1,c2,c3,c5):',str(self.c1) + ', ' + str(self.c2) + ', ' + str(self.c3) +', ' + str(self.c5))
        print('           Class Imbalace Rate(IR):',self.IR)
        print('                            Kernel:',self.kernel_name)
        print('              Kernel parameter(mu):',self.mu)
        print('                        Train size:',str(self.C.shape[0])+' \u00d7 '+str(self.C.shape[1]))
        print('              Data reading time(s):',self.reading_time)
        print('                  Training time(s):',self.training_time)
        print('                               SSE:',self.SSE)
        print('---------------------------------------------------------------')
        
       
