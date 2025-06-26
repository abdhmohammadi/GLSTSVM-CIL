
from math import acos,pi,exp
#import sys
import sys
from os.path import isfile
from time import perf_counter
import numpy as np
import pandas as pd
sys.path.append('/base-tools')
from BaseTSVMClassifier import BaseTSVMClassifier

class LSFLSTSVMCIL(BaseTSVMClassifier):
  
  def __init__(self,c0 = 1,c1 =1,c2=1,c3=1,c4=1,mu=1,kernel_name='rbf',epsilon=0.00001,iteration=100,label_name = None,compute_errors_and_angles=True) -> None:
    super().__init__(kernel_name= kernel_name, mu=mu)
    self.c0 = c0
    self.c1 = c1
    self.c2 = c2
    self.c3 = c3
    self.c4 = c4
    self.epsilon = epsilon
    self.iteration = iteration
    self.label_name = label_name
    self.compute_errors_and_angles = compute_errors_and_angles

  def predict(self,X_test):
  
    predicted = []

    X = np.asarray(X_test)
  
    l = len(self.z1)

    alpha = self.z1[0:self.A.shape[0]]
    beta  = self.z1[self.A.shape[0]:l]

    e1 = np.ones(self.B.shape[0])
    e2 = np.ones(self.A.shape[0])

    b1 = (e2.dot(alpha) + e1.dot(beta))/self.c3

    lambda_= self.z2[0:self.B.shape[0]]
    gamma_ = self.z2[self.B.shape[0]:l]

    b2 = -(e1.dot(lambda_) + e2.dot(gamma_))/self.c4
    
    for x in X: 
      x = np.asarray(x)

      KxA = self.Kernel(np.asarray(x).reshape(1,-1),self.A) 
      KxB = self.Kernel(np.asarray(x).reshape(1,-1),self.B)
    
      f1 = (KxA.dot(alpha) + KxB.dot(beta))/self.c3 + b1 

      f2 = -(KxB.dot(lambda_) + KxA.dot(gamma_))/self.c4 + b2 

      for i in range(len(f1)):
        if min(abs(f1[i]),abs(f2[i])) == abs(f1[i]): predicted.append(self.classes[0])
        else                                     : predicted.append(self.classes[1])
    
    return predicted

    def fit(self,X_train,y_train):
    
    self.A, m1, self.B, m2, self.classes,self.n_features = self.split(X_train,y_train)
    
    start = perf_counter()
    # Hyperplane 1:
    #Center of self.A
    s = sum(self.A,0)/self.A.shape[0]    
    # Center of self.B
    h = sum(self.B,0)/self.B.shape[0]
    S2 = self.fuzzy_weights(self.B,self.IR,self.c0,s,h)
    S1 = np.identity(m1)

    self.wc_time = perf_counter() - start

    start = perf_counter()
    
    KA  = self.Kernel(self.A, self.A)
    KB  = self.Kernel(self.B, self.B)
    KAB = self.Kernel(self.A, self.B)
    v = np.hstack((np.zeros(m1),np.ones(m2)))
    # Solving methods
    self.z1 = self.smo(np.hstack((np.vstack((KA + self.c3*np.identity(m1), KAB.T)),
                             np.vstack((KAB ,KB + (self.c3/self.c1)*S2)))) + 
                             np.ones((KA.shape[0] + KB.shape[0],KA.shape[0] + KB.shape[0])), 
                             np.hstack((np.zeros(KA.shape[0]),np.ones(KB.shape[0]))), self.c3, self.epsilon,self.iteration)
    self.alpha = self.z1
    # Hyperplane 2:       
    self.z2 = self.smo(np.hstack((np.vstack((KB + self.c4*np.identity(m2),KAB)),
                              np.vstack((KAB.T ,KA + (self.c4/self.c2)*S1))))+ np.ones((m1+m2,m1+m2)),
                              np.hstack((np.zeros(m2),np.ones(m1))), self.c4, self.epsilon,self.iteration)
    self.beta = self.z2    
    self.training_time = perf_counter() - start
    
    if self.kernel_name == None or self.kernel_name =='':
      self.w1 = (np.hstack((self.A.T, self.B.T)) @ self.z1/self.c3).reshape((self.n_features,1))
      self.b1 = sum(self.z1)/self.c3
      
      self.u1 = np.vstack((self.w1,[self.b1]))
                          
      self.w2 = (np.hstack((self.B.T, self.A.T)) @ self.z1/self.c4).reshape((self.n_features,1))
      self.b2 = sum(self.z2)/self.c4
      
      self.u2 = np.vstack((self.w2,[self.b2]))
      
      # calculate angle
      if self.compute_errors_and_angles == True:
              v1 = self.u1[:-1]
              v2 = self.u2[:-1]
              length = np.linalg.norm(v1) * np.linalg.norm(v2)
              
              self.angle = acos((v1.T @ v2)/length)*180/pi 
            
              eta1 = - S1 @ self.alpha[:m1]
              ssd1 = eta1 @ eta1
              eta2 = - S2 @ self.beta[:m2]
              ssd2 = eta2 @ eta2
              self.SSD = (ssd1+ssd2)**0.5


  def smo(self,Q,v,c = 1.0,epsilon = 0.0001,iteration = 100):
    no_row = Q.shape[0]

    z = np.zeros(no_row)

    diag = 2*Q.diagonal()
    # F is a vector for differentials of f(z) respect to all zi
    # Fi is differentials f(z) respect to zi
    F = np.array(-Q.dot(z)- c*v)

    norm = np.linalg.norm(F)
    iter = 0

    while norm > epsilon and iter < iteration:

        F2 = np.square(F)

        i = np.argmax(F2/diag)

        # step size
        t = F[i]/Q[i,i]
        # Updateing z
        z[i]= z[i] + t
        #recomputing F, F2 and norm
        F = np.array(-Q.dot(z) - c*v)

        norm = np.linalg.norm(F)

        iter = iter + 1
    return z

  def fuzzy_weights(self,M,IR,c0,center_a,center_b):
  
    S = np.zeros((M.shape[0],M.shape[0]))
    # Euclidean distance between tow centroid
    d = np.linalg.norm(center_a - center_b)  

    distance = []
    # Compute R2:
    # maximom distance among the negative samples from its centeriod 
    for item in self.B: distance.append(np.linalg.norm(item - center_b))
    R2 = max(distance)
    
    ir_ = 1/(IR + 1)
    i = 0
    for item in M:
      # distance of sample i from center of positive class
      d1  = np.linalg.norm( item - center_a)
      # distance of sample i from center of negative class
      d2  = np.linalg.norm( item - center_b) 
      # main formula
      w=(ir_)+(IR*ir_)*(exp(c0*((d1-d2)/d-d2/R2))-exp(-2*c0))/(exp(c0)-exp(-2*c0))

      S[i,i] = 1/(w*w)
    
      i = i + 1
    
    return S

# END OF fuzzy_weights function
  def classification_report(self):
        print()
        print('--------------------- ALSGWLSTSVM-CIL REPORT ------------------')
        print('Hyperplane parameters(c0,c1,c2,c3,c4):',str(self.c1) + ', ' + str(self.c2) + ', ' + str(self.c3) +', ' + str(self.c4))
        print('       Weight parameters(c0):',str(self.c0))
        print('           Class Imbalace Rate(IR):',self.IR)
        print('                            Kernel:',self.kernel_name)
        print('              Kernel parameter(mu):',self.mu)
        print('                        Train size:',str(self.A.shape[0] + self.B.shape[0])+' \u00d7 '+str(self.A.shape[1]))
        print('                  Positive samples:',self.A.shape[0])
        print('                  Negative samples:', self.B.shape[0])
        print('          Weight computing time(s):',self.wc_time)
        print('                  Training time(s):',self.training_time)
        print('                               SSE:',self.SSE)
        print('                               SSD:',self.SSD)
        print('                            Angle :',self.angle)
        print('---------------------------------------------------------------')
        
       
