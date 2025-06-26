
import sys
from time import perf_counter
import numpy as np

sys.path.append('/base-tools')
from BaseTSVMClassifier import BaseTSVMClassifier

class LSTSVM(BaseTSVMClassifier):

  def __init__(self,c1 =1.0,c2= 1.0 ,kernel_name = 'rbf',mu=0.1,epsilon=0.1):
     super().__init__(kernel_name = kernel_name, mu=mu)
     self.c1=c1
     self.c2=c2
     self.epsilon =epsilon
     self.wc_time = 0.0
    
  def fit(self,X_train,y_train):
    
    A, m1, B, m2, self.classes , self.n_features= self.split(X_train,y_train)
    start = perf_counter()

    if self.kernel_name == None or self.kernel_name =='':

      F = np.hstack((B,np.ones((m2,1))))
      E = np.hstack((A,np.ones((m1,1))))

      w1 = -np.linalg.inv((F.T@F+(1/self.c1)*E.T@E).astype(np.float32)) @ F.T

      w1 = w1  @ np.ones((w1.shape[1],1))

      w2 = -np.linalg.inv((E.T @ E + (1/self.c2)*F.T@F).astype(np.float32))@ E.T

      w2 = w2  @ np.ones((w2.shape[1],1))

    else:

      self.C = np.vstack((A,B))
      G = self.Kernel(A,self.C)
      H = self.Kernel(B,self.C)

      G = np.hstack((G, np.ones((G.shape[0],1))))
      H = np.hstack((H, np.ones((H.shape[0],1))))

      if m1 <= m2:
        HH  = H @ H.T
        
        Y = H.T @ np.linalg.inv(self.epsilon*np.identity(H.shape[0]) + HH) @ H
        
        Y = (np.identity(Y.shape[0]) - Y)/self.epsilon

        YG = Y @ G.T
        GY = G @ Y
        GYG = G @ YG

        I = np.identity(GYG.shape[0])

        w1 = -(Y - YG @ np.linalg.inv(self.c1*I + GYG) @ GY)@H.T
        w1 = w1 @ np.ones((w1.shape[1],1))
        
        w2 = (Y - YG @ np.linalg.inv((1/self.c2)*I + GYG) @ GY) @ G.T
        
        w2 = w2 @ np.ones((w2.shape[1],1))*self.c2

      else:
        GG  = G @ G.T
        Y = G.T @ np.linalg.inv(self.epsilon*np.identity(G.shape[0]) + GG) @ G
        Y = (np.identity(Y.shape[0]) - Y)/self.epsilon

        YH = Y@H.T
        HY  = H @ Y
        HYH = H @ YH
        
        I = np.identity(HYH.shape[0])

        w1 = -self.c1*(Y - YH @ np.linalg.inv((1/self.c1)*I + HYH) @ HY) @ H.T
         
        w1 = w1 @ np.ones((w1.shape[1],1))
         
        w2 = (Y - YH @ np.linalg.inv(self.c2*np.identity(H.shape[0]) + HYH) @ HY) @ G.T

        w2 = w2 @ np.ones((w2.shape[1],1))
    
    self.training_time = perf_counter() - start
    self.w1 = w1
    self.w2 = w2 

  def predict(self,x_test):

    predicted = []
    
    X = np.asarray(x_test)
    
    for x in X:
      x = np.asarray(x)

      if self.kernel_name == None or self.kernel_name =='':
        x = np.hstack((x,[1]))
      else:

        x = np.asarray(x).reshape(1, -1)

        x = self.Kernel(x,self.C) 

        x = np.hstack((x[0],[1]))
       
      f1 = x @ self.w1
      f2 = x @ self.w2
      
      predicted.append( self.classes[0] if abs(f1)<= abs(f2) else self.classes[1])
      
    return predicted
  
  def classification_report(self):
        print()
        print('--------------------- LSTSVM REPORT ------------------')
        print('Hyperplane parameters(c1,c2):',str(self.c1) + ', ' + str(self.c2))
        print('           Class Imbalace Rate(IR):',self.IR)
        print('                            Kernel:',self.kernel_name)
        print('              Kernel parameter(mu):',self.mu)
        print('                        Train size:',str(self.C.shape[0])+' \u00d7 '+str(self.C.shape[1]))
        print('              Data reading time(s):',self.reading_time)
        print('          Weight computing time(s):',self.wc_time)
        print('                  Training time(s):',self.training_time)
        print('---------------------------------------------------------------')
        
       
