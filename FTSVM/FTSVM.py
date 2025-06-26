
import time
import numpy
import sys
from qpsolvers import solve_qp

sys.path.append('/base-tools')
from BaseTSVMClassifier import BaseTSVMClassifier

class FTSVM(BaseTSVMClassifier):
  
  def __init__(self,c1 =1.0, c2= 1.0, kernel_name = 'rbf',mu=0.1, solver = 'cvxopt',epsilon = 0.01):
     super().__init__(kernel_name=kernel_name, mu=mu)
     self.c1=c1
     self.c2=c2
     self.solver = solver
     self.epsilon = epsilon
     
  def fit(self,X_train,y_train): 

    t1 = time.perf_counter()
    
    A, m1, B, m2, self.classes, self.n_features = self.split(X_train,y_train)

    start = time.perf_counter()
    t1 = start - t1

    S1 = self.compute_weights(A)
    S2 = self.compute_weights(B)
    
    self.wc_time = time.perf_counter() - start
    
    e1 = numpy.ones(m1).reshape(-1,1)
    e2 = numpy.ones(m2).reshape(-1,1)
    
    start = time.perf_counter()

    if self.kernel_name == None or self.kernel_name=='':
      H = numpy.hstack((A,e1))
      G = numpy.hstack((B,e2))
      self.C = None
    else:
        self.C = numpy.vstack((A,B))
        k1 = self.Kernel(A,self.C)
        H = numpy.hstack((self.Kernel(A,self.C),e1))
        G = numpy.hstack((self.Kernel(B,self.C),e2))
    
    H_ = numpy.linalg.inv(H.T@H + self.epsilon* numpy.eye(H.shape[1]))
    
    # Solves the quadratic problem:
    # minimize 1/2*xT.P.x + q.x
    # s.t       G.x <= h
    #           A.x  = b
    #           lb <= x <= ub
    ################################
    # we must solve:
    # maximize e2T.a - 1/2*aT.G(HT.H + epsilon*I)^{-1}GT.a
    # s.t         0<= a <= c1
    P = G @ H_ @ G.T
    alpha = solve_qp(P = P ,q = -e2, lb = numpy.zeros(m2), ub = self.c1 * S2.diagonal(),solver=self.solver)

    alpha = alpha.reshape(m2,1)

    u = -H_ @ G.T @ alpha

    b1 = numpy.float64(u[len(u)-1])

    w1 =u[:len(u)-1].T[0]

    # Secund hyperplane
    G_ = numpy.linalg.inv(G.T @ G + self.epsilon * numpy.eye(G.shape[1]))

    P = H @ G_ @ H.T

    alpha = solve_qp(P = P ,q = -e1, lb = numpy.zeros(m1), ub = self.c2 * S1.diagonal(),solver=self.solver)
    
    u = G_ @ H.T @ alpha.reshape(m1,1)

    b2 = numpy.float64(u[len(u)-1])

    w2 =u[:len(u)-1].T[0]

    self.b1 = b1
    self.b2 = b2
    self.w1 = w1
    self.w2 = w2
    
    self.training_time = time.perf_counter() - start + t1
    
    return b1, w1, b2, w2

  def predict(self,X_test):

    X = numpy.asarray(X_test)

    predict = []

    for x in X:

        x = numpy.asarray(x)
        if self.kernel_name != '' and self.kernel_name !=None: x = self.Kernel(x.reshape(-1,1).T,self.C)

        value1 = abs(x @ self.w1 + self.b1) 
        value2 = abs(x @ self.w2 + self.b2)
        predict.append(self.classes[0] if value1 <= value2 else self.classes[1])

    return predict
  
  def compute_weights(self, data,epsilon=0.01):
    
    data = numpy.asarray(data)
    # Center of class(mean or median is used)
    center =sum(data,0)/len(data)
    # init cubic zero matrix
    norms = numpy.zeros((data.shape[0],data.shape[0]))

    for i in range(len(data)):
       
       point = numpy.asarray(data[i])
       norms[i][i] = numpy.linalg.norm(point- center)
    
    # calculate mem function as 1- d(x)/(max(d)+ epsilon)
    d_max = norms.diagonal().max()
    norms = norms/(d_max + epsilon)
    norms = numpy.identity(len(data)) - norms
    return numpy.asarray(norms)
  
  def classification_report(self):
        print()
        print('--------------------- FTSVM REPORT ------------------')
        print('Hyperplane parameters(c1,c2):',str(self.c1) + ', ' + str(self.c2))
        print('           Class Imbalace Rate(IR):',self.IR)
        print('                            Kernel:',self.kernel_name)
        print('              Kernel parameter(mu):',self.mu)
        print('                        Train size:',str(self.C.shape[0])+' \u00d7 '+str(self.C.shape[1]))
        print('              Data reading time(s):',self.reading_time)
        print('          Weight computing time(s):',self.wc_time)
        print('                  Training time(s):',self.training_time)
        print('---------------------------------------------------------------')
        
       
