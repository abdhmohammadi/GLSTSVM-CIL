from tarfile import data_filter
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
num_samples = 350
num_features = 2
num_informative_features = 1
rate = 6
for r in range(1,21):
    name = str(r)+'-noise05'

    df = pd.read_csv('D:/Data Science/Python/datasets/Syntetics/noise05/'+name +'.csv')
    df0 = df[df['label']==0]
    df1 = df[df['label']==1]
    n0 = df0.shape[0]
    n1=df1.shape[0]

    test0 = df0.sample(n=int(0.30*n0)+1,random_state=42)
    test1 = df1.sample(n=int(0.30*n1)+1,random_state=42)
    test   = np.vstack((test0.values,test1.values))

    train0 = df0.drop(index= test0.index)
    train1 = df1.drop(index= test1.index)
    train  = np.vstack((train0,train1))

    df = pd.DataFrame(data=test,columns=test0.columns)
    df = df.sample(n=df.shape[0],random_state=42)
    df.to_csv(index=False,path_or_buf='D:/Data Science/Python/datasets/Syntetics/noise05/'+name +'-test.csv')

    df = pd.DataFrame(data=train,columns=train0.columns)
    df =df.sample(n=df.shape[0],random_state=42)
    df.to_csv(index=False,path_or_buf='D:/Data Science/Python/datasets/Syntetics/noise05/'+name +'-train.csv')
    print(r)

print('DONE.')

"""
X, y = make_classification(n_samples=num_samples, n_features=num_features,weights=[1/(rate+1),1-1/(rate+1)],
                            n_informative=num_informative_features, n_redundant=0,
                            n_clusters_per_class=1, random_state=100)

data = np.hstack((X,y.reshape(-1,1)))
df = pd.DataFrame(data=data,columns=['X','Y','label'])

df.to_csv(index=False,path_or_buf='D:/Data Science/Python/datasets/Syntetics/'+name +'-normal.csv')

X1 = X + np.random.normal(0,0.5,X.shape)
data = np.hstack((X1,y.reshape(-1,1)))
df = pd.DataFrame(data=data,columns=['X','Y','label'])

df.to_csv(index=False,path_or_buf='D:/Data Science/Python/datasets/Syntetics/'+name +'-noise05.csv')


X2 = X + np.random.normal(0,1,X.shape)
data = np.hstack((X2,y.reshape(-1,1)))
df = pd.DataFrame(data=data,columns=['X','Y','label'])

df.to_csv(index=False,path_or_buf='D:/Data Science/Python/datasets/Syntetics/'+name +'-noise1.csv')

X3 = X + np.random.normal(0,2,X.shape)
data = np.hstack((X3,y.reshape(-1,1)))
df = pd.DataFrame(data=data,columns=['X','Y','label'])

df.to_csv(index=False,path_or_buf='D:/Data Science/Python/datasets/Syntetics/'+name +'-noise2.csv')
"""
exit()

"""y = y.reshape(num_samples,1)
X = np.hstack((X,y))

df = pandas.DataFrame(data=X,columns = ['X','Y','Label'])

df.to_csv(index=False,path_or_buf='D:/DATA SCIENCE/Python/datasets/Imbalance rate/Normal-100x2--.csv')
"""
print((X).mean(axis=0))