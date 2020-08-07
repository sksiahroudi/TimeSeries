from sklearn import svm
import numpy as np
from sklearn.utils import resample
def supportVec(X,y,Train,Label):
    rbf_svc = svm.SVC(kernel='rbf')
    for itm in y:
        if itm==1:
            itm=0
        else:
            itm=1
    rbf_svc.fit(X, y)

    supportVectors=rbf_svc.support_vectors_
    positive_size=Train["Label"].sum()

    Ind=rbf_svc.support_

    XP=Train[Train.index.isin( Ind)]
    support_size,e=XP.shape
    if (support_size>positive_size):
        XPS=resample(XP,replace=False,n_samples=positive_size,random_state=123)
    XN=Train[~Train.index.isin( XPS.index)]
    return XPS,XN
