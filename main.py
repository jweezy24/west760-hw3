import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def parse_d2z():
    path = "data/D2z.txt"
    points = []
    with open(path,"r+")as f:
        for line in f.readlines():
            x,y,z = line.split(" ")
            x = float(x)
            y = float(y)
            z = int(z)
            p = (x,y,z)
            points.append(p)
    return points

def parse_emails():
    df = pd.read_csv('data/emails.csv')
    features = df.to_numpy()[:,1:-1]
    labels = df.to_numpy()[:,-1]
    return features,labels

def eval_neighbors_d2z(p,data,labels=None,k=1,use_email_dataset=True,p_is_matrix=True):
    distances = []
    import time
    if use_email_dataset:
        assert(type(labels) !=  type(None))

        p = p.astype(float)
        data = data.astype(float)
        start = time.time()
        if not p_is_matrix:
            p = p.reshape((1,len(p)))
            p = p.astype(float)
            for i in range(len(data)):
                p2 = data[i,:].astype(float)
                z = labels[i]
                # d = euclidean(p,p2)
                p2 = p2.reshape((1,len(p2)))
                
                
                d = cdist(p,p2,metric="euclidean")
                d = d.flatten()[0]
                distances.append((d,z))
            distances.sort(key=lambda x: x[0])
        else:
            d = cdist(p,data,metric="euclidean")
            ret = []
            labels = labels.astype(int)
            confidence = []
            
            for d_to_p in d:
                ls = labels[np.argsort(d_to_p)][:k]
                ls2 = labels[np.argsort(d_to_p)][:k]
                ls = np.bincount(ls)
                b = np.argmax(ls)
                tracker = [0,0]
                for i in ls2:
                    tracker[i]+=1
                conf = tracker[b]/sum(tracker)
                # print(b,conf)
                ret.append(b)
                confidence.append(1-conf)
            end = time.time()
            print(f"Done with distances. Time:{end-start} seconds")
            return ret,confidence                 
        
    else:
        for x,y,z in data:
            p2 = (x,y)
            d = euclidean(p,p2)
            distances.append((d,z))
        distances.sort(key=lambda x: x[0])

    
    if k>1:
        print(f"Done with distances. Time:{end-start} seconds")
        nearest = distances[:k]
        assert(len(nearest) == k)
        table = {}
        for d,l in nearest:
            if l not in table.keys():
                table[l] =1
            else:
                table[l]+=1
        its = list(table.items())
        its.sort(key=lambda x: x[1])
        return its[0][0]
    else:
        print(f"Done with distances. Time:{end-start} seconds")
        return distances[0][1]        

def fake_dataset():
    data = []
    i= -2
    j= -2
    while i <= 2:
        j=-2
        while j <= 2:
            p = (i,j)
            data.append(p)
            j+=0.1
        i+=0.1
    return data

def eval_folds(data,test_set=False):
    from sklearn.neighbors import KNeighborsClassifier
    features,labels = data
    # features = preprocessing.normalize(features)
    fold_size = 1000
    ks = [1,3,5,7,10]
    ks = [5]
    x_axis = []
    y_axis = []
    max_ac = 0
    for k in ks:
        with open("tmp_file.txt", "a+") as f:
                f.write(f"k = {k}\n")
        c=0
        for i in range(0,features.shape[0],fold_size):
            tracking = [[0,0],[0,0]]
            testing_labels = labels[i:i+fold_size]
            testing = features[i:i+fold_size,:]
            training = np.vstack((features[0:i,:],features[i+fold_size:,:]))
            training_labels = np.hstack((labels[0:i],labels[i+fold_size:]))
            
            if c!= test_set:
                continue
        
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(training, training_labels.astype(int))
            results,confidences = eval_neighbors_d2z(testing,training,use_email_dataset=True,labels=training_labels,k=k,p_is_matrix=True)
            confidences = neigh.predict_proba(testing)
            # print(confidences)
            x,y,t = roc_curve(testing_labels.astype(int),confidences[:,1])
            ac = auc(x,y)
            if ac > max_ac:
                max_ac = ac
                x_axis = x
                y_axis = y
                print(max_ac)
                
            for j in range(testing.shape[0]):
                p = testing[j]
                l = results[j]
                # l = eval_neighbors_d2z(p,training,use_email_dataset=True,labels=training_labels,k=k)
                tracking[testing_labels[j]][l]+=1
        
            with open("tmp_file.txt", "a+") as f:
                f.write(f"f{c}={tracking}\n")
            c+=1
    
    return x_axis,y_axis,max_ac

def evaluate_metrics():
    f1 = [[471, 252], [173, 104]]
    f2 = [[465, 250], [185, 100]]
    f3 = [[487, 228], [183, 102]]
    f4 = [[486, 229], [179, 106]]
    f5 = [[425, 290], [192, 93]]
    folds = [f1,f2,f3,f4,f5]
    count = 0

    for f in folds:
        
        accuracy = (f[0][0] + f[1][1])/(f[0][0] + f[1][1]+ f[1][0]+ f[0][1])
        recall = (f[0][0])/(f[0][0] + f[1][0])
        percision = (f[0][0])/(f[0][0] + f[0][1])

        print(f"FOLD {count+1}\t Accuracy: {accuracy}\tRecall:{recall}\tPercision:{percision}")
        count+=1 

def sigma(w,x,l=None,return_probs=False):
    w = w.astype(float)
    x = x.astype(float)

    p = (w@x.T)
    res = (1/(1+ np.exp(-1*p)))
    if not return_probs:
        if len(x.shape)>1:
            for i in range(len(res)):
                pred = res[i]
                if pred >= 0.5:
                    res[i] = 1
                else:
                    res[i] = 0
        else:
            if res >= 0.5:
                return 1
            else:
                return 0
    else:
        return res
    return res

def step(w,x,y,lr):
    
    w = w -  lr*(x.T@(sigma(w,x,l=y)-y))
    return w

def logistic_regression_predict(x,w):
    validate = w@x.T
    p = [0 for i in range(5000)]
    c=0
    for pred in validate:
        if pred >= 0:
            p[c] = 1
        else:
            p[c]=0
    # print(p)
    return np.array(p)

def train_lr(training):
    w = np.ones(3000)
    w_old = np.zeros(3000)
    training,y = training
    training = preprocessing.normalize(training)
    lr = 0.01#2/5000
    max_iters = 5000
    c = 0
    min_w = []
    min_v = 100
    hit_max = False
    while euclidean(w_old,w) > 0.1:

        w_old = w
        w = step(w,training,y,lr)
        preds = sigma(w,training)
        loss = sum(abs(y.flatten()-preds))/5000
        if loss < min_v:
            min_w = w
            min_v = loss 
        if c > max_iters:
            hit_max = True
            break
        print(f"Loss = {loss}, C = {c}")
        c+=1
    if hit_max:
        return min_w
    else:
        return w


def logistic_regression(data):
    w = np.ones(3000)
    x,y = data
    y = y.astype(int)
    fold_size = 1000
    max_auc = 0
    x_axis = []
    y_axis =[]
    test_set_num=0
    for i in range(0,x.shape[0],fold_size):
        tracking = [[0,0],[0,0]]
        testing_labels = y[i:i+fold_size]
        testing = x[i:i+fold_size,:]
        training = np.vstack((x[0:i,:],x[i+fold_size:,:]))
        training_labels = np.hstack((y[0:i],y[i+fold_size:]))
        w = train_lr((training,training_labels))
        predictions = []
        for j in range(0,len(testing)):
            p=testing[j]
            pred = sigma(w,p)
            tracking[pred][testing_labels[j]]+=1
        
        vals = sigma(w,testing,return_probs=True)
        predictions = vals/vals.sum()
        X,Y,z = roc_curve(testing_labels,predictions)
        ac = auc(X,Y)
        if ac > max_auc:
            max_ac = ac
            x_axis=X
            y_axis=Y
            test_set_num = i
            print(f"TEST SET = {i}")
        
        with open("./tmp2.txt","a+") as f:
            f.write(f"{tracking}\n")
    
    return w,x_axis,y_axis,max_ac,test_set_num

def make_knn_cross_validation_plot():
    k = 1
    f0=[[591, 124], [51, 234]]
    f1=[[615, 108], [37, 240]]
    f2=[[624, 92], [45, 239]]
    f3=[[613, 93], [53, 241]]
    f4=[[542, 152], [73, 233]]
    k1_folds = [f0,f1,f2,f3,f4]
    k = 3
    f0=[[597, 118], [36, 249]]
    f1=[[624, 99], [51, 226]]
    f2=[[626, 90], [54, 230]]
    f3=[[637, 69], [51, 243]]
    f4=[[549, 145], [82, 224]]
    k3_folds = [f0,f1,f2,f3,f4]
    k = 5
    f0=[[595, 120], [43, 242]]
    f1=[[634, 89], [59, 218]]
    f2=[[637, 79], [50, 234]]
    f3=[[632, 74], [57, 237]]
    f4=[[551, 143], [77, 229]]
    k5_folds = [f0,f1,f2,f3,f4]
    k = 7
    f0=[[594, 121], [42, 243]]
    f1=[[635, 88], [51, 226]]
    f2=[[640, 76], [49, 235]]
    f3=[[639, 67], [59, 235]]
    f4=[[551, 143], [78, 228]]
    k7_folds = [f0,f1,f2,f3,f4]
    k = 10
    f0=[[631, 84], [53, 232]]
    f1=[[650, 73], [58, 219]]
    f2=[[655, 61], [61, 223]]
    f3=[[658, 48], [65, 229]]
    f4=[[571, 123], [95, 211]]
    k10_folds = [f0,f1,f2,f3,f4]

    all_folds= [k1_folds,k3_folds,k5_folds,k7_folds,k10_folds]
    x_axis = [1,3,5,7,10]
    y_axis = []
    for folds in all_folds:
        count = 0
        average_acc = 0
        for f in folds:
            
            accuracy = (f[0][0] + f[1][1])/(f[0][0] + f[1][1]+ f[1][0]+ f[0][1])
            recall = (f[0][0])/(f[0][0] + f[1][0])
            percision = (f[0][0])/(f[0][0] + f[0][1])

            # print(f"FOLD {count+1}\t Accuracy: {accuracy}\tRecall:{recall}\tPercision:{percision}")
            average_acc+=accuracy
            count+=1 
        average_acc/=5
        y_axis.append(average_acc)

    plt.plot(x_axis,y_axis)
    plt.savefig("roc_thing.pdf")

def make_plot(data,fake):
    import matplotlib.patches as mpatches
    my_colors = {0:'red',1:'blue',2:'blue'}

    l1_patch = mpatches.Patch(color='red', label='Labeled 0')
    l2_patch = mpatches.Patch(color='blue', label='Labeled 1')
    l3_patch = mpatches.Patch(color='black', label='Training Data')
    
    for i in fake:
        l = eval_neighbors_d2z(i,data,k=3)
        plt.scatter(i[0],i[1],color=my_colors[l])
    for p in data:
        plt.scatter(p[0],p[1], color="black")

    plt.legend(handles=[l1_patch,l2_patch,l3_patch])
    plt.savefig("plt.pdf")

            
if __name__ == "__main__":
    # data = parse_d2z()
    # fake = fake_dataset()
    # make_plot(data,fake)

    data_emails = parse_emails()
    # evaluate_metrics()
    w,x_axis,y_axis,ac1,test_set = logistic_regression(data_emails)
    x_axis2,y_axis2,ac2 = eval_folds(data_emails,test_set=test_set)
    plt.plot(x_axis,y_axis,label=f"Logistic Regression AUC = {ac1}")
    plt.plot(x_axis2,y_axis2, label=f"5NN Algorithm AUC = {ac2}")
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.legend()
    plt.title("ROC Curve of 5NN vs Logistic Regression")

    plt.savefig("ROC_CURVE.pdf")
    # make_knn_cross_validation_plot()