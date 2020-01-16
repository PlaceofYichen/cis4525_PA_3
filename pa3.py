'''
Add the code below the TO-DO statements to finish the assignment. Keep the interfaces
of the provided functions unchanged. Change the returned values of these functions
so that they are consistent with the assignment instructions. Include additional import
statements and functions if necessary.
'''

import csv
import numpy as np
import matplotlib.pyplot as plt

'''
The loss functions shall return a scalar, which is the *average* loss of all the examples
'''

'''
For instance, the square loss of all the training examples is computed as below:

def squared_loss(train_y, pred_y):

    loss = np.mean(np.square(train_y - pred_y))

    return loss
'''

def logistic_loss(train_y, pred_y):
    
    # TO-DO: Add your code here
    # reshape to (-1, )
    train_y = train_y.reshape(-1)
    pred_y = pred_y.reshape(-1)

    #loss = np.mean(-train_y*pred_y + np.log(1.0+np.exp(pred_y)))
    loss = np.mean(np.log(1.0+np.exp(-train_y*pred_y)))

    return loss

def hinge_loss(train_y, pred_y):
    
	# TO-DO: Add your code here
    train_y = train_y.reshape( -1)
    pred_y = pred_y.reshape(-1)
    
    loss = np.mean(np.clip(1-train_y*pred_y, 0, 1-train_y*pred_y))

    return loss

'''
The regularizers shall compute the loss without considering the bias term in the weights
'''
def l1_reg(w):

    # TO-DO: Add your code here
    w = w.reshape(-1)
    l1 = np.sum(np.abs(w))

    return l1

def l2_reg(w):

    # TO-DO: Add your code here
    w = w.reshape(-1)
    l2 = np.sqrt(np.sum(np.power(w, 2)))
    return l2

def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=None, regularizer=None):

    # TO-DO: Add your code here 
    col = len(train_x[0])
    
    w = np.random.random((1, col+1))
    losses = []
    for itr in range(1000):
        h = 1e-4
        grad_w = []
        # calculate the original loss
        ori_loss = loss(train_y, train_x.dot(w[:, :-1].T)+w[:, -1])
        # if regularizer function is not None, means that need to regularize the weights
        if regularizer is not None:
            ori_loss += lambda_val*regularizer(w[:, :-1])
            
        for i in range(col):
            w_i = np.zeros_like(w)
            w_i[:, i] += h
            w_i += w

            # calculate the difference after and before w_i adds h
            g_w_i = loss(train_y, train_x.dot(w_i[:, :-1].T)+w_i[:,-1]) - ori_loss
            
            # if regularizer function is not None, means that need to regularize the weights
            if regularizer is not None:
                g_w_i += lambda_val*regularizer(w_i[:, :-1])
            
            # get gradient of w_i
            g_w_i = g_w_i/h
            grad_w.append(g_w_i)
        
        grad_b = loss(train_y, train_x.dot(w[:, :-1].T)+w[:, -1]+h) - ori_loss
        grad_b = grad_b/h
        
        grad_w = np.array(grad_w).reshape(1, -1)

        # update weights and bias
        w[:, :-1] = w[:, :-1] - learn_rate*grad_w
        w[:, -1] = w[:, -1] - learn_rate*grad_b

        # record all training loss, and plot
        ls = loss(train_y, train_x.dot(w[:, :-1].T)+w[:, -1])
        losses.append(ls)
    
    plt.clf()
    plt.plot([i for i in range(1, len(losses)+1)], losses)
    plt.ylabel('loss', size=14)
    plt.xlabel('# itr', size=14)
    plt.title('Train loss', size=16)
    name = "lr_{}_{}_lambda_value_{}_{}".format(learn_rate, loss.__str__, lambda_val, regularizer.__str__ if regularizer is not None else 'None')
    plt.title("lr={}, loss={}, lambda={}, reg={}".format(learn_rate, loss.__str__, lambda_val, regularizer.__str__ if regularizer is not None else 'None'), size=14)
    plt.savefig(name+'_train_loss.png')
    return w

def test_classifier(w, test_x):

    # TO-DO: Add your code here
    
    pred_y = test_x.dot(w[:, :-1].T) + w[:, -1]

    return pred_y

# get the accuracy
def acc(test_y, pred_y):
    
    test_y = test_y.reshape(-1)
    pred_y = pred_y.reshape(-1)
    
    # if pred_y <=0 ,then get -1 ,else get 1
    pred_y = np.where(pred_y <= 0, -1, 1)
    # count all the elements if pred_y == test_y
    count = np.sum(np.where(pred_y==test_y, 1, 0))
    
    return count / pred_y.shape[0]

# standardize the data set, make it easy for training
def standardization(data):
    m = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    
    return (data-m) / sigma

# split x and y based on size
def split_train_test(X, y, test_size=0.2):
    pt = np.random.permutation(y.size)
    split_i = int(y.size * test_size)
    test_pt = pt[:split_i]
    train_pt = pt[split_i:]
    
    return X[train_pt], X[test_pt], y[train_pt], y[test_pt]

# make the k-fold in dataset as validation set;
#  the others are training sets.
def split_train_val(data, k=0):
    train, val = None, None
    for i in range(len(data)):
        if i== k:
            val = data[i]
        else:
            train = np.concatenate([train, data[i]]) if train is not None else data[i]
    
    return train, val
    

def main():

    # Read the training data file
    szDatasetPath = 'winequality-white.csv'
    listClasses = []
    listAttrs = []
    bFirstRow = True
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if bFirstRow:
                bFirstRow = False
                continue
            if int(row[-1]) < 6:
                listClasses.append(-1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))
            elif int(row[-1]) > 6:
                listClasses.append(+1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))

    dataX = np.array(listAttrs)
    dataY = np.array(listClasses)

    # 5-fold cross-validation
	# Note: in this assignment, preprocessing the feature values will make
	# a big difference on the accuracy. Perform feature normalization after
	# spliting the data to training and validation set. The statistics for
	# normalization would be computed on the training set and applied on
	# training and validation set afterwards.
	# TO-DO: Add your code here
 
    dataX = standardization(dataX)
    
    # split train and test dataset
    X_train, X_test, y_train, y_test = split_train_test(dataX, dataY, test_size=0.2)

    # split k-folds
    k_folds_X, k_folds_y = np.vsplit(X_train, 5), np.hsplit(y_train, 5)
    
    
    logistic_loss.__str__ = 'logistic'
    hinge_loss.__str__ = 'hinge'
    l1_reg.__str__ = 'l1'
    l2_reg.__str__ = 'l2'
    
    # all posible params
    loss_funcs = [logistic_loss, hinge_loss]
    reg_funcs = [None, l1_reg, l2_reg]
    lrs = [1e-2, 5e-2]
    lamds = [1e-5, 1e-6]
    
    for ls_func in loss_funcs:
        for reg in reg_funcs:
            for ld in lamds:
                for lr in lrs:
                    acc_scores = []
                    test_acc = []
                    for k in range(5):
                        X_train, X_val = split_train_val(k_folds_X, k=k)
                        y_train, y_val = split_train_val(k_folds_y, k=k)
                        
                        w = train_classifier(X_train, y_train, lr, ls_func, lambda_val=ld, regularizer=reg)
                        
                        # val
                        pred_y = test_classifier(w, X_val)
                        acc_score = acc(y_val, pred_y)
                        acc_scores.append(round(acc_score, 4))

                        # test
                        pred_y = test_classifier(w, X_test)
                        acc_score = acc(y_test, pred_y)
                        test_acc.append(round(acc_score, 4))
                    
                    print("*"*50)
                    print("{:15s}: {}".format('loss func', ls_func.__str__))
                    print("{:15s}: {}".format('regular func', reg.__str__ if reg is not None else 'not use reg function'))
                    print("{:15s}: {}".format('lambda', ld))
                    print("{:15s}: {}".format('learning rate', lr))
                    print("{:15s}: {}".format("k-fold val", acc_scores))
                    print("{:15s}: {}".format("test acc", test_acc))
                    print()
                    
    return None

if __name__ == "__main__":

    main()
    

