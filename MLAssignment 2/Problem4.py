import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.externals import joblib

try_N = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
class_n = 107
classes=[i+1 for i in range(class_n)]

def importtrain():
    train_fraction = 0.9;
    try:
        with open('train.fasta','r') as f:
            tuples=f.readlines()
    except:
        print('Failed to read train.fasta')
        exit()

    tuples.remove('\n')
    tuples = [line.strip() for line in tuples];    

    seq = list()
    next_seq = str()
    for line in tuples:
        if line.startswith('>'):
            if next_seq:
                seq.append(next_seq)
            next_seq=""
            continue
        else:
            next_seq+=line

    seq.append(next_seq)

        
    try:
        with open('train_ans.txt', 'r') as f:
            tuples=f.readlines()
    except:
        print('Failed to read train_ans.txt')
        exit()

    label = [int(line.split('\t')[-1].strip()) for line in tuples]
    print('Data imported.')

    num_of_data = len(seq)
    
    num_of_train = int(num_of_data*train_fraction)
    num_of_valid = num_of_data-num_of_train;

    train_set = (seq[0:num_of_train], label[0:num_of_train])
    
    valid_set = (seq[num_of_train:], label[num_of_train:])

    return  (train_set, valid_set)

def train(train_set):
    train_seq, train_lab = train_set

    classified = dict()
    for i in classes:
        classified[i] = list()

    for [index,sequence] in enumerate(train_seq):
        tmp = sequence.replace('A','0')
        tmp = tmp.replace('T','1')
        tmp = tmp.replace('G','2')
        tmp = tmp.replace('C','3')
        vectorized = []
        vectorized.extend(tmp)
        vectorized = [int(element) for element in vectorized]
        vectorized = np.matrix(vectorized)
        vectorized = vectorized.transpose()
        classified[train_lab[index]].append(vectorized)

    hmms = dict()
    
    
    for N in try_N:
        print('Start N =',N)
        for i in classes:
            hmms[i] = hmm.MultinomialHMM(n_components=N)
        for i in classes:
            print('Start class :',i)
            X=list()
            lengths=tuple()
            X=np.concatenate(classified[i])
            for sequence in classified[i]:
                lengths+=(len(sequence),)
            hmms[i].fit(X,lengths)

        foldername = 'C:/Users/Jihun/Desktop/HMMs/'

        for i in classes:
            filename = 'HMM'+str(N)+'_'+str(i)+'.txt'
            path=foldername+filename
            joblib.dump(hmms[i], path)
    

def predict(hmms, X):    
    prob=[]
    for i in classes:
        prob.append(hmms[i].score(X))

    return prob.index(max(prob))+1

def predicts(hmms, X, lengths):
    pass
   

def eval_Ein(hmms, train_set):
    pass

def test(data_set):
    data_x, data_y = data_set
    foldername = 'C:/Users/Jihun/Desktop/HMMs/'

    newdata_x=list()
    lengths=tuple()
    for [index,sequence] in enumerate(data_x):
        tmp = sequence.replace('A','0')
        tmp = tmp.replace('T','1')
        tmp = tmp.replace('G','2')
        tmp = tmp.replace('C','3')
        vectorized = []
        vectorized.extend(tmp)
        vectorized = [int(element) for element in vectorized]
        vectorized = np.matrix(vectorized)
        vectorized = vectorized.transpose()
        newdata_x.append(vectorized)
        lengths+=len(sequence)

    data_x=newdata_x
    print('Start evaluating...')
    for N in try_N:
        hmms = dict()
        for i in classes:
            filename = 'HMM'+str(N)+'_'+str(i)+'.txt'
            path = foldername+filename
            hmms[i] = joblib.load(path)

        num_of_data=len(data_x)
        wrong = 0
        preds = predicts(hmms,data_x,lengths)
        for i in range(num_of_data):
            if data_y[i]!=preds[i]:
                wrong = wrong + 1


        '''
        for i in range(num_of_data):
            
            pred = predict(hmms,data_x[i])
            ans = data_y[i]
            if ans != pred:
                wrong = wrong + 1
                print('[%i,%i], pred:%i, ans:%i'%(N,i,pred,ans))
            else:
                print('[%i,%i], O'%(N,i))
        '''
        error = wrong * 100/num_of_data
        print('For N=%i, error : %.3f%%'%(N,error))

def main():
    train_set, valid_set= importtrain()
    train(train_set)
    #test(train_set)
    
if __name__=='__main__':
    main()



'''
for sequence in sequences[0:100]:
    C_n = sequence.count('C')
    G_n = sequence.count('G')
    length = len(sequence)
    CG_n = sequence.count('CG')
    CG_E = C_n*G_n//length
    print('Observed CG : %i, Expected CG : %i. Length is %i. Ratio : %f. Content : %.1f%%'%(CG_n,CG_E,length,CG_n/CG_E, CG_n*100/length))

'''


'''
model = hmm.GaussianHMM(n_components=3, covariance_type='full')
model.startprob_=np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7,0.2,0.1],[0.3,0.5,0.2],[0.3,0.3,0.4]])

model.means_=np.array([[0,0],[3,-3],[5,10]])
model.covars_=np.tile(np.identity(2),(3,1,1))
X,Z=model.sample(50)

#plt.plot(X[:,0], X[:,1], '.-', label='obersvations', ms=6, mfc='orange',alpha=0.7)
#plt.show()
print(Z)
print(X)
remodel = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
remodel.fit(X)
Z2 = remodel.predict(X)
'''