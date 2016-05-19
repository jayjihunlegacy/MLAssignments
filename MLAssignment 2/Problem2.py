from MLP import *
import pickle
import timeit
TRAIN_FILE_NAME = 'Train.csv'

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

def import_data():
    '''
    Method to import training data. (Possibly test data, too)
    RETURN data sets.
    '''

    train_portion = 0.8

    with open(TRAIN_FILE_NAME, 'r') as f:
        tuples = f.readlines()
    
    #separate legend.
    legend = tuples[0]
    tuples = tuples[1:]

    data_x = list()
    data_y = list()
    for record in tuples:
        #split csv format into list.
        splitted = record.split(',')

        #take only input(x)
        datum_x = [float(value) for value in splitted[1:-1]]

        #take only output(y)
        datum_y = 1 if splitted[-1].strip()=='Pass' else 0

        #put into list
        data_x.append(datum_x)
        data_y.append(datum_y)
    
    num_of_total = len(data_x)
    num_of_train = int(num_of_total*train_portion)

    #split data into 1.Train and 2.Valid
    train_x = data_x[0:num_of_train]
    train_y = data_y[0:num_of_train]

    train_set = shared_dataset((train_x,train_y))

    valid_x = data_x[num_of_train:]
    valid_y = data_y[num_of_train:]
    valid_set = shared_dataset((valid_x,valid_y))

    return (train_set, valid_set)

def initialize_weights(random=False):
    if random:
        return None


    '''
    Method to initialize weights in Neural Network.

    First, attempt to import from file.

    RETURN None when no file exists.
    '''
    return None

def sgd_optimize(dataset,param_v):
    '''
    Method to actually perform Stochastic Gradient Descent
    Save weights after optimization.

    '''
    #hyper parameters
    feats=24
    learning_rate=0.001
    batch_size=100
    L1_reg=0.001
    L2_reg=0.001
    n_epochs = 5000

    train_x, train_y = dataset[0]
    valid_x, valid_y = dataset[1]

    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size

    #build model
    print('Building models...')
    index = T.lscalar()
    x=T.matrix('x')
    y=T.ivector('y')

    rng = numpy.random.RandomState(11557)
    model = MLP(
        rng=rng,
        input=x,
        n_in=feats,
        n_hidden = feats//2,
        n_out=2,
        param_v=param_v
        )
    cost = (model.NLL(y) + L1_reg*model.L1 + L2_reg*model.L2_sqr)

    valid_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: valid_x[index * batch_size : (index + 1) * batch_size],
            y: valid_y[index * batch_size : (index + 1) * batch_size]
            }
        )
    print('1. Valid built')
    gparams = [T.grad(cost, param) for param in model.params]

    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(model.params, gparams)]
    print('2. Gradient calculated.')
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[index * batch_size : (index + 1) * batch_size],
            y: train_y[index * batch_size : (index + 1) * batch_size]
            }
        )
    print('3. Train built.')
    print('Training model...')
    patience = 1000000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience//2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1)%validation_frequency == 0:
                validation_losses = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%'%(epoch,minibatch_index+1,n_train_batches,this_validation_loss*100))

                if this_validation_loss < best_validation_loss :
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter*patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter


            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%'%(best_validation_loss*100))
    print('The code run for %d epochs, with %f epochs/sec' %(epoch, epoch/(end_time-start_time)))

    print('Saving weights...')


def test():
    pass


def main():
    print('Start program...')
    dataset = import_data()
    param_v = initialize_weights()
    sgd_optimize(dataset,param_v)

if __name__=='__main__':
    main()