import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out,w_values,b_values):
        self.W = theano.shared(value=w_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def NLL(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, w_values, b_values):
        self.input = input
        W = theano.shared(value=w_values,name='W',borrow=True)
        b = theano.shared(value=b_values,name='b',borrow=True)
        self.W=W
        self.b=b
        lin_output = T.dot(input,self.W) + self.b
        self.output = activation(lin_output)
        self.params = [self.W,self.b]

class MLP(object):
    def __init__(self,rng,input,n_in,n_hidden,n_out,param_v):
        activation_f = T.tanh
        if param_v is not None:
            [hidden_W_value,
            hidden_b_value,
            logRegression_W_value,
            logRegression_b_value] = param_v
        else:
            hidden_W_value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6/(n_in+n_hidden)),
                    high=numpy.sqrt(6/(n_in+n_hidden)),
                    size=(n_in,n_hidden)
                    ),
                dtype=theano.config.floatX
                )

            if activation_f == T.nnet.sigmoid:
                hidden_W_value*=4

            hidden_b_value = numpy.zeros((n_hidden,),dtype=theano.config.floatX)

            logRegression_W_value = numpy.zeros((n_hidden,n_out),dtype=theano.config.floatX)

            logRegression_b_value = numpy.zeros((n_out,), dtype=theano.config.floatX)        

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation_f,
            w_values=hidden_W_value,
            b_values=hidden_b_value
            )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            w_values=logRegression_W_value,
            b_values=logRegression_b_value
            )

        self.L1=(
            abs(self.hiddenLayer.W).sum()
            +abs(self.logRegressionLayer.W).sum()
            )

        self.L2_sqr=(
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
            )

        self.NLL = self.logRegressionLayer.NLL

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params+self.logRegressionLayer.params

        self.input = input