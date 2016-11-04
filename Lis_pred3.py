# coding:utf-8
import argparse
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
from models.PredNet import PredNet
from chainer import serializers
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default='10000', type=int, help='learning epoch')
parser.add_argument('--unchain_step', '-u', default='100', type=int, help='unchain freq')
parser.add_argument('--save_step', '-s', default='500', type=int, help='model save freq')
args = parser.parse_args()


class PredNet3Layer(chainer.Chain):

    def __init__(self , width , height , channels, batchSize):
        super(PredNet3Layer , self).__init__()
        self.add_link("l1",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))
        self.add_link("l2",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))
        self.add_link("l3",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))


    def __call__(self, x):

        h = self.l1(x)
        h = self.l2(x)
        h = self.l3(x)

        return h

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()


def learn(data_root_dir="./movies/"):
    """
    movie : npy file (preprocessed)
    Returns:

    """

    train_data = np.load(data_root_dir + "train/train.npy" )
    print "movie  loaded"
    xp = cuda.cupy if args.gpu >= 0 else np
    model = L.Classifier( PredNet3Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    model.compute_accuracy = False


    # load model
    #model = pickle.load(open("./model.pkl"))

    if args.gpu >= 0:
        print "get device"
        cuda.get_device(args.gpu).use()

        xp = cuda.cupy
        print "to_gpu"
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        print('Running on a GPU')
    else:
        print('Running on a CPU')

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_movie = train_data[:,:-1]
    train_teacher = train_data[:,1:]

    print "data size(s)", train_movie.shape , train_teacher.shape

    # learning loop
    for epoch in range(args.epoch):
        print "epoch (" , epoch , ") start "
      
        # model save
        if epoch % args.save_step == 0 :
            serializers.save_npz("3Layer_" + str(epoch) + "_" + "model.npz" , model)
            print "model saved"

        acc_loss = 0

        # movie loop
        for movie, teacher in zip(train_movie,train_teacher):
            model.predictor.reset_state()
            loss = 0
            # frame loop 
            for frame in range(len(movie)):
                #t = F.expand_dims(chainer.Variable(xp.array(train_teacher[i])) , axis = 0 )
                x = F.expand_dims(chainer.Variable(xp.array(movie[frame])) , axis = 0 )
                t = F.expand_dims(chainer.Variable(xp.array(teacher[frame])) , axis = 0 )
                loss += model(x, t)

                #learning technique for LSTM/RNN
                if frame % args.unchain_step == 0:
                    model.zerograds()
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()

                    acc_loss += loss.data
                    loss = 0

        print "acc_loss : " , acc_loss

    serializers.save_npz("3Layer_" + str(epoch) + "_" + "model.npz" , model)


if __name__ == "__main__":
     
    print args    

    start = time.time()
    learn()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

    # load example
    #model = L.Classifier( PredNet3Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    #model.compute_accuracy = False
    #serializers.load_npz("3Layer_0_model.npz",model)
    #print "load success"
    #exit()


