# coding:utf-8
import argparse
import gym
import numpy as np
import pickle
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
from models.PredNet import PredNet
from chainer import serializers

import time

import time



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default='10000', type=int, help='learning epoch')
parser.add_argument('--unchain_step', '-u', default='100', type=int, help='unchain freq')
parser.add_argument('--save_step', '-s', default='500', type=int, help='model save freq')
args = parser.parse_args()

class PredNet1Layer(chainer.Chain):

    def __init__(self , width , height , channels, batchSize):
        super(PredNet1Layer , self).__init__()
        self.add_link("l1",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))


    def __call__(self, x):

        h = self.l1(x)

        return h

    def reset_state(self):
        self.l1.reset_state()



def learn(data_root_dir="./movies/"):
    """
    movie.pickleを用いて，PredNetの学習を行う
    Returns:

    """

    train_data = np.load(data_root_dir + "train/train.npy" )
    print train_data.shape
    print "movie  loaded"
    xp = cuda.cupy if args.gpu >= 0 else np
    model = L.Classifier( PredNet1Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
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

    print "data size(s)"
    print train_movie.shape , train_teacher.shape

    for epoch in range(args.epoch):
        print "epoch (" , epoch , ") start "
      
        # model save
        if epoch % args.save_step == 0 :
            serializers.save_npz("1Layer_" + str(epoch) + "_" + "model.npz" , model)
            print "model saved"

        acc_loss = 0
        model.predictor.reset_state()

        # movie loop
        for movie, teacher in zip(train_movie,train_teacher):
            #x = F.expand_dims(chainer.Variable(xp.array(train_movie[0])) ,axis = 0)

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

                    acc_loss += loss.data
                    loss = 0
                    optimizer.update()

        print "acc_loss : " , acc_loss

    # movie generate
    #test_data = pickle.load(open("./movies/" + movie_type  + "_movie.pkl","r"))

    #movie_len = 60
    #size = (160,128)
    #movie = np.zeros((movie_len,3 ,size[0],size[1]),dtype=np.float32)
    #for i, seq in enumerate(range(movie_len)):
    #    # 現在の環境では，5ステップ前後で1周
    #    _ = model(seq, seq)
    #    image = np.asarray( model.y.data)
    #    movie[i] = image



    #pickle.dump(model,open("1Layer_Test_" + str(epoch) + "_" + "movie.pkl" , "wb") , -1)

    serializers.save_npz("1Layer_" + str(epoch) + "_" + "model.npz" , model)

def make_error_movie(model_file="./model.pkl", movie_len = 60,input_movie="./movies/normal_movie.pkl",out_name="output_movie.mp4"):
    """
    既存のモデルを読み込み，ある環境でのエラーを動画化する

    Args:
        model: pickle file

    Returns:

    """
    size = (160,128)

    data = pickle.load(open(input_movie,"rb"))
    model = pickle.load(open(model_file,"rb"))

    movie = np.zeros((movie_len,3 ,size[0],size[1]),dtype=np.float32)

    for i, seq in enumerate(range(movie_len)):
        # 現在の環境では，5ステップ前後で1周
        _ = model(seq, seq)
        image = model.y.data
        movie[i] = image
    # TODO make numpy array	
    #make_movie(movie,file_name=out_name,fps=30)





if __name__ == "__main__":
  
    

    start = time.time()
    learn()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

    # load example
    #model = L.Classifier( PredNet1Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    #model.compute_accuracy = False
    #serializers.load_npz("1Layer_0_model.npz",model)
    #print "load success"


