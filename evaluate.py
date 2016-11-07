# coding:utf-8
import argparse
import numpy as np
import pickle
import chainer
from chainer import functions as F
from Lis_pred1 import PredNet1Layer
from Lis_pred2 import PredNet2Layer
from Lis_pred3 import PredNet3Layer

from chainer import links as L
from chainer import serializers
from chainer import cuda
from models.PredNet import PredNet

import time
import os

from preprocess import get_movies
from preprocess import get_movie_filename
from preprocess import make_movie
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model',  type=str ,default="1Layer_20_model.npz")
parser.add_argument('--len',  type=int , default=600)
parser.add_argument('--out_name',  type=str , default="/workspace/output_movie.avi") # setting for docker run
parser.add_argument('--data_root_path',  type=str , default="./movies/test/")
args = parser.parse_args()



def make_predict_movie(model_file_name,movie_len = 600,out_name="output_movie.mp4"):
    """
    Args:
        input_movie : mp4
        model: pickle file

    Returns:

    """

    movie_name_list = get_movie_filename(path="./workspace/movies",directory="test")
    size = (160,128)
    movies = get_movies(movie_name_list, frame_count=args.len, size=size)

    model = L.Classifier( PredNet1Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    #model = L.Classifier( PredNet2Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    #model = L.Classifier( PredNet3Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    serializers.load_npz("/workspace/models/" + model_file_name ,model)


    if args.gpu > 0 :
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        print('Running on a GPU')
    else:
        print('Running on a CPU')

    predict_movie = np.zeros((movie_len - 1, 3 ,size[1],size[0]),dtype=np.float32)
    for i in range(len(movies)):
        make_movie(movies[i],file_name="./workspace/movie_"+ str(i) + "_real.avi",fps=30)

    for i in range(len(movies)):
        model.predictor.reset_state()
        x = movies[i]
        teacher = movies[i][1:]

        generated_movie = np.zeros((len(teacher),3,size[1] , size[0]),dtype=np.float32)
        for frame, teacher in enumerate(teacher):
            seq = F.expand_dims(x[frame] , axis=0)
            t = F.expand_dims(teacher , axis=0)
            loss = model(seq, t)
            image = model.y.data
            generated_movie[frame] = image
        model_name = args.model.split(".")[0]
        make_movie(generated_movie,file_name="./workspace/"+ model_name +"_predicted_movie_"+ str(i)+".avi",fps=30)





if __name__ == "__main__":
    make_predict_movie(args.model ,args.len , args.out_name)

