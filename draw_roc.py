# coding:utf-8
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter
from chainer import serializers

from preprocess import get_movie_filename
from preprocess import get_movies
from Lis_pred1 import PredNet1Layer
from Lis_pred2 import PredNet2Layer
from Lis_pred3 import PredNet3Layer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--len',  type=int , default=600)
args = parser.parse_args()



file_names = sorted(get_movie_filename(path="./workspace/movies", directory = "test"))
anomaly_files = ["3-a-1.mp4","3-a-2.mp4","3-a-3.mp4","3-a-4.mp4","3-a-5.mp4"] # 3-a-1 is 0 


# prepare data
error_csv = open("./workspace/movies/test/error.csv","r")
labels  = []
for line in error_csv:
    line = line.replace("\r\n","").split(",")[1:]
    line = map(lambda x : 0 if x == "" else 1, line)
    labels.append(line)

    
labels = np.array(labels)
labels = labels.T
labels = np.r_[labels , np.zeros((labels.shape))] # anomaly * 5 , normal * 5


# load classifier
model = L.Classifier( PredNet1Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
model.compute_accuracy = False
serializers.load_npz("./workspace/models/1Layer_99_model.npz",model)

xp = chainer.cuda.cupy if args.gpu >= 0 else np

if args.gpu >= 0 :
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    model.predictor.to_gpu()
    print('Running on a GPU')
else:
    print('Running on a CPU')



whole_loss = []
whole_label = []

# 初回実行時のみ
#size = (160,128)
#movies = get_movies(file_names, frame_count=600, size=size)
#np.save("./workspace/movies/test/test.npy", movies)

# 次回以降
movies = np.load("./workspace/movies/test/test.npy")
#print movies.shape

 
for movie_id , movie_name in enumerate(sorted(file_names)):
    print movie_id , " start "
    model.predictor.reset_state()
    movie = movies[movie_id][:-1]
    teacher = movies[movie_id][1:]

    # label select
    if movie_name.split("/")[-1] in anomaly_files:
        label = labels[anomaly_files.index(movie_name.split("/")[-1])]
    else:
        label = np.zeros((598,))

    for t in range(len(label)) :
        loss = model(F.expand_dims(chainer.Variable(xp.asarray(movie[t]),volatile=True) , axis = 0), F.expand_dims(chainer.Variable(xp.asarray(teacher[t]),volatile=True) ,axis=0 ))
        loss = chainer.cuda.to_cpu(loss.data)
        whole_loss.append(loss)
        whole_label.append(label[t])


# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(whole_label, whole_loss )
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc="lower right")
pl.savefig("/workspace/roc.png" , format = "png")
