# -*- coding:utf-8 -*-
import argparse
import os
import cv2 as cv
import numpy as np
from PIL import Image

import pickle
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter


def get_movie_filename(path,directory , extention="mp4"):
    """
    get file_name which has specified extention (ex. mp4 )

    :param path: relative path
    :param extention:
    :return: file_names(full path)
    """

    file_names = list(filter(lambda file_name: extention in file_name, os.listdir(path + "/" + directory )))
    file_names = map(lambda file_name: os.getcwd()  + path[2:] + "/" + directory + "/" + file_name, file_names)

    return file_names


def stack(source, dist, module=np):
    """

    :param source: source stacked nparray(or Variable)
    :param dist:
    :param module:
    :return: stacked nparray
    """

    if source is None:
        source = dist
    else:
        source = module.vstack((source, dist))

    return source


def get_movies(full_paths, frame_count=600, size=(160, 128), dtype=np.float32):
    """

    :param full_paths:
    :param frame_count:
    :return: nparray  , shape = (movie_file , frame_count , RGB , width , height)
    """

    movie_batches = None

    for file_name in full_paths:
	print "file_name " , file_name 
        movie = cv.VideoCapture(file_name)
        movie_stack = None

        for frame in range(frame_count):
            try:
                ret, image = movie.read()
                image = cv.resize(image, size)
                image = np.asarray(image, dtype=dtype)
                image = image.transpose(2, 0, 1) / 255
                movie_frame = np.expand_dims(image, axis=0)

                movie_stack = stack(movie_stack, movie_frame)

            except:
                print("something wrong (in get_movie)")
                exit(0)

        movie_stack = np.expand_dims(movie_stack, axis=0)
        movie_batches = stack(movie_batches, movie_stack)

    # ( length , movie , RGB , height , width )
    # movie_batches = movie_batches.transpose(1, 0, 2, 3, 4).astype(np.float32)

    # ( movie , length , RGB , height , width )
    movie_batches = movie_batches.transpose(0, 1, 2, 3, 4).astype(np.float32)

    return movie_batches


def make_movie(nparray_movie, file_name, fps):
    """
    :param nparray: shape = (length ,3,width , height)
    :return:
    """

    # fps = movie.get(cv.CAP_PROP_FPS)
    width = nparray_movie.shape[3]
    height = nparray_movie.shape[2]
    #fourcc = cv.VideoWriter_fourcc('P', 'I', 'M', '1')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    vout = cv.VideoWriter()
    success = vout.open(file_name, fourcc, fps, (width, height), True)
    for _image in nparray_movie:
        image = _image * 255
        pil_image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8)[:, :, ::-1].copy())
        cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        vout.write(cv.resize(cv_image, (width, height)))
    vout.release()
    del vout
    print file_name , "made" 


def make_teacher_signal(nparray_movies):
    """
    :param nparray_movie: ( length , movie , RGB , height , width )
    :return: TupleDataset ( t_frame ,  t+1_frame)
    """

    input_movie = nparray_movies[:-1]
    teacher_movie = nparray_movies[1:]
    dataset = chainer.datasets.TupleDataset(input_movie, teacher_movie)

    return dataset


# (batch , len , RGB , height , width )
# (len , batch , RGB , height , width ) -> こっちの方が妥当っぽい

# numpy を綺麗に画像化する
# image_RGB = Image.fromarray(_image.transpose(1,2,0).astype(np.uint8)[: , : , ::-1].copy())


if __name__ == "__main__":
    # DO CONVERT MP4 TO NPY
    parser = argparse.ArgumentParser(description='movie pre-process.')
    parser.add_argument('--directory', '-d', default="train", type=str, help="targt directory")
    args = parser.parse_args()
    root_path = "./workspace/movies"
  
    # in docker
    movie_name_list = get_movie_filename(path=root_path,directory=args.directory)
    print movie_name_list
    movies = get_movies(full_paths = movie_name_list,)
    print movies.shape
    np.save(root_path + "/" + args.directory + "/train.npy",movies)
    
    




