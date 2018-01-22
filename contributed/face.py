# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# import pickle
import os

# import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet


gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/../model_checkpoints/20170512-110547"
# classifier_model = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_1.pkl"
# debug = False

# should I add an attribute for filename? might help with identification...
class Face:
    def __init__(self):
        # self.name = None
        self.bounding_box = None
        self.image = None
        # self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detector()
        self.encoder = Encoder()
        self.identifier = Identifier()
    
    # adds identity to Identifier
    def add_identity(self, image, threshold=1):
        faces = self.detect.find_faces(image)

        assert len(faces) == 1, "baseline image should contain exactly one face"
        
        face = faces[0]
        # face.name = person_name
        face.embedding = self.encoder.generate_embedding(face)[0]
        self.identifier.face = face
        self.identifier.threshold = threshold
        
        return None
    
    # checks if identity is in an image or set of images
    def identify(self, images):
        # check if identity initialized
        self.identifier.has_identity()
        
        # single image passed
        if type(images) is not list:
            images = [images]
            
        # detect faces and compute embeddings 
        detected = [self.detect.find_faces(image) for image in images]
        nfaces = [len(sublist) for sublist in detected]
        face_ind = [i for i,length in enumerate(nfaces) if length > 0]
        # unpacks list of lists
        faces = [face for sublist in detected for face in sublist]        
        embeddings = self.encoder.generate_embedding(faces)

        # identify embeddings
        identities = np.zeros(len(images),dtype=bool)
        recognized = np.split(self.identifier.identify(embeddings),
                              np.cumsum([nface for nface in nfaces if nface > 0])[:-1])
        identities[face_ind] = [np.any(item) for item in recognized]
        
        return identities

# class Identifier:
#     def __init__(self):
#         with open(classifier_model, 'rb') as infile:
#             self.model, self.class_names = pickle.load(infile)
# 
#     def identify(self, face):
#         if face.embedding is not None:
#             predictions = self.model.predict_proba([face.embedding])
#             best_class_indices = np.argmax(predictions, axis=1)
#             return self.class_names[best_class_indices[0]]

class Identifier:
    def __init__(self):
        self.face = None
        self.threshold = None
    
    # check if identity has been initialized    
    def has_identity(self):
        assert self.face is not None, "identity not initialized"
        
    def identify(self, embeddings):
        # check if initialized
        self.has_identity()
        
        # get embedding of identity       
        identity = self.face.embedding
        
        # euclidean distances between embeddings    
        ips = np.dot(embeddings,identity)
        sc_threshold = 1 - (self.threshold ** 2) / 2.
        
        # face matches if inner product greater than scaled threshold                            
        return ips > sc_threshold
        


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, faces):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
        if not np.iterable(faces):
            faces = [faces]
            
        # split into batches
        batch_size = 100
        nfaces = len(faces)
        nbatches = (nfaces-1) // batch_size + 1
        
        prewhiten_faces = [facenet.prewhiten(face.image) for face in faces]
        # initialize output
        results = np.zeros((nfaces,embeddings.get_shape()[1]))

        # Run forward pass to calculate embeddings
        for i in range(nbatches):
            inds = range(i*batch_size,min((i+1)*batch_size,nfaces))
            data = [prewhiten_faces[ind] for ind in inds]
                
            feed_dict = {images_placeholder: data, phase_train_placeholder: False}
            results[inds,...] = self.sess.run(embeddings, feed_dict=feed_dict)
            
        return results


class Detector:
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_minsize=80,face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_minsize = face_minsize
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.face_minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            # face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)
            
            # scale margin to be proportional to face width
            face_width = bb[2] - bb[0]
            scale = face_width / (self.face_crop_size - self.face_crop_margin)
            scaled_face_crop_margin = 2 * (scale * self.face_crop_margin // 2)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - scaled_face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - scaled_face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + scaled_face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + scaled_face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces
