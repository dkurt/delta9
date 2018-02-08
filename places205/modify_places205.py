# This script is used to merge origin Places205-AlexNet model from http://places.csail.mit.edu/downloadCNN.html
# with corresponding scene attributes detection matrix of size 4096x102
# It creates new .caffemodel with one more InnerProduct
import caffe
import scipy.io
import numpy as np

# Read scene attributes detector (4096x102 matrix)
attr = scipy.io.loadmat('sceneattributepredictor/sceneAttributeModel205.mat')

# Load Caffe model
net = caffe.Net('places205_alexnet.prototxt',
                'placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel',
                caffe.TEST)

# Set scene attributes matrix as weights of fully-connected layer
net.params['fc9'][0].data[:,:] = attr['W_sceneAttribute'].astype(np.float32)

# Save model
net.save('places205_alexnet.caffemodel')

# Read mean values (B: 104.929729064, G: 113.254710686 R: 116.209692099)
blob = caffe.proto.caffe_pb2.BlobProto()
with open('placesCNN_upgraded/places205CNN_mean.binaryproto', 'rb') as f:
    blob.ParseFromString(f.read())
    data = np.array(blob.data).reshape([blob.channels, blob.height, blob.width])
    print np.mean(data[0]), np.mean(data[1]), np.mean(data[2])
