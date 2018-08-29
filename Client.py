from tensorflow_serving.apis import predict_pb2, prediction_service_pb2,prediction_service_pb2_grpc
from tensorflow.contrib.util import make_tensor_proto
import tensorflow as tf
import numpy as np
import grpc

channel = grpc.insecure_channel('localhost:9000')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'simple'
request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

fname = 'dinos.txt'
with open(fname) as f:
    training_data = f.read()
    training_data = training_data.replace('\n', ' ')
    training_data = training_data.lower()

chars = list(set(training_data))

data_size, vocab_size = len(training_data), len(chars)
print("The size of the data is %d and vocab size %d" % (data_size, vocab_size))

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

enter = input('Enter 3 letters :')
letters = enter.strip()
letters = list(letters)
letters = [char_to_ix[i] for i in letters]

for i in range(20):
    fixing_shape = np.float32(np.zeros((512,3,1)))
    cell_input = np.reshape(np.array(letters), [-1, 3, 1])
    x = np.float32(cell_input+fixing_shape)
    request.inputs['x_input'].CopyFrom(
            make_tensor_proto(x))
    result_future = stub.Predict(request,timeout=10)
    values = np.reshape(result_future.outputs['y_output'].float_val,(512,27))
    values = np.reshape(values[1],(1,27))
    location = int(np.argmax(values,axis=1))
    letters.append(location)
    enter += '%s'%(ix_to_char[location])
    letters = letters[-3:]

print({'outputs':enter})