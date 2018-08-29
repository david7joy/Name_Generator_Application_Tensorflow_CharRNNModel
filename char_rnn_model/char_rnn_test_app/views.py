from django.shortcuts import render
from tensorflow_serving.apis import predict_pb2,prediction_service_pb2_grpc
from tensorflow.contrib.util import make_tensor_proto
import tensorflow as tf
import numpy as np
import grpc

def result(name):
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'simple'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    char_to_ix = {' ': 0,'a': 1,'b': 2,'c': 3,'d': 4,'e': 5,'f': 6,'g': 7,'h': 8,'i': 9,'j': 10,'k': 11,'l': 12,'m': 13,
                 'n': 14,'o': 15,'p': 16,'q': 17,'r': 18,'s': 19,'t': 20,'u': 21,'v': 22,'w': 23,'x': 24,'y': 25,'z': 26 }

    ix_to_char = {0: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k',
                  12: 'l', 13: 'm', 14: 'n', 15: 'o',
                  16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

    srch = name
    # enter = input('Enter 3 letters :')
    letters = srch.strip()
    letters = list(letters)
    letters = [char_to_ix[i] for i in letters]


    for i in range(20):
        fixing_shape = np.float32(np.zeros((512, 3, 1)))
        cell_input = np.reshape(np.array(letters), [-1, 3, 1])
        x = np.float32(cell_input + fixing_shape)
        request.inputs['x_input'].CopyFrom(
            make_tensor_proto(x))
        result_future = stub.Predict(request, timeout=10)
        values = np.reshape(result_future.outputs['y_output'].float_val, (512, 27))
        values = np.reshape(values[1], (1, 27))
        location = int(np.argmax(values, axis=1))
        letters.append(location)
        srch += '%s' % (ix_to_char[location])
        letters = letters[-3:]

    return srch

def test(request):
    if request.method=='POST':
        srch = request.POST['srh']
        value = result(srch)

        return render(request, 'char_rnn_test_app/test.html', {'value': value})

    return render(request, 'char_rnn_test_app/test.html')

# Create your views here.
