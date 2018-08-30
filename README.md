# Name_Generator_Application_Tensorflow_CharRNNModel

The project is lstm based name generator application built with tensorflow rnn.

I am using a virtualenv with conda and I have created a new virtual env for tf serving.
The key goal of this project is to understand :

1) Create and Save a Tensorflow Model so that it can be loaded to with Tensorflow serving 
model server and use it in a production scenario. 

2) Serve the model with Tensorflow Serving ModelServer.( Using a Docker )

3) Send requests and get responses ( Try a simple integration with Django Web Application Framework)

#### 1. Creating a LSTM model

`LSTM_Model_test.py` covers a basic model - which can either add, subtract or 
multiply. The model is very primitive in nature, however, its the same basic
fundamental for more complex models.

#### 2. Save the model

`simple_model.py` also covers how to save a model. 

We could wrap this code in an API endpoint written in a Python framework like Flask, Falcon or similar, and voilá we have an API. 
But there are some really good reasons you don’t want to do it that way:

- If your model(s) are complex and run slowly on CPU, you would want to run your models on more accelerated hardware (like GPUs). Your API-microservice(s), on the other hand, usually run fine on CPU and they’re often running in “everything agnostic” Docker containers. In that case you may want to keep those two kinds of services on different hardware.

- If you start messing up your neat Docker images with heavy TensorFlow models, they grow in every possible direction (CPU usage, memory usage, container image size, and so on). You don’t want that.

- Let’s say your service uses multiple models written in different versions of TensorFlow. Using all those TensorFlow versions in your Python API at the same time is going to be a total mess.

- You could of course wrap one model into one API. Then you would have one service per model and you can run different services on different hardware. Perfect! Except, this is what TensorFlow Serving ModelServer is doing for you. So don’t go wrap an API around your Python code (where you’ve probably imported the entire tf library, tf.contrib, opencv, pandas, numpy, …). TensorFlow Serving ModelServer does that for you.

steps : 

1. `model_input` ,`model_output` :  First we have to grab the input and output tensors.

2. `signature_definition` : Create a signature definition from the input and output tensors. The signature definition is what the model builder use in order to save something a model server can load.
We must give inputs and outputs for signature definition and also `method_name`, this is mandatory without this saving doesn't work. There are 3 kinds of methods such as
predict, classfiy and regress methods. If we don't one of these methods then the saving will give an error.

3. `builder` and `builder.save` : Save the model at a specified path where a server can load it from.

#### 3. Serving the model 

` Notes : Make sure you have installed docker before using this, else you may get an error`.

Use the below command to start docker based container for serving your tf ModelServer.

`docker run -it -p 9000:9000 --name simple -v $(pwd)/models/:/models/ epigramai/model-server:light --port=9000 --model_name=simple --model_base_path=/models/simple_model`

- A container is basically an instance of an image. We pass a lot of options and flags here, so I’ll explain what they all do.
- When you do docker run, you run the image epigramai/model-server:light. The default entrypoint for this image is tensorflow_model_server. This means that when you run the container, you also start the model server.
- Because the model is not built into the image (remember, the image is just the model server) we make sure the container can find the model by mounting (-v) the models/ folder to the container.
- The -it option basically tells docker to show you the logs right in the terminal and not run in the background. The name option is just the name of the container, this has nothing to do with TensorFlow or the model.
- Then there’s the -p option, and this one is important. This option tells docker to map its internal port 9000 out to port 9000 of the outside world. The outside world in this case is your computer also known as localhost. If we omitted this option, the model server would serve your model on port 9000 inside the container, but you would not be able to send requests to it from your computer.
- The three last flags are sent all the way to the model server. The port is 9000 (yep, the port we are mapping out to your machine). With the model_name flag we give our model a name. And with the last flag we tell the model server where the model is located. Again, the model is not in the image, but because we used the -v option and mounted the folder to the container, the model server can find the model inside the running container.

**The options are for Docker (the ones with a single -) and the flags (port, model_name, model_base_path) are parameters to the model server.**

Docker commands for use : 

- Try `docker ps -a` to check running containers. 
- Try `docker stop simple && docker rm simple` for closing and removing a container. 
- Try `docker rm Container ID` for removing a existing container.

We can use our virtualenv terminal to run the server. If you are using pycharm or conda env in any other IDE, The Terminal  will start a new TF Serving ServerModel.
So what we achieve at the end of this is start a new container that serves our previously saved model

Now, how do we use this saved model from our container where we have a TF Model server.

So, the model server is not a regular HTTP endpoint where we can sent POST, GET Requests.
Its actually a gRPC service. A gRPC enables us to directly call methods on a server rather than endpoints such as `getFriend()` instead of `\getFriend`
The gRPC client side has a stub that provides all the methods the server has.

The client works without TensorFlow installed. 

`Note : (I tried using this on a virtual env but it fails, it looks like this is something to do with conda.)`

This method makes it easier to integrate our model with other apps. 
All we need to do is give the app our client and show them how to use it. Basically, give them hostname, model name and model version name amd it should be done.

Try : `pip install git+ssh://git@github.com/epigramai/tfserving-python-predict-client.git`.

If it doesn't work then try `pip install git+https://github.com/epigramai/tfserving-python-predict-client.git`, for some people ssh doesn't work. 

Once completed, make sure your server is running and call the client and test it.
Look at `Client.py` file that test various different models from `models` folder. 

I’ll go through what’s happening here in detail:

- The first line imports the client. ProdClient is the client you would use to send requests to an actual model server. (Check pip freeze to see predict client)

- Pass the host, model name and version to the client’s constructor. Now the client knows which server it’s supposed to communicate with.

- To do a prediction we need to send some request data. As far as the client knows, 
the server on localhost:9000 could host any model. So we need to tell the client what our input tensor(s) are called, 
their data types and send data of the correct type and shape (exactly what you do when you pass a feed_dict to sess.run). Now, in_tensor_name ‘inputs’ is 
the same ‘inputs’ that we used in the signature definition in part 1. The input tensor’s data type must also match the one of the placeholder.
Note that we use the string ‘DT_INT32’ (and not tf.int32).

- Finally call client.predict and you should get a response in the form of a dictionary.

Once this is done we can integrate this response with another application using Django web application framework.

Note : added `grpcio` to conda using `conda install grpcio`.

Model Training : 

Output : 
`WARNING:tensorflow:From /Users/davidjoy/Desktop/pycharm/Char_rnn_project/LSTM_Model_test.py:71: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See @{tf.nn.softmax_cross_entropy_with_logits_v2}.

2018-08-29 15:43:01.601269: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Iter= 2001, Average Loss= 0.796811, Average Accuracy= 75.29%
Iter= 4001, Average Loss= 0.236399, Average Accuracy= 90.90%
Iter= 6001, Average Loss= 0.222687, Average Accuracy= 91.27%
Iter= 8001, Average Loss= 0.217329, Average Accuracy= 91.45%
Iter= 10001, Average Loss= 0.214098, Average Accuracy= 91.51%
Iter= 12001, Average Loss= 0.212658, Average Accuracy= 91.51%
Iter= 14001, Average Loss= 0.211830, Average Accuracy= 91.54%
Iter= 16001, Average Loss= 0.211341, Average Accuracy= 91.56%
Iter= 18001, Average Loss= 0.210890, Average Accuracy= 91.60%
Iter= 20001, Average Loss= 0.210099, Average Accuracy= 91.58%

Process finished with exit code 0`

Sources: 

https://medium.com/epigramai/tensorflow-serving-101-pt-1-a79726f7c103

https://medium.com/epigramai/tensorflow-serving-101-pt-2-682eaf7469e7

