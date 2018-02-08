# Scene recognition in your browser

This sample is based on [Places205](http://places.csail.mit.edu/downloadCNN.html) neural network.

## Prepare a model
* Download a ready model from [Google Drive](https://drive.google.com/open?id=1BpnMdMeoDrY-oBFoPyWFxYMcHZKkxUWP)
or follow the next steps.

* Download [Places205-AlexNet](http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz)
model and scene attributes detector from [here](http://places.csail.mit.edu/model/sceneattributepredictor.zip).

* Run `modify_places205.py` script. It produces a single `.caffemodel` file that
is merged with the places classification network and scene attributes classificator.
You need to have a [Caffe framework](http://caffe.berkeleyvision.org/) installed.
