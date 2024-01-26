# Deep Learning (Convolutional Neural Networks) for Traffic Sign Recognition
Deep Learning (Convolutional Neural Networks) for Traffic Sign Recognition 

## Problem Description
Recognition of traffic signs is a challenging real-world problem of high industrial relevance. Although commercial systems have reached the market and several studies on this topic have been published, systematic unbiased comparisons of different approaches are missing and comprehensive benchmark datasets are not freely available.

Traffic sign recognition is a multi-class classification problem with unbalanced class frequencies. Traffic signs can provide a wide range of variations between classes in terms of color, shape, and the presence of pictograms or text. However, there exist subsets of classes (e. g., speed limit signs) that are very similar to each other.

The classifier has to cope with large variations in visual appearances due to illumination changes, partial occlusions, rotations, weather conditions, etc.

Humans are capable of recognizing the large variety of existing road signs with close to 100% correctness. This does not only apply to real-world driving, which provides both context and multiple views of a single traffic sign, but also to the recognition from single images.

## Data
The German Traffic Sign Benchmark (GTSRB) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. GTSRB has the following properties:

Single-image, multi-class classification problem

More than 40 classes

More than 50,000 images in total

Large, lifelike database


GTSRB Data is available from multiple resources:

(1) https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

(2) https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html (the original resource)

and

(3) https://huggingface.co/datasets/bazyl/GTSRB


<b> Note: The dataset is too big to upload to the project's GitHub repository. Please use wget to obtain a local copy: </b>

wget www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download/archive.zip

and

wget https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/GTSRB_label_names.csv

followed by 

unzip archive.zip

in the same ("top level") directory where you download/place GTSRB_label_names.csv.

After this, the code should run as provided.

## Script train.py
The script train.py can be used to:
        Train the final model
        Save it to a file in the .h5 format (which is then converted to tflite using convert.py)

## Script convert.py
The script convert.py can be used to:
        Load the model previously saved in .h5 format, convert it tflite format, and save


## Script predict_from_inside_Docker.py
The script predict_from_inside_Docker.py can be used to:
        Load the model (for use within Docker)
        Define the Lambda handler

Similarly, the script predict_outside_Docker.py can be used to:
        Load the model (for use outside Docker, since we then need to use the tflite version included with Tensorflow)
        Define the Lambda handler

You can execute predict_outside_Docker.py to locally test the model on a single sample (traffic sign):

	python predict_outside_Docker.py

The expected output is

{'Predicted': 'Yield'}

## Files with dependencies
        Pipenv and Pipenv.lock
        Dockerfile

## Setting up the virtual environment

Install pipenv, to create a virtual environment:

        pip install pipenv

Install dependencies using the requirements file:

        pipenv install -r requirements.txt

Activate the virtual environment:

        pipenv shell

Now run

        python train.py
to train and save the best model, and

        python predict_from_inside_Docker.py
        python test.py

to test the saved model on 1 sample

## Dockerization

Build the Docker container:

        docker build -t traffic_sign_recognition .

Run the Docker container:

        docker run -it -p 8080:8080 traffic_sign_recognition:latest

As before, to test that the prediction app is running properly via Docker, you can type

        python test.py

to test the saved model on 1 sample (a traffic sign)

and the expected output is

{'Predicted': 'Yield'}

## Deployment

Deployment was carried out using AWS Lambda (using the Docker image created previously), ECR and a REST API

To run the service, go to

The URL is

https://jt48qss34g.execute-api.us-east-1.amazonaws.com/test/predict

which is specified/used in

        test_AWS.py

and once the container is running via AWS, you can run

        python test_AWS.py

to test the saved model on 1 sample (a traffic sign)

and the expected output is (as before)

{'Predicted': 'Yield'}

Here are screenshots of the service created on AWS (Lambda function creation, Lambda function testing, API creation, Local testing using test_AWS.py):

<img src="https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/AWS_Lambda_Function_Created.png" style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">

<img src="https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/AWS_Lambda_Function_Tested.png" style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">

<img src="https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/AWS_API_Gateway_Resource_Tested.png" style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">
<img src="https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/AWS_Lambda_Function_Tested_Locally_using_AWS_URL.png" style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">
and finally, a traffic sign that was used for testing:
<img src="https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00051.png" style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">
