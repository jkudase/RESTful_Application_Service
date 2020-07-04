# Objective

Online Application Service that accepts Human Pose Skeletal key points of a sign video and return the label of the sign as a JSON Response. The Key points are generated using TensorFlow’s Pose Net.

•	For generating the key points, please follow instructions in the link -> https://github.com/prashanthnetizen/posenet_nodejs_setup

In this project, our group has developed an online RESTful application Service that can classify American 
Sign Language. We have trained 4 models that helps classify the input provided in form of JSON data into
it's corresponding sign language.

#Models
The 4 models developed using the training data are as follows:

model 1: KNeighborsClassifier

model 2: RandomForestClassifier

model 3: DecisionTreeClassifier

model 4: ExtraTreesClassifier

#Training Method:

All the data provided by Professor and our own video dataset was used to train the above four models. 
The complete data that can be used to train these models and generate PKL files can be obtained at below
google drive link:
https://drive.google.com/file/d/177GSIxh67l6UOoYOImgGpHVJ-lybInfC/view?usp=sharing

The four pkl files could be generated once you place the above CSV and the modelTraining.py in the same 
folder and execute the modelTraining.py script. Methods like K-Fold Cross Validation were used to determine
which classifiers would work the best for the given data. Accuracy was also measured using test_size split of 33%.
Feel free to uncomment the functions and the model in order to check the validation accuracies for each of the model.

#Deployment:

The above models have been deployed to the Amazon EC2 server. The serverApp.py script is run using flask on this EC2 
instance.

#Testing

The url that can be used to test the script and obtain labels is as follows: 
http://54.189.183.128:5000/getLabels

Using CURL:

Please send the body (data) of JSON file as the input, since the server is designed to handle this type of input data.
For e.g. the curl request that can be used from the linux terminal is: 
curl -d "@test.json" -H "Content-Type: application/json" -X POST http://54.189.183.128:5000/getLabels

USing Postman:

Put the above EC2 link in the 'POST' Request field.
Paste the json data in the body section of the application. 
Body should be 'raw' and 'json' type.

=======

