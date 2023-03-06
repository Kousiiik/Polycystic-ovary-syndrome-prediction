# Inspiration
Polycystic ovary syndrome is a health problem caused by an imbalance of reproductive hormones that often is misdiagnosed.
My Web application has a deployed Machine learning model which helps doctors diagnose whether a person has PCOS based on the input on hormones entered with 90.79 % accuracy. The website also provides details regarding the syndrome which include all the common symptoms, causes, and how to diagnose it.

## ML-Model-Deployment
This is a PCOS Predictive Analysis Engine which can be deployed at any cloud engine 
The code and impimentation is done by Kousik Katari
### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has three major parts :
1. model.py - This contains code for our Machine Learning model to predict based on training data.
2. app.py - This contains Flask APIs that receives details through GUI or API calls, computes the precited value based on our model and returns it.
3. templates - This folder contains the HTML template.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :

Enter valid numerical values in all  input boxes and hit Predict.

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```



