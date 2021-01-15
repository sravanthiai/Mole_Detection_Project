# Mole_Detector_API
Convolutional Neural Network image classifier using Keras and tensorflow backed, deployed on Heroku.

# Introduction
The purpose of the project is to develop a tool that would be able to detect moles that need to be handle by doctors.The project will be available on a simple web page where the user could upload a picture of the mole and see the result.

# The Team
The team comprises of Manasa Devi Noolu,Davy Nimbona,Opaps Ditudidi and Sravanthi Tarani.

# Project Organization
| Content	| Description |
|-------- | -------------|
| Task 1	| Preparation Dataset |
| Task 2	| Importing Dataset |
| Task 3	| Creating and Saving a Model |
| Task 4	| Creating a Flask Application |
| Task 5	| Deployment and Creating a Docker File |

# Built With

* Python
* Numpy
* Pandas
* Scikit-learn
* TensorFlow
* OpenCV
* Docker
* Heroku

# Prerequisites
You'll need the packages/software described above.

# Installation

  ## HEROKU
   ### Install the Heroku CLI:
   * The Heroku Command Line Interface (CLI) makes it easy to create and manage your Heroku apps directly from the terminal.       It’s an essential part of using Heroku.
      ```
     'sudo snap install --classic heroku'
      ```
   * Deployment on Heroku:
      * Heroku favours Heroku CLI therefore using command line is (ensure the CLI is up-to-date) crucial at this step.
      ```
      heroku login
      ```
      * After logging in to the respective Heroku account, the container needs to be registered with Heroku using
      ```
      heroku container:login
      ```
      * Once the container has been registered, a Heroku repo would be required to push the container which could be created :
      ```
      heroku create <yourapplicationname>
      ```
      **NOTE** : If there is no name stated after 'create', a random name will be assigned.
      * When there is an application repo to push the container, it is time to push the container to web :
      ```
      heroku container:push web --app <yourapplicationname>
      ```
      
      * Following the 'container:push' , the container should be released on web to be visible with
      ```
      heroku container:release web --app <yourapplicationname>
      ```
      * If the container has been released properly, it is available to see using
      ```
      heroku open --app <yourapplicationname>
      ```
      * Logging is also critical especially if the application is experiencing errors :
      ```
      heroku logs --tail <yourapplicationname>
      ```
      **IMPORTANT NOTE**: While with localhost and Docker it is not mandatory to specify the PORT, if one would like to deploy on Heroku, the port needs to be specified within the 'app.py' to avoid crashes.
      

# The API

API recieves an image, and returns a response 'you are in danger' or 'you're not in danger'.

| Problem |	Data	| Methods |	Libs | Link |
|---------|-------|---------|------|------|
|Deployment|	Image input|GET, POST	|```flask```|(https://github.com/manasanoolu7/Mole_Detection_Project/blob/main/app.py) |

* **Url**:

https://mole-detection.herokuapp.com/


* **Success Response:**

# Deployement



 

