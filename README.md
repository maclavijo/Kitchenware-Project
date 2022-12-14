# Kitchenware Classification Project

## Problem description

The objective of this project is to classify images from kitchenware set.  There are cups, glasses, knifes, spoons, forks and plates. There are 5 different ML models and the model with best performance will be chosen.

The original data can be found in kaggle.com/competitions/kitchenware-classification/data. The image folder and required files have been already added to the GitHub repository https://github.com/maclavijo/Kitchenware-Kaggle.

## File structure
```
├─── .ipynb_checkpoints
├─── images
├─── Models
├─── test
└─── train
     ├───cup
     ├───fork
     ├───glass
     ├───knife
     ├───plate
     ├───spoon
     └── ...
```

### Dependency and enviroment management

Pipfile and Pipfile.lock files are provided. Copy the content of this folder to your machine. Then from the terminal of your IDE of preference (in the correct work directory) the following:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pipenv install<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pipev shell

Now you will be in the virtual environment and will be able to run the files locally<br>

### To run the project locally from your machine

From your console (in the correct work directory) and after the environment has been created (previous step):<br>

You can run the train and predict files from here<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python train.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python Predict.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or the jupyter notebook from anaconda (from anaconda prompt):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;jupyter notebook
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;then find and open the file Kitchenware.ipynb<br>
<br>
or you can run the streamlit app<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;streamlit run diabetes_app.py

You can now view your Streamlit app in your browser.
Local [http://localhost:8501](http://localhost:8501) or Network [http://10.97.0.6:8501](http://10.97.0.6:8501)


### Containerization

Dockerfile has been provided. To create and run the image, from your IDE terminal do the following (within the work directory):

1. First option: Create and run the app yourself.<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Create image:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docker build -t diabetes_app_streamlit .<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Run image:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docker run -p 8501:8501 diabetes_app_streamlit<br>

You can now access the Streamlit app in your web browser: Local URL: [http://localhost:8501](http://localhost:8501) or from URL: [http://0.0.0.0:8501](http://0.0.0.0:8501)<br>

2. Second option: To run it using docker hub repository:<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Download image from hub run command:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docker pull supermac789/diabetes_app_streamlit:latest<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Run the command from your terminal:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;docker run -p 8501:8501 supermac789/diabetes_app_streamlit:latest<br>

You can now access the Streamlit app in your web browser: Local URL: [http://localhost:8501](http://localhost:8501) or from URL: [http://0.0.0.0:8501](http://0.0.0.0:8501)<br>

### Cloud deployment - Streamlit cloud

The app can be found and run from [https://maclavijo-diabetespredictionproject-diabetes-app-7rf1nc.streamlit.app/](https://maclavijo-diabetespredictionproject-diabetes-app-7rf1nc.streamlit.app/).