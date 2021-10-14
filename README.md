# VAS Regression

<!-- ABOUT THE PROJECT -->
## About The Project

The goal of the project is to predict the VAS value based on landmarks extracted from videos of people that experience different level of pain.
The Visual Analogue Scale (VAS) is a psychometric response scale which is usually used questionnaires. It is a measurement instrument for subjective characteristics or attitudes that cannot be directly measured.
VAS is the most common pain scale for quantification of endometriosis-related pain and skin graft donor site-related pain.

For achieve our goal we need complete the following tasks:
* Reimplementing the model described by in [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://arxiv.org/abs/2010.02803).
* Create a dataset starting from landmarks sequences extracted from videos.
* Do some Preprocessing to the data for make them fit our model.
* Train and test our regression model.


<!-- GETTING STARTED -->
## Getting Started

The project is all implemented using Google Colab. 
For using that is just needed to download all the notebook and the file  ./notebook/2d_skeletal_data_unbc_coords.zip.

### Dataset creation

First of you need to create a structured dataset starting from the raw data. 
You can do this operation by executing the notebook called 'dataset_generation.ipynb'.
In this notebook you can select what landmarks take in to account for the dataset creation.
All is needed to do is just set the correct paths of the input raw data and the output file.
As output you will get a .csv file.
The main columns of the output are:
* 'Sequenza': it represent the ID of the sequence.
* 'Frame': it represent the ID of a frame inside a sequence.
* Columns that represent the features: it can be of variable length.
* 'Label: it represent the VAS value associated to each sequence. The value is the same for each item of the same sequence.

A row in the .csv is something like ['Sequenza','Frame','Feature 1',...'Feature N','Label']

### Preprocessing

Then we need to preprocess the dataset for make it fit our model. 
In particular we need to have the sequence all of the same length. For achieve this we used applied Oversampling and Undersampling to the sequences of the dataset. We also applied some normalization techniques.
In this case it's just needed to set the correct paths at the start of the 'preprocessing.ipynb' notebook.
It need as input two .csv file that one represent the train set and one the test set.
As output it will give two files, one for the test set and one for the train set.

### Training

The third phase is define our model and train it.
For doing this task you can ran the 'train.ipynb' notebook. 
It will perform a subdivision of the data in validation and train sets, and then perform the model training. To use this file it's just needed as the notebook described before to set the correct file paths. In this case you need to set the input file path and the output file paths for the Tensorboard visualization and the model saving.
In particular we save in './logs/gradiente_tape' a folder identified by the time when the code is executed that contains a ./train and ./validation folder for the summary of each phase.

The data that we save for Tensorboard visualization in regard of the training:
* Train Loss: it's used the Mean Squared Error.
* Train Error: it's used the Mean Absolute Error.
* Validation Loss: it's used the MeanSquaredError.
* Validation Error: it's used the MeanAbsoluteError.
* Train Predictions Histogram: that represent the distribution of the predicted label at each epoch.
* Train Ground Truth Histogram: that represent the distribution of the GT label at each epoch.
* Valid Predictions Histogram: that represent the distribution of the predicted label at each epoch.
* Valid Ground Truth Histogram: that represent the distribution of the GT label at each epoch.
* Train Confusion Matrix: that represent the Confusion Matrix at each epoch.

Additional data regarding the model that are saved for Tensorboard as text are:
* All the model Hyperparameters.
* The model summary.

An example of model data.


![Model Summary][model-summary]


### Testing

The last phase is to evaluate the model on the test set. The notebook for doing this is 'test.ipynb'.
For run this code you need to set the train set path, the path to the model that need to be loaded and the log path correctly. We suggest to use the same name used during the training.

The data we saved for the Tensorboard visualization are:
* Test Error: it's used the Mean Absolute Error.
* Test Predictions Histogram: that represent the distribution of the predicted label.
* Test Ground Truth Histogram: that represent the distribution of the GT label.
* Test Confusion Matrix: that represent the Confusion Matrix.

### Visualization on Tensorboard

To enable the visualization of the results on Tensorboard you just need to execute the following code on Google Colab:
1.Mount your GDrive with all the logs saved.
  ```
  from google.colab import drive
  drive.mount('/content/gdrive')
  !ls -l '/content/gdrive/My Drive/to_logs_directory/
  ```
2.And then execute Tensorboard.
   ```
   %reload_ext tensorboard
   %tensorboard --logdir '/content/gdrive/My Drive/to_logs_directory/'
   ```
   
As an alternative we loaded a simple notebook that follow our notation for the directory tree that can just be executed changing the paths.




<!-- MARKDOWN LINKS & IMAGES -->
[model-summary]: images/model_summary.jpg
