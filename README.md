# VAS Regression

<!-- ABOUT THE PROJECT -->
## About The Project

The goal of the project is to predict the VAS value based on landmarks extracted from videos of people that experience different level of pain.
The Visual Analogue Scale (VAS) is a psychometric response scale which is usually used for questionnaires. It is a measurement instrument for subjective characteristics or attitudes that cannot be directly measured.
VAS is the most common pain scale for quantification of endometriosis-related pain and skin graft donor site-related pain.

To achieve our goal we need to complete the following tasks:
* Reimplementing the model described by in [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://arxiv.org/abs/2010.02803).
* Creating a dataset starting from landmarks sequences extracted from videos.
* Doing some Preprocessing to the data to make them fit our model.
* Training and testing our regression model.


<!-- GETTING STARTED -->
## Getting Started

The project is all implemented using Google Colab. 
To do that it is just needed to download all the notebook and the file  ./notebook/2d_skeletal_data_unbc_coords.zip.

### Dataset creation

First of all you need to create a structured dataset starting from the raw data. 
You can do this operation by executing the notebook called 'dataset_generation.ipynb'.
In this notebook you can select what landmarks take in to account for the dataset creation.
What we have to do is just set the correct paths of the input raw data and the output file.
As output you will get a .csv file.
The main columns of the output are:
* 'Sequenza': it represents the ID of the sequence.
* 'Frame': it represents the ID of a frame inside a sequence.
* Columns that represent the features: it can be of variable length.
* 'Label: it represents the VAS value associated to each sequence. The value is the same for each item of the same sequence.

A row in the .csv is something like ['Sequenza', 'Frame', 'Feature 1', ..., 'Feature N', 'Label']

### Preprocessing

Then we need to preprocess the dataset to make it fit our model. 
In particular we need to have the sequence all of the same length. To achieve this we used to apply Oversampling and Undersampling to the sequences of the dataset. We also applied some normalization techniques.
In this case it's just needed to set the correct paths at the start of the 'preprocessing.ipynb' notebook.
It needs as input two .csv file that one represents the train set and the other one the test set.
As output it will give two files, one for the test set and one for the train set.

### Training

The third phase is defining our model and training it.
To do this task you can ran the 'train.ipynb' notebook. 
It will perform a subdivision of the data in validation and train sets, and then performes the model training. To use this file it's just needed as the notebook described before to set the correct file paths. In this case you need to set the input file path and the output file paths for the Tensorboard visualization and the model saving.
In particular we save in './logs/gradiente_tape' a folder identified by the time when the code is executed that contains a ./train and ./validation folder for the summary of each phase.

As concerned as the training, the data that we save for Tensorboard visualization are:
* Train Loss: we used the Mean Squared Error.
* Train Error: we used the Mean Absolute Error.
* Validation Loss: we used the MeanSquaredError.
* Validation Error: we used the MeanAbsoluteError.
* Train Predictions Histogram: that represents the distribution of the predicted label at each epoch.
* Train Ground Truth Histogram: that represents the distribution of the GT label at each epoch.
* Valid Predictions Histogram: that represents the distribution of the predicted label at each epoch.
* Valid Ground Truth Histogram: that represents the distribution of the GT label at each epoch.
* Train Confusion Matrix: that represents the Confusion Matrix at each epoch.

Additional data regarding the model that are saved for Tensorboard as text are:
* All the model Hyperparameters.
* The model summary.

An example of model data.


![Model Summary][model-summary]


### Testing

The last phase is evaluating the model on the test set. The notebook for doing this is 'test.ipynb'.
To run this code you need to set the train set path, the path to the model that needs to be loaded and the log path correctly. We suggest you to use the same name used during the training.

The data we saved for the Tensorboard visualization are:
* Test Error: we used the Mean Absolute Error.
* Test Predictions Histogram: that represents the distribution of the predicted label.
* Test Ground Truth Histogram: that represents the distribution of the GT label.
* Test Confusion Matrix: that represents the Confusion Matrix.

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
   
As an alternative we uploaded a simple notebook, called 'result.ipynb' that follows our notation for the directory tree that can just be executed changing the paths.

Here an example of Tensorboard visualization.

![Tensorboard Example][tensorboard-example]




<!-- MARKDOWN LINKS & IMAGES -->
[model-summary]: images/model_summary.jpg
[tensorboard-example]: images/tensorboard_sample.png


Under the Histogram tab you can see all the histogram stored for the analysis of the predictions and ground truth distributions.
Under the Text tab you can see the model hyperparameters and the model layers.
The Confusion Matrixes can be seen under the Images tab, while under the scalar tab the losses and the error values are displayed.
On the bottom left you can filter the data. They are visualized by the model name and by the phase (train, test or validation) that you want to analize.
