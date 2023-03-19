# HAR_PAMAP2

This project develops a Human Activity Recognition System using a single sensor. The data from the PAMAP2 dataset is used to train a convolutional neural network to classify between 8 activties.

1. To run the code, download the PAMAP2 dataset and place it in the project directory
2. Run train.py to train the model with the data from the PAMAP2 dataset

### File Description
1. model.py - Creates the developed convolutional neural network
2. data_processing.py - Prepares the PAMAP2 data into samples to be input to the model
3. dataset.py - Creates a training and testing dataset
4. hyp_par_opt.py - Runs hyperparameter optimization of the neural network model using Optuna with tree-parzen based optimization