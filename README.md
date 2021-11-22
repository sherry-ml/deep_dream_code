# deep_dream_code
This repository will contain all the code files/functions/class definitions which will be imported/pulled into main jupyter notebook and run there.
Structure of the repository at this moment is:
  1) utils.py which contains functions mainly related to transformations, augmentations, gradcam, creation of dataloaders/datasets etc
  2) main.py which contains functions used to run train and test loops, defining optimizer etc
  3) utils folder which is empty currently but will be used to contain code files containing code for specific funcionailty 
  4) models folder which contains class and function definitions for different CNN models
  5) code_factory which contains individual code files for different test and train functionality. This is not implemented as of now. In future there will be a wrapper function for each of the functionality in main.py which will pull the relevant code from code_factory folder.

A) models/resnet.py : Contains Class and function definition for resnet18 and resnet34 models
B) main.py : Contains the relevant functions to run training and testing loops and optimizer definition. More information about these functions can be found at https://github.com/sherry-ml/EVA7/blob/main/S7/README.md 
    - train function: Contains code to run training for one epoch
    - test function: Contains code to run validation loop on test dataset after end of each training epoch
    - train_test_model: This function intergrates the above two functions, defines optimizer and runs training/test loop for defined number of epochs and finds misclasssified images and stores it in a list.
c) utiity.py: Contains code that defines default_DL, set_compose_params, tl_ts_mod  functions and C_10_DS Class. More information about these functions can be found at https://github.com/sherry-ml/EVA7/blob/main/S7/README.md. 
    - It also contains code that implements gradcam functionality thorugh GradCAM class
    - show_sample_img function which takes a batch from testloader and shows number of images passed on as an argument to the function.
    - torch_device which returns the device to be used depending on if the cuda functionality is present or not
    - view_model_summary : Prints the model summary 
    - display_incorrect_images : This function displays incorrect images along with predicted and actual labels.
    - show_plots : Displays the graph for training and test loss 
