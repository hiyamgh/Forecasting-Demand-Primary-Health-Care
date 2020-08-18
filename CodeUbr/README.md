## Reproducing the Paper Experiments

## Contents

  This folder contains  all code that is necessary to replicate the
  experiments reported in the paper *"Ministry of Public Health: Predicting Demand on Health Care Centers in Lebanon"*. 
  
  
  This directory contains the following folders
  
  - **python_run_files** folder - the run files used to reproduce the experiments
  
  - **bash_scripts** folder - the bash scripts used to run the experiments on AUB HPC cluster - Octopus. More information about Octopus can be found here: [https://hpc-aub-users-guide.readthedocs.io/en/latest/octopus/overview.html](https://hpc-aub-users-guide.readthedocs.io/en/latest/octopus/overview.html)
  
  - **Lime** folder - the Lime notebooks that are used to run Lime experiments on our data.

  **python_run_files - Content**

   - **neural_networks** folder - contains the run files for neural networks.**Disclaimer**: Neural Networks experiments are done **only on the collated datasets** (as the separated datasets are small in size (~ 150 rows each) it is not worthy doing them)

      - **with/without relevance matrix**
          - **all_columns**
             - **nn_ac_nrelmat.py** file - neural networks on ``all_columns`` dataset **without relevance matrix**
             - **nn_ac_wrelmat.py** file - neural networks on ``all_columns`` dataset **with relevance matrix**

          - **all_columns_minus_weather**
             - **nn_acmw_nrelmat.py** file - neural networks on ``all_columns_minus_weather`` dataset **without relevance matrix**
             - **nn_acmw_wrelmat.py** file - neural networks on ``all_columns_minus_weather`` dataset **with relevance matrix**

          - **all_columns_minus_weather_minus_lags**
             - **nn_acmwml_nrelmat.py** file - neural networks on ``all_columns_minus_weather_minus_lags`` dataset **without relevance matrix**
             - **nn_acmwml_wrelmat.py** file - neural networks on ``all_columns_minus_weather_minus_lags`` dataset **with relevance matrix**

          - **all_columns_minus_weather_minus_vdc**
             - **nn_acmwmv_nrelmat.py** file - neural networks on ``all_columns_minus_weather_minus_vdc`` dataset **without relevance matrix**
             - **nn_acmwmv_wrelmat.py** file - neural networks on ``all_columns_minus_weather_minus_vdc`` dataset **with relevance matrix**

          - **univariate**
            - **nn_uni_nrelmat.py** file - neural networks on ``univariate`` dataset **without relevance matrix**
            - **nn_uni_wrelmat.py** file - neural networks on ``univariate`` dataset **with relevance matrix**


          - **nn_acmw_nrel.py** file - neural networks on ``all_columns_minus_weather`` dataset
          - **nn_acmwml.py** file - neural networks on ``all_columns_minus_weather_minus_lags`` dataset
          - **nn_acmwmv.py** file - neural networks on ``all_columns_minus_weather_minus_vdc`` dataset
          - **nn_uni.py** file - neural networks on ``univariate`` dataset

      - **No Oversampling**
          - **nn_nosmote.py** file - neural networks without oversampling. Applied to all datasets except ``univariate``
          - **nn_uni_nosmote** file - neural networks without oversampling. Applied to ``univariate`` dataset


   - **shallow** folder - contains the run files for the shallow models.**Disclaimer**: Shallow experiments are done **only on the collated datasets** (as the separated datasets are small in size (~ 150 rows each) it is not worthy doing them)

      - **with/without relevance matrix**
          - **shallow_norelmatrix.py** file - shallow models **without relevance matrix**. Applied to all datasets except ``univariate``
          - **shallow_withrelmatrix.py** file - shallow models **with relevance matrix**. Applied to all datasets except ``univariate``

          - **shallow_uni_norelmatrix.py** file - shallow models **without relevance matrix**. Applied to ``univariate`` dataset
          - **shallow_uni_withrelmatrix.py** file - shallow models **with relevance matrix**. Applied to ``univariate`` dataset

      - **No Oversampling**
          - **shallow_nosmote.py** file - shallow models without oversampling. Applied to all datasets except ``univariate``
          - **shallow_uni_nosmote.py** file - shallow models without oversampling. Applied to ``univariate`` dataset



  **bash_scripts - Content**
   - **with/without relevance matrix**
     - ** all_columns **
      - **nn_ac_nrelmat.sh** file - bash script to run **nn_ac_nrelmat.py** - **without relevance matrix**
      - **nn_ac_wrelmat.sh** file - bash script to run **nn_ac_nrelmat.py** - **with relevance matrix**

        - ** all_columns_minus_weather **
          - **nn_acmw_nrelmat.sh** file - bash script to run **nn_acmw_nrelmat.py** - **without relevance matrix**
          - **nn_acmw_wrelmat.sh** file - bash script to run **nn_acmw_nrelmat.py** - **with relevance matrix**

        - ** all_columns_minus_weather_minus_lags **
          - **nn_acmwml_nrelmat.sh** file - bash script to run **nn_acmwml_nrelmat.py** - **without relevance matrix**
          - **nn_acmwml_wrelmat.sh** file - bash script to run **nn_acmwml_nrelmat.py** - **with relevance matrix**

        - ** all_columns_minus_weather_minus_vdc **
          - **nn_acmwmv_nrelmat.sh** file - bash script to run **nn_acmwmv_nrelmat.py** - **without relevance matrix**
          - **nn_acmwmv_wrelmat.sh** file - bash script to run **nn_acmwmv_nrelmat.py** - **with relevance matrix**

        - ** univariate **
          - **nn_ac_nrelmat.sh** file - bash script to run **nn_uni_nrelmat.py** - **without relevance matrix**
          - **nn_ac_wrelmat.sh** file - bash script to run **nn_uni_nrelmat.py** - **with relevance matrix**

    - **No Oversampling**
      - **nn_oct_nosmote.sh** file - bash script to run **nn_nosmote.py**
      - **nn_uni_nosmote.sh** file - bash script to run **nn_uni_nosmote.py**



   **Lime - Content**: We will be using the trained models saved in the ``trained_models`` directory of the output folder
   of each of the experiments in order to run Lime on them

   - **slw_ac.ipynb** file - contains the Lime notebook for analyzing results of shallow models on ``all_columns`` dataset
   - **slw_acmw.ipynb** file - contains the Lime notebook for analyzing results of shallow models on ``all_columns_minus_weather`` dataset
   - **slw_acmwml.ipynb** file - contains the Lime notebook for analyzing results of shallow models on ``all_columns_minus_weather_minus_lags`` dataset
   - **slw_acmwmv.ipynb** file - contains the Lime notebook for analyzing results of shallow models on ``all_columns_minus_weather_minus_vdc`` dataset
   - **slw_univariate.ipynb** file - contains the Lime notebook for analyzing results of shallow models on ``univariate`` dataset

   - **ac_df_train.csv** file - the collated training data of the ``all_columns`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns
   - **ac_df_test.csv** file - the collated testing data of the ``all_columns`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns

   - **acmw_df_train.csv** file - the collated training data of the ``all_columns_minus_weather`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns
   - **acmw_df_test.csv** file - the collated testing data of the ``all_columns_minus_weather`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns

   - **acmwml_df_train.csv** file - the collated training data of the ``all_columns_minus_weather_minus_lags`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns
   - **acmwml_df_test.csv** file - the collated testing data of the ``all_columns_minus_weather_minus_lags`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns

   - **acmwmv_df_train.csv** file - the collated training data of the ``all_columns_minus_weather_minus_vdc`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns
   - **acmwmv_df_test.csv** file - the collated testing data of the ``all_columns_minus_weather_minus_vdc`` dataset with one hot encoded vectors removed and replaced by the actual categorical columns

   The above datasets are created automatically by running Lime notebooks:

   - **slw_ac.ipynb** will create **ac_df_train.csv** and **ac_df_test.csv**
   - **slw_acmw.ipynb** will create **acmw_df_train.csv** and **acmw_df_test.csv**
   - **slw_acmwml.ipynb** will create **acmwml_df_train.csv** and **acmwml_df_test.csv**
   - **slw_acmwmv.ipynb** will create **acmwmv_df_train.csv** and **acmwmv_df_test.csv**
   - **slw_univariate.ipynb** will not create any dataset


  **This folder's Content**

  - **cross_validation_oversampling.py** file - the main python file for doing cross validation with hyper parameter tuning for the shallow models with SMOGN embedded. SMOGN will be applied on the training data after picking the best set of hyper parameters from cross validation and re-training the model with the chosen set of hyper parameters on the training data again just before testing it on the testing data 
  
  - **models_hyperparams_grid.py** file - contains lists of hyper parameters the code in **cross_validation_oversampling.py** will be looping over through the grid search process in cross validation
  
  - **DIBSRegress.R** file - the code implementing the SMOGN strategy for utility based regression

  - **smogn.R** file - R contains main R functions to be called from python. Contains the following oversampling algorithms:
    - **SMOGN**: for more information please use the following [paper](http://proceedings.mlr.press/v74/branco17a/branco17a.pdf)
    - **SmoteR**: for more information please use the following [link](https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/SmoteRegress)
    - **Gaussian Noise**: for more information please use the following [link](https://rdrr.io/cran/UBL/man/gaussNoiseRegress.html)
    - **Random Undersampling**: for more information please use the following [link](https://www.rdocumentation.org/packages/UBL/versions/0.0.6/topics/RandUnderRegress)
  
  - **neural_networks.py** file - contains the code for applying neural networks with utility based regression
  
  - **relevance_helper.py** file - contains helper functions for plotting data distribution (before and after SMOGN/SmoteR/RandomUnderSample/GN), rare values, and relevance plot
  
  - **utility_based_error_metrics.py** file - contains function that issues utility based error metrics given the actual and predicted columns
  
  - **container.py** & **container_univariate.py** file - contains different combinations of distance, relevance matrices, and input paths. Used as helper files in the **python_run_files**
  
  - **README.md** this file
  

## Replicating Experiments 

  In order to replicate the experiments in our paper, you must do the following:
  
  1. place the appropriate python run file from **python_run_files** in the same directory were: cross_validation_oversampling.py, container.oy, container_univariate.py, DIBSRegress.R, models_hyperparams_grid.py, neural_networks.py, relevance_helper.py, smogn.R, and utility_based_regression.py are found (i.e. all these files must be in the same directory)
  2. If you want to use the bash scripts in order to submit your jobs on an HPC cluster (with **SLURM** scheduler):
     - the bash scripts must also be put in the same directory with the files mentioned in the first bullet


## Output of the Experiments

The experiments will produce results in the mentioned ``output_folder`` parameter in the code inside the run files in **python_run_files** folder. The output folder will contain the following:

  - **actual_vs_predicted** folder - contains the actual vs predicted lineplots of predictions on the testing data for each model 
  
  - **actual_vs_predicted_scatter** folder - contains the actual vs predicted scatter plots of the predictions on the testing data with a bisector on the diagonal of the image which helps in assessing how precise predictions are for each model
  
  - **learning_curve** folder - contains the learning curves produced by each model
  
  - **output_vector_datasets** folder - contains the testing datasets with the `predicted` column added next to the column of the `target_variable` for each model
  
  - **plots** folder - contains
    
    - density plot of the data distribution before and after SMOGN/SmoteR/RandomUnderSample/GN
    
    - training data line plot of the target variable with rare values marked
    
    - relevance plot of the target variable in the training data 
  
  - **train_test_before_modeling** folder - contains the training and testing datasets before applying any modelling exercise. If the user has provided the training and testing data him/herself, those will be the ones. If the user has provided only one dataset and he/she has specified the ``split_ratio`` parameter that determines the ratio of the testing data portion of the dataset provided, then the two datasets will be that dataset itself split accordingly between training and testing
  
## Necessary Software

In order to replicate the experiments in **CodeUbr** you will need a working installation 
of R. Check [https://www.r-project.org/] if you need to download and install it.

You must have R 3.6.x

In your R installation you also need to install the following additional R packages:

  - DMwR
  - performanceEstimation
  - UBL
  - uba: [link](https://www.dcc.fc.up.pt/~rpribeiro/uba/)
  - operators
  - class
  - fields
  - ROCR
  - Hmisc
  - R tools version 35. [link](https://cran.r-project.org/bin/windows/Rtools/)


  All the above packages, with the exception of uba package, can be installed from CRAN Repository directly as any "normal" R package. Essentially you need to issue the following commands within R:

```r
install.packages(c("DMwR", "performanceEstimation", "UBL", "operators", "class", "fields", "ROCR"))
install.packages("Hmisc")
```


 Before you install the uba package, you need to have the latest version of **R tools**. Check [https://cran.r-project.org/bin/windows/Rtools/](https://cran.r-project.org/bin/windows/Rtools/)


 Additionally, you will need to install uba package from a tar.gz file that you can download from [http://www.dcc.fc.up.pt/~rpribeiro/uba/](http://www.dcc.fc.up.pt/~rpribeiro/uba/). 

 
 For installing this package issue the following command within R:
```r
install.packages("uba_0.7.7.tar.gz",repos=NULL,dependencies=T)
```

Other than R, in order to run the remaining experiments in **Code_ubr** as well as the experiments in **Code** you need the following python modules
 
  - rpy2 version 2.9.5
  - pandas version 0.24.0
  - keras
  - tensorflow
  - sklearn
  - xgboost
  - scipy
  - matplotlib
  - numpy


### References
[1] Branco, P. and Torgo, L. and Ribeiro R.P. (2017) *"SMOGN: a Pre-processing Approach for Imbalanced Regression"*  Procedings of Machine Learning Research, LIDTA 2017, Full Paper, Skopje, Macedonia. (to appear).
