# Forecasting Demand of Primary Health Care in Lebanon: insight from the Syrian Refugee Crisis

Lebanon is a middle-income Middle Eastern country that has been hosting around 1.5 million Syrian Refugees, a figure representing one of the largest concentrations of refugees per capita in the world. The enormous influx of displaced refugees remains a daunting challenge for the national health services in a country whose own population is at 4 million, a problem that is exacerbated by the lack of national funds to allow respective municipalities to sufficiently balance its own services between host and refugee communities alike, prompting among the Lebanese population a sense of being disadvantaged by the presence of refugees

Our manuscript henceforth addresses the following question: **can we analyse the spatiotemporal surge in demand recorded by primary health care centers through data provided by the Lebanese Ministry of Health in light of the peaks in events emanating from the Syrian war, and further model it in order to yield reasonably accurate predictions that can assist policy makers in their readiness to act on surges in demand within prescribed proximity to the locations in Syria where the peaks have taken place?**

To this end, we embark on a process that analyses data from the Lebanese ministry of public health, representing primary health care demand, and augment it with data from the ***Syrian Violations Data Center***, a leading repository for documenting casualties in the Syrian war. The objective of this study is to analyse the surge in demand on primary health care centers using data from MoPH in reference to the peaks in the Syrian war derived from the VDC, to produce both **pointwise** as well as **probabilistic forecasting** of the demand using a suite of statistical and machine learning models, to improve the recall values of surges in demand using **utility based regression** for capturing rare events – in our context, instances when the demand surges due to large scale events in Syria –  and to reveal the rules and interactions between major input features that led these models to their prediction, using machine learning interpretability techniques.

## Initial Input and Datasets
   - [Faour.xlsx](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/input/Faour): is the original dataset of patient visits to Lebanese Health Centers
   - [VDC Datasets](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/input/vdc): Contains the three different datasets taken from **Syrian Violations Data Center**, one for each year (2014, 2015, 2016). [vdc_grouped.csv](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/input/vdc/vdc_grouped.csv) is these three grouped together into one data frame
   - [Weather.csv](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/output/weather) is the weather data from the International Weather Station at Beirut International Airport. The weather data is scraped in a **daily** manner, for the **Beirut Governorate**, for the years 2014, 2015, 2016 from [https://www.wunderground.com/history/monthly/lb/beirut/](https://www.wunderground.com/history/monthly/lb/beirut/)

### Translations
All of the text entries in this dataset were in Arabic and so were translated using dictionaries that we developed in our capacity as native speakers. Dictionaries for translations could be found in here: [translations.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/helper_codes/translations.py)

### Preprocessing, Data Cleaning, Helper Codes:
   - **Pre-processing and Data Cleaning**: [data_set.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/helper_codes/data_set.py), [data_subset.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/helper_codes/data_subset.py)
   - **Visualizations**: [data_visualize.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/helper_codes/data_visualize.py)
   - **Weather data scraping**: [generate_weather_data.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/generate_weather_data.py)
   - **VDC Data Handling**: [generate_vdc_data.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/generate_vdc_data.py)
        - **Adaptive binning** for generating **civilians rank**
        - **distances** from **Lebanese health centers** to **places of death in Syria in times of large scale events**
        - distribution of **civilians rank** by **year**
   - **VDC Plots**: [vdc\_plots](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/eda_plots/vdc_plots) contains all plots related to VDC explorations
   - **Generating inital 12 data subsets**: [generate_Faour_data_subsets.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/generate_Faour_datasubsets.py) This script is responsible for subsetting the original input data [Faour.xlsx](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/input/Faour/Faour.xlsx) into 12 data subsets, each representing one of the three governorates on interest (Akkar, Beqaa, Tripoli) and one of the services in the health centers (General Medicine, Gynecology, Pediatrics, and Pharmacy). the Focus was on Syrian Nationalities. The datasubsets could be found here: [Faour datsubsets](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/output/Faour_datasubsets). *The datasubsets follow the following naming structure: [service name]\_[Syria]\_[Governorate name]*
   - **Generating the final 12 datasubsets**: Here, we take the initial 12 datasubsets (the buller above) and we:
        - apply weekly down-sampling (in [Faour.xlsx](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/input/Faour/Faour.xlsx)) demand was in a daily basis. For a more general view, we aggregated demand to make it weekly
        - we added vdc data next to each week
        - we added weather data next to each week
        - we added distance from the large scale events in Syria next to each week
    **Script**: [generate_multivariate_datasubsets.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/generate_multivariate_datasubsets.py), which generates the data subsets and saves them here: [multivariate data subsets](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/output/multivariate_datasubsets)

## Input - Collated Data

   - **Collated Datasets**: [generate_collated_separated_input_datasets.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/generate_collated_separated_input_datasets.py): collates the 12 data subsets together whilst adding **one hot encoding** to the service name and Governorate. The resulting **collated data sets** are found in:
        - [input/all\_with\_date](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/input/all_with_date/collated): **including** the ``date`` column
        - [input/all\_without\_date](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/input/all_without_date/collated) **without including** the ``date`` column
   - **Separated Datasets**: contains the separated datasets which we have used for collation. These are found in the following directory: [input/separated](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/input/all_with_date/separated)
   - **Disclaimer**: in each of tehe directories above, we set 9 column variations, removing certain columns and seeing which variation achieves teh best results. Therefore, there exists the *[minus\_colmn name]* in the directories above

## Exploratory Data Analysis
   - **Time Series**: [Code/timeseries.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/timeseries.py): Generates:
        -  **time series plots**: ACF, PACF, auto-correlation, persistence model, line plot, histogram, density plot.
        - **white noise detection**: L-jung box test
        - **stationarity checking**: Augmented Dickey-Fuller Test
        - adds lags, trend, and seasonality to the time series.
   - **Time series plots**: [temporal\_structure\_plots](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/output/temporal_structure_plots)
   - **EDA**: [eda.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/eda.py): Genrates all plots related to exploratory data analysis.
   - **EDA Plots**: [eda\_plots](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/eda_plots)

## Supervised Learning
We have done a suite of Machine learning models with cross validation. We have used 10-folds-10-reperats for cross validation.
   - **Cross Validation**: [cross\_validation.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/cross_validation.py): the code that handles cross validation with hyper parameter tuning (Grid Search), produces the learning curves, applies scaling to the data.
  - **Hyper parameters**: [model_hyperparams_grid.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/models_hyperparams_grid.py) contains dictionaries of hyper parameters to be search by Grid Search in [cross\_validation.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/cross_validation.py))
  - [container.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/container.py) contains paths and scaling options to each dataset
  - **Python Run Files for Column Variations**: [shallow collated run files](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/Code/python_run_files/shallow/train_collated_test_collated) is a directory containing all run files for applying cross validation on the **9** column variation we have. The [cross_validation.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/Code/cross_validation.py) code is generaic and can be applied to any dataset as indicated below.
  - ***Disclaimer** In order to use the run files, put them as main children of the [Code directory](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/Code)*


### Supervised Learning Results

|                        dataset                       |   best_model   |    r2    |   rmse   |    mse   |    mae   |
|:----------------------------------------------------:|:--------------:|:--------:|:--------:|:--------:|:--------:|
| all_columns                                          | linear_svr     | 0.825409 | 37.4547  | 1402.855 | 25.91299 |
| all_columns_minus_weather                            | linear_svr     | 0.824745 | 37.52578 | 1408.184 | 25.84897 |
| all_columns_minus_weather_minus_lags                 | ada_boost      | 0.40074  | 69.3909  | 4815.098 | 46.28747 |
| all_columns_minus_weather_minus_vdc                  | linear_svr     | 0.823036 | 37.70834 | 1421.919 | 25.84334 |
| all_columns_minus_weather_minus_distance             | linear_svr     | 0.824562 | 37.54541 | 1409.658 | 25.87975 |
| all_columns_minus_weather_minus_civilians            | linear_svr     | 0.822958 | 37.71668 | 1422.548 | 25.86042 |
| all_columns_minus_weather_minus_lags_minus_distance  | gradient_boost | 0.50776  | 62.89025 | 3955.183 | 43.24637 |
| all_columns_minus_weather_minus_lags_minus_civilians | ada_boost      | 0.386509 | 70.20998 | 4929.441 | 47.02291 |
| univariate                                           | linear_svr     | 0.818153 | 38.22504 | 1461.153 | 26.03276 |

## Utility-Based-Regression (SMOGN)
We have an imbbalanced regression, were **rare events**, which are mainly high demand values, are poorly present in the data and they do add confusion to the machine learning models and degrade performance. We have used *Branco's* approach in the paper [SMOGN: a Pre-processing Approach for Imbalanced Regression](http://proceedings.mlr.press/v74/branco17a/branco17a.pdf) and forked [their repository](https://github.com/paobranco/SMOGN-LIDTA17/tree/master/). We have used their **R codes** and called some appropriate helper functions from our **python** code using python's [rpy2 module]()
  - [DIBSRegress.R](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/DIBSRegress.R) Code that contains SMOGN algorithm
  - [smogn.R](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/smogn.R) contains helper functions in R
  - [cross_validation_smogn.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/cross_validation_smogn.py) contains the code that applies "SMOGN" inside cross validation
  - [extract\_rare.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/extract_rare.py) helper code for getting indices of rare values in testing data, extracting relevance function, and plotting rare values before and after oversampling
  - [utility\_based\_error\_metrics.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/utility_based_error_metrics.py) code that computes **utility-based-error-metrics** Like
  **F<sub>1</sub>, precision, and recall**

### Requirements for Running Utility-Based-Regression

The experimental design was implemented in both Python and R language. Both code and data are in a format suitable for Python and R environment.

In order to replicate the experiments in **CodeUbr** you will need a working installation of R. Check [https://www.r-project.org/] if you need to download and install it.

You must have R ``3.6.x``

In your R installation you also need to install the following additional R packages:

  - DMwR
  - performanceEstimation
  - UBL
  - uba: [https://www.dcc.fc.up.pt/~rpribeiro/uba/](https://www.dcc.fc.up.pt/~rpribeiro/uba/)
  - operators
  - class
  - fields
  - ROCR
  - Hmisc
  - R tools version 35. [https://cran.r-project.org/bin/windows/Rtools/](https://cran.r-project.org/bin/windows/Rtools/)


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

Other than R, in order to run the remaining experiments in **CodeUbr** as well as the experiments in **Code** you need the following python modules

  - rpy2 version 2.9.5
  - pandas version 0.24.0
  - keras
  - tensorflow
  - sklearn
  - xgboost
  - scipy
  - matplotlib
  - numpy

Check the other README files in each folder to see more detailed instructions on how to run the experiments.





