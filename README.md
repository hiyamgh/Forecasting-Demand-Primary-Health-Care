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
    - **Time series plots**: [temporal\_structur\_plots](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/output/temporal_structure_plots)
    - **EDA**: [eda.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/initial_input/eda.py): Genrates all plots related to exploratory data analysis.
    - **EDA Plots**: [eda\_plots](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/tree/master/initial_input/eda_plots)





