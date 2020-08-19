# Forecasting Demand of Primary Health Care in Lebanon: insight from the Syrian Refugee Crisis

Lebanon is a middle-income Middle Eastern country that has been hosting around 1.5 million Syrian Refugees, a figure representing one of the largest concentrations of refugees per capita in the world. The enormous influx of displaced refugees remains a daunting challenge for the national health services in a country whose own population is at 4 million, a problem that is exacerbated by the lack of national funds to allow respective municipalities to sufficiently balance its own services between host and refugee communities alike, prompting among the Lebanese population a sense of being disadvantaged by the presence of refugees

Our manuscript henceforth addresses the following question: **can we analyse the spatiotemporal surge in demand recorded by primary health care centers through data provided by the Lebanese Ministry of Health in light of the peaks in events emanating from the Syrian war, and further model it in order to yield reasonably accurate predictions that can assist policy makers in their readiness to act on surges in demand within prescribed proximity to the locations in Syria where the peaks have taken place?**

To this end, we embark on a process that analyses data from the Lebanese ministry of public health, representing primary health care demand, and augment it with data from the ***Syrian Violations Data Center***, a leading repository for documenting casualties in the Syrian war. The objective of this study is to analyse the surge in demand on primary health care centers using data from MoPH in reference to the peaks in the Syrian war derived from the VDC, to produce both **pointwise** as well as **probabilistic forecasting** of the demand using a suite of statistical and machine learning models, to improve the recall values of surges in demand using **utility based regression** for capturing rare events – in our context, instances when the demand surges due to large scale events in Syria –  and to reveal the rules and interactions between major input features that led these models to their prediction, using machine learning interpretability techniques.

## Utility Based Regression
We have an **im-balanced regression** problem. The most rare and relevant cases of demand are poorly available in the data, and by applying **Exploratory Data Analysis** we realize that the rare events, where demand is vvery high, are mainly in the year 2016. This causees performance degredation for the machine learning models as they learn from an in-balanced environement. In order to address this, we have mainly refrred to [http://proceedings.mlr.press/v74/branco17a/branco17a.pdf](http://proceedings.mlr.press/v74/branco17a/branco17a.pdf) for oversampling our data in order to achieve balanced regression.

We have forked the source code for utility-based-regression (SMOGN) from here [https://github.com/paobranco/SMOGN-LIDTA17](https://github.com/paobranco/SMOGN-LIDTA17)

## Content
<!---
  - [DIBSRegress.R](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/DIBSRegress.R):  the code implementing the SMOGN strategy for regression
  - [cross_validation_smogn.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/cross_validation_smogn.py): the code for applying cross validation with oversampling (SMOGN). In this code, we call the appropriate **R** functions from python using Rpy2 module
  - **Helper Codes**
      - [extract_rare.py](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/extract_rare.py)
      - [smogn.R](https://github.com/hiyamgh/Forecasting-Demand-Primary-Health-Care/blob/master/CodeUbr/smogn.R) --->

## Probabilistic Forecasting

## SHAP

