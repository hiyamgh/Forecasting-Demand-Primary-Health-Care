
'''
* weekly_dfs_to_difference & biweekly_dfs_to_difference are data subsets that had too many lags in their
ACF plots so we will difference them by 1 in order to reduce the number of lags
so we can include the lags as columns in our multivariate_datasubsets data

* weekly_lags & biweekly_lags are the number of lags to include in each data subset
(as columns in the data)
We extract them by looking at the first values in the acf plot that are outside the
shaded region

* in the data subsets that needed differencing, we looked at the lags in the acf plots
after differencing the data
'''

weekly_stationary_dfs = ['Gynaecology_akkar', 'Gynaecology_Tripoli', 'Pediatrics_akkar', 'Pharmacy_bikaa']


weekly_dfs_to_difference = {
    'General Medicine_akkar': 1,
    'General Medicine_bikaa': 1,
    'General Medicine_Tripoli': 1,
    'Gynaecology_bikaa': 1,
    'Pediatrics_bikaa': 1,
    'Pediatrics_Tripoli': 1,
    'Pharmacy_akkar': 1,
    'Pharmacy_Tripoli': 1
}

biweekly_dfs_to_difference = {
    'General Medicine_bikaa': 1,
    'General Medicine_Tripoli': 1,
    'Gynaecology_bikaa': 1,
    'Pediatrics_bikaa': 1,
    'Pharmacy_akkar': 1,
    'Pharmacy_bikaa': 1,
}


weekly_lags = {
    'General Medicine_akkar': 2,
    'General Medicine_bikaa': 2,
    'General Medicine_Tripoli': 2,
    'Gynaecology_akkar': 5,
    'Gynaecology_bikaa': 2,
    'Gynaecology_Tripoli': 4,
    'Pediatrics_akkar': 5,
    'Pediatrics_bikaa': 2,
    'Pediatrics_Tripoli': 2,
    'Pharmacy_akkar': 2,
    'Pharmacy_bikaa': 5,
    'Pharmacy_Tripoli': 2
}

biweekly_lags = {
    'General Medicine_akkar': 3,
    'General Medicine_bikaa': 2,
    'General Medicine_Tripoli': 2,
    'Gynaecology_akkar': 5,
    'Gynaecology_bikaa': 2,
    'Gynaecology_Tripoli': 2,
    'Pediatrics_akkar': 3,
    'Pediatrics_bikaa': 1,
    'Pediatrics_Tripoli': 4,
    'Pharmacy_akkar': 2,
    'Pharmacy_bikaa': 1,
    'Pharmacy_Tripoli': 3
}

weekly_lags_without_diff = {
    'General Medicine_akkar': 4,
    'General Medicine_bikaa': 5,
    'General Medicine_Tripoli': 5,
    'Gynaecology_akkar': 5,
    'Gynaecology_bikaa': 5,
    'Gynaecology_Tripoli': 4,
    'Pediatrics_akkar': 5,
    'Pediatrics_bikaa': 5,
    'Pediatrics_Tripoli': 5,
    'Pharmacy_akkar': 5,
    'Pharmacy_bikaa': 5,
    'Pharmacy_Tripoli': 5
}

biweekly_lags_without_diff = {
    'General Medicine_akkar': 3,
    'General Medicine_bikaa': 5,
    'General Medicine_Tripoli': 4,
    'Gynaecology_akkar': 5,
    'Gynaecology_bikaa': 5,
    'Gynaecology_Tripoli': 2,
    'Pediatrics_akkar': 3,
    'Pediatrics_bikaa': 5,
    'Pediatrics_Tripoli': 4,
    'Pharmacy_akkar': 5,
    'Pharmacy_bikaa': 4,
    'Pharmacy_Tripoli': 3
}