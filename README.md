# SupervisedClusteringTeleassistance

# DATA EXPLORATION
In this first phase, the data exploration, we focused on analyzing the file containing the data to understand what it was about, how many and which features were present, what they represent, and we looked for any missing values. This also helps to understand which features are more useful and which can be eliminated in a process of feature selection.
First, we opened the "Parquet" file and loaded it into a "dataframe".
We then counted the null/missing values for each feature, and this was the result:

![image](https://github.com/user-attachments/assets/f654a595-3446-482e-8174-cf5aa5884be4)

We observe that there are 484,291 records. Most of the features do not have null values, except for a few, which we investigated further.
Specifically, we discovered that the features “codice_provincia_residenza” and “codice_provincia_erogazione,” which had null values, actually corresponded to the value "NA," which refers to Naples. Similarly, by analyzing the ISTAT code, we found that the 135 missing values of “comune_residenza” were not missing but referred to a municipality named "None."
Next, we checked for duplicates in the “id_pazienti” feature, discovering that most patients had only one visit during the analyzed period. Therefore, we decided not to use this feature.

The feature “id_prenotazione” proved to be of little use, as each booking is unique.


Additionally, many pairs of features are overlapping, and therefore redundant. For example, the columns within the following pairs all identify the same thing. As a result, it is possible to use only one column from each pair (the one with the code).

![image](https://github.com/user-attachments/assets/db334a32-b339-41bd-b8b2-159edc369df6)

We then analyzed the number of sub-keys within the features, from which we can see the number of ASLs present, the number of hospitals, how many types of healthcare services exist, how many people leave their own region to receive services, and so on. This also helps us to identify any anomalous data, which, in this case, did not occurred.

![image](https://github.com/user-attachments/assets/9d605da8-b774-4ac7-b830-32ba76846b71)

We counted the number of healthcare providers, which is just under 40,000. This means that, on average, each doctor had 12 patients. We believe that this data is not useful for calculating the increase in teleconsultations, so we will not use it.

Additionally, we want to create a new feature, called "waiting time," based on the two columns “booking date” and “service delivery date.” This could be useful to understand whether this parameter affects the increase in teleconsultations.

This phase of data exploration has been extremely helpful in understanding the nature of the data and identifying which features might be more useful, as well as in cleaning up any messy data.
