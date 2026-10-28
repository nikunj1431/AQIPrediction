AQI Predictor

Predicts next-day AQI based on last-week pollutant features and date for different cities, incorporating the day of the week (weekends have different pollutant levels than weekends) and if it's a festival (festivals like diwali tend to have a large number of firecrackers being used). 

This project demonstrates three models: Linear Regression, Random Forest, and Decision Tree.

Each model is demonstrated through a seperate jupyter notebook, each handling data cleaning, regression, classification and results by itself.
Each notebook also demonstrates the results within the jupyter notebook code. 

The following tech stack was used:
- Pandas
- Numpy
- matplotlib
- scikit-learn

Usage instructions:
Clone/Download the repo and unzip. Ensure that the notebook being run and the folder 'AQIData2015to20' are located in the same directory at the same level. Then simply run each cell individually in order or run all at the same time. 

Dependencies:
- Pandas 3.1.0
- numpy 2.5.4
- matplotlib 3.11.2
- sklearn 1.9.1
