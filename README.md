<h1> NEM data analysis challenge </h1>

<h2> Phase 1 </h2>

<b>Author</b>: Andreu Mora (andreu@hoberlab.com)

<b>Date</b>: 24/08/17


The task has been split in two subtasks: 

* <b>ETL (Extract Transform Load)</b>: The original file has been processed with Apache Spark on the cloud using Databricks.com and Amazon Web Services. The Python code (using pyspark) has been attached as a notebook in the "ETL" folder, it actually runs in Databricks (.ipynb), and has been exported as well in HTML (.html) and pure python code (.py). This notebook used 5 machines on AWS (1 driver, 5 workers r3.xlarge) and processed the input file in around 20 minutes with a cost of around 4EUR. The output file has been renamed to "dataset_processed.csv".

* <b>EDA (Exploratory Data Analysis)</b>: The output file "dataset_processed.csv" has been processed in a Jupyter Notebook using pandas, matplotlib, scikit-learn. The results and code are in-line and can be explored and executed within the notebook (.ipynb), the notebook has been exported as well in HTML (.html), PDF (.pdf) and Python source code (.py). All the files for EDA are contained in the "EDA" folder.

A report on EDA has been written and attached as well in PDF. The report summarized the figures contained in the EDA notebook, wiht the sole purpose to fit the 5 page requirement.

Contents:
- `README.md`: this file
- `dataset_processed.csv` the output file
- `ETL` folder: contains code for the Extract, Transform, Load analysis.
- `EDA` folder: contains code for the Exploratory Data Analysis
- `EDA_report.pdf`: cointains a 5 page report, which is the summary of the exploration donde in the EDA notebook.

From 25/08/17 onwards these files will be as well available on https://github.com/drublackberry/NEM-data-challenge

<h2> Phase 2 </h2>

<b>Author</b>: Andreu Mora (andreu@hoberlab.com) 
<b>Date</b>: 26/09/17 

The prediction tasks consists in predicting the output power of 5 wind turbines based on historical sensor data and weather forecasts. 

The delivered material is structured as follows: 

* `PRD/src` folder. Contains the notebook with the guided explanations and plots to both understand and process the data. The notebook (.ipynb) can be executed within the right python environment (sklearn, matplotlib, pandas, numpy, scipy). The same notebook has been exported to python code and html. 

* `PRD/data`: the output CSV files for every day that the prediction was needed. It contains three columns: asset, range, pred_WTRUPower. There is one file per day. 

* `PRD_Prediction_Wind_Power_Approach`: report that presents the approach followed and the performance of a Ridge regression 

* `PRD_Prediction_Wind_Power_NeuralNetwork` report showing the performance of a fully-connected neural network for regression. It does not repeat the approach explanation. 

From 1/10/17 onwards these files will be as well available on https://github.com/drublackberry/NEM-data-challenge