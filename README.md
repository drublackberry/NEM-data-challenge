** NEM data analysis challenge**
** Phase 1 **

Author: Andreu Mora (andreu@hoberlab.com)
Date: 24/08/17

The task has been split in two subtasks: 

* 1) <b>ETL (Extract Transform Load)</b>: The original file has been processed with Apache Spark on the cloud using Databricks.com and Amazon Web Services. The Python code (using pyspark) has been attached as a notebook in the "ETL" folder, it actually runs in Databricks (.ipynb), and has been exported as well in HTML (.html) and pure python code (.py). This notebook used 5 machines on AWS (1 driver, 5 workers r3.xlarge) and processed the input file in around 20 minutes with a cost of around 4EUR. The output file has been renamed to "dataset_processed.csv".

* 2) <b>EDA (Exploratory Data Analysis)</b>: The output file "dataset_processed.csv" has been processed in a Jupyter Notebook using pandas, matplotlib, scikit-learn. The results and code are in-line and can be explored and executed within the notebook (.ipynb), the notebook has been exported as well in HTML (.html), PDF (.pdf) and Python source code (.py). All the files for EDA are contained in the "EDA" folder.

A report on EDA has been written and attached as well in PDF. The report summarized the figures contained in the EDA notebook, wiht the sole purpose to fit the 5 page requirement.

Contents:
- "README.md": this file
- "dataset_processed.csv" the output file
- "ETL" folder: contains code for the Extract, Transform, Load analysis.
- "EDA" folder: contains code for the Exploratory Data Analysis
- "EDA_report.pdf": cointains a 5 page report, which is the summary of the exploration donde in the EDA notebook.

From 25/08/17 onwards these files will be as well available on https://github.com/drublackberry/NEM-data-challenge