# Spark

Program Overview

• This program is built following the "Simple Text Classification with Apache Spark" on https://www.kaggle.com/bkarabay/simple-text-classification-with-apache-spark. The goal was mainly to convert the column comment text into a column of sparse vectors for use in a classification algorithm in the spark ml library. 

Library Used
• The pyspark.ml library is the key point used for machine learning with Spark DataFrames.


Spark Context and Load Data

Spark context was built,and training and testing data were loaded.

Sparse Vector Conversion

