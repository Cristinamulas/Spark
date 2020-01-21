# Spark

- Program Overview

• This program is built following the "Simple Text Classification with Apache Spark" on https://www.kaggle.com/bkarabay/simple-text-classification-with-apache-spark. The goal was mainly to convert the column comment text into a column of sparse vectors for use in a classification algorithm in the spark ml library. 

- Library Used
• The pyspark.ml library is the key point used for machine learning with Spark DataFrames.


- Spark Context and Load Data

Spark context was built,and training and testing data were loaded.

- Sparse Vector Conversion

Basic sentence tokenizer was used, and then count the words in a document as raw features using HashingTF. Then build the idf model and transform the original token frequencies into their tf-idf counterparts, thus the features were created.

- Model Training

 The model was built using "features" and "toxic" on training data.
 
 - Results DataFrame Creation
 
 The testing data was converted in same context as training data; and then make predictions for each class and output in a data 
fr
