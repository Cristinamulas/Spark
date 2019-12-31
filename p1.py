#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
import sys


reload(sys)
sys.setdefaultencoding('utf8')
hc = SparkSession.builder.appName("Toxic Comment Classification").getOrCreate()

def to_spark_df(fin):
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    df = hc.createDataFrame(df)
    return(df)

train = to_spark_df(sys.argv[1])
test = test = to_spark_df(sys.argv[2])

out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]

print('1============================================================================================')
train.show(5)
print('1============================================================================================')

print('2============================================================================================')
train.filter(F.col('toxic') == 1).show(5)
print('2============================================================================================')

tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
wordsData = tokenizer.transform(train)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
tf = hashingTF.transform(wordsData)

tf.select('rawFeatures').take(2)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf) 
tfidf = idfModel.transform(tf)

tfidf.select("features").first()

REG = 0.1

lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)

print('5============================================================================================')
tfidf.show(5)
print('5============================================================================================')

lrModel = lr.fit(tfidf.limit(5000))

res_train = lrModel.transform(tfidf)

print('6============================================================================================')
res_train.select("id", "toxic", "probability", "prediction").show(20)
print('6============================================================================================')

print('7============================================================================================')
res_train.show(5)
print('7============================================================================================')

extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())

print('8============================================================================================')
res_train.withColumn("proba", extract_prob("probability")).select("proba", "prediction").show()
print('8============================================================================================')

test_tokens = tokenizer.transform(test)
test_tf = hashingTF.transform(test_tokens)
test_tfidf = idfModel.transform(test_tf)

test_res = test.select('id')

test_res.show(5)

test_probs = []
for col in out_cols:
    print(col)
    lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
    print("...fitting")
    lrModel = lr.fit(tfidf)
    print("...predicting")
    res = lrModel.transform(test_tfidf)
    print("...appending result")
    test_res = test_res.join(res.select('id', 'probability'), on="id")
    print("...extracting probability")
    test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
    test_res.show(5)

print('10===========================================================================================')
test_res.show(5)
print('10===========================================================================================')


