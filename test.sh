#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /p1/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /p1/output/

/usr/local/hadoop/bin/hdfs dfs -mkdir -p /p1/input/train/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /p1/input/test/

/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../test-data/proj2p1_data/train.csv /p1/input/train/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../test-data/proj2p1_data/test.csv /p1/input/test/

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./p1.py ../../test-data/proj2p1_data/train.csv ../../test-data/proj2p1_data/test.csv

/usr/local/hadoop/bin/hdfs dfs -rm -r /p1/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /p1/output/
