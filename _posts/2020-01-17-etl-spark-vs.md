---
layout: post
title: "ETL: Spark versus ..."
date: 2020-01-20 17:00
author: pwais
---

This post strives to describe how [Spark](https://spark.apache.org/) functions as an [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) tool and provide citations comparing it to other such tools like Ray, Beam, Dask, etc.

## Spark in a Nutshell

### From MapReduce ... 

Spark is popularly known as an open source implementation of [MapReduce](http://stevekrenzel.com/finding-friends-with-mapreduce).  Historically, MapReduce has been used to build analytics jobs on large, distributed datasets that could not otherwise be queried using an SQL engine. For example, in a database of email addresses, one could compute the top N most frequent email domains (e.g. `yahoo.com`) using a simple SQL query.  If the data were very large and distributed, one could use MapReduce to define a Map of email address to domain (e.g.  `me@yahoo.com` -> `yahoo.com`) and a Reduce that groups results with the same domain and counts them.  In 2004, when MapReduce was first [published](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf), the MapReduce solution to this problem was novel and challenges included tuning performance and generalizing the job to support a variety of queries.  By 2012, Spark and other tools have provided [well-optimized SQL interfaces](https://pages.databricks.com/rs/094-YMS-629/images/1211.6176.pdf) to distributed data.  One can apply [Spark SQL](https://spark.apache.org/docs/2.4.4/sql-programming-guide.html) to the distributed setting above and Spark will effectively auto-generate a MapReduce job for you.  (Spark will even compile the Map string transformation [into highly-optimized bytecode](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html)).

### ...to Interactive Cluster Computing

Spark is much more (and much less) than a MapReduce platform.  At its core, Spark is a distributed execution engine.  

<center><img src="{{site.baseurl}}/assets/images/simple_spark_diagram.png" width="550px" style="border-radius: 8px; border: 8px solid #ddd;" /></center>

When you run a Spark job:

 * Your machine, the driver, gains interactive control of JVM (and/or Python) processes on worker machines.  (Your local machine might host worker processes when running in local mode).
 * Code from the driver is transparently serialized (e.g. via [cloudpickle](https://github.com/cloudpipe/cloudpickle)) and sent as bytes to the workers, who execute the code.  The workers might send back a result, or they might hold the result in worker memory as a distributed dataset.  (The workers can even [spill to local disk](https://spark.apache.org/docs/2.4.4/rdd-programming-guide.html#rdd-persistence)).  Library code can be torrented out to workers using the SparkFiles API (see blogpost on making this automatic).
 * Workers can also write independently (with or without a Reduce step) to a distributed datastore like [AWS S3](https://aws.amazon.com/s3/).  This strategy allows a Spark job to scale network throughput arbitrarily.
 * Workers can join and leave the cluster as one’s job is executing.  When workers leave (or a task fails), Spark will automatically try to re-compute any lost results using the available workers.
 * When dealing with input data that is already distributed (e.g. in [Hadoop HDFS](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html), [Gluster](https://github.com/kubernetes/examples/tree/386580936e2183b72a08a6a964a85143790ba2a2/staging/spark/spark-gluster), etc), Spark will use data locality information to try to co-locate computation with data.  (This is a central feature of MapReduce).
 * Above, we noted that Spark can transparently distribute user code (e.g. functions, classes, and libraries) in a job run.  If one needs Spark to distribute a dockerized runtime environment with user library dependencies, Spark offers [integration with Kubernetes](https://spark.apache.org/docs/2.4.4/running-on-kubernetes.html), which will transparently distribute, cache docker images for, and set up dockerized worker environments on a pet-job basis.  This feature provides one of the easiest and most robust solutions for scaling a complete runtime environment from a single user machine to a large cluster of machines.
 * The overhead of Spark itself?  Spark's task execution latency is largely network-bound, and Spark serializes task results with the most efficient solution available (and for Python uses standard `pickle` and not `cloudpickle`).  Spark carefully manages JVM memory and [affords tuning](https://spark.apache.org/docs/2.4.4/tuning.html#memory-tuning); for Python, memory management affords efficient Spark DataFrame jobs.

## Spark versus ...

### Beam

[Beam](https://beam.apache.org/) seeks to be a software architecture that abstracts away the underlying execution of ETL.  Beam focuses on streaming workflows where [Hadoop](https://hadoop.apache.org/) was originally a de-facto tool: crunching log files that live on disk or that stream directly from a webapp.  Beam is a favorite among Google Cloud sales engineers (via the auto-scaling [Dataflow](https://cloud.google.com/dataflow) product) and Google Research engineers who lack a Python-focused ETL tool due to Google's internal deprecation of MapReduce (public examples: [1](https://github.com/tensorflow/lingvo/blob/ed1f8f899c615e2efc30adf86bdcafdce6df9542/lingvo/tools/beam_utils.py) [2](https://github.com/tensorflow/transform) ).  Beam was first published as [Google Dataflow](http://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf).

Beam offers a [feature matrix](https://beam.apache.org/documentation/runners/capability-matrix/) to roughly compare Beam to others, including Spark.  There are a number of factors that make Spark preferable to Beam.

#### Interactivity as an Extra

To iteratively develop a Beam `Pipeline` in a notebook environment, Beam requires using a [specialized Interactive Runner API](https://github.com/apache/beam/blob/master/sdks/python/apache_beam/runners/interactive/examples/Interactive%20Beam%20Example.ipynb). Cached data from partial execution, if any, is always spilled to disk and requires expensive text-based serialization.  Unlike Spark Dataframes, Beam `PCollection`s are incompatible with [pandas](https://pandas.pydata.org/).  Notebook-based software writing typically helps a user write not only correct code but also discover outlier data cases quickly and interactively.  A Beam user who needs to examine basic statistics (e.g. the range or mean) of a value during a full or partial run of user code must undertake much more effort than a Spark user, who can use either Spark's Dataframe API, or load a piece of data into `pandas` for fast analysis and graphing.

#### Performance 

Beam's core abstractions are effective at isolating the user program from the execution engine... but are these abstractions necessary?  Perhaps yes in some cases, but Beam's abstractions nevertheless appear to impart a considerable performance penalty in [benchmark of common jobs](https://arxiv.org/pdf/1907.08302.pdf).

#### Alpha-level Support For SQL

SQL is often a highly efficient substitute for Python code in ETL jobs: SQL can result in far less code, SQL can be much more readable than a set of Python functions, SQL can be portable to other contexts (e.g. from ETL to a database like [Presto](https://prestodb.io/), [Hive](https://hive.apache.org/), or [BigQuery](https://cloud.google.com/bigquery)), and runtime optimization of SQL queries is both well-studied and built into most engines.  In 2019, Beam provided [alpha-quality SQL support](https://medium.com/weareservian/exploring-beam-sql-on-google-cloud-platform-b6c77f9b4af4) for its Java API **only**.  

#### Per-job Dependencies 

While Beam does offer an affordance for [shipping your own Python library](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) with a job, your library must have a `setup.py`, and the library cannot be updated live during job execution (as can be done easily with [`oarphpy.spark` in a Jupyter Notebook]({{site.baseurl}}{% post_url 2020-01-17-atrick %}))). Unlike Spark, Beam does not offer distribution of arbitrary binary files; the user would need to manually copy the file to [the job's scratch directory on distributed storage](https://beam.apache.org/releases/javadoc/2.4.0/org/apache/beam/sdk/options/PipelineOptions.html#getTempLocation--).  






### Ray

### Dask





** include beam, note Google engineers pushing.  Beam is slow paper:  




# From Notebook


# Spark DataFrame -> Tensorflow Dataset

This notebook serves as a playground for testing `oarphpy.spark.spark_df_to_tf_dataset()`.  See also the unit tests for this utiltiy.


```python
# Common imports and setup
from oarphpy.spark import NBSpark
from oarphpy.spark import spark_df_to_tf_dataset
from oarphpy import util

import os
import random
import sys

import numpy as np
import tensorflow as tf
from pyspark.sql import Row

spark = NBSpark.getOrCreate()
```

    /usr/local/lib/python3.6/dist-packages/google/protobuf/__init__.py:37: UserWarning: Module oarphpy was already imported from /opt/oarphpy/oarphpy/__init__.py, but /opt/oarphpy/notebooks is being added to sys.path
      __import__('pkg_resources').declare_namespace(__name__)
    2019-12-27 21:01:03,560	oarph 336 : Trying to auto-resolve path to src root ...
    2019-12-27 21:01:03,561	oarph 336 : Using source root /opt/oarphpy 
    2019-12-27 21:01:03,589	oarph 336 : Generating egg to /tmp/op_spark_eggs_e2392756-5287-4e0e-bdb3-3bc52ee6cde4 ...
    2019-12-27 21:01:03,641	oarph 336 : ... done.  Egg at /tmp/op_spark_eggs_e2392756-5287-4e0e-bdb3-3bc52ee6cde4/oarphpy-0.0.0-py3.6.egg


## Test on a "large" 2GB random dataset

Create the dataset


```python
NUM_RECORDS = 1000

DATASET_PATH = '/tmp/spark_df_to_tf_dataset_test_large'
def gen_data(n):
  import numpy as np
  y = np.random.rand(2 ** 15).tolist()
  return Row(part=n % 100, id=str(n), x=1, y=y)
rdd = spark.sparkContext.parallelize(range(NUM_RECORDS))
df = spark.createDataFrame(rdd.map(gen_data))
if util.missing_or_empty(DATASET_PATH):
    df.write.parquet(DATASET_PATH, partitionBy=['part'], mode='overwrite')
```


```bash
%%bash -s "$DATASET_PATH"
du -sh $1
```

    2.7M	/tmp/spark_df_to_tf_dataset_test_large


Test reading the dataset through Tensorflow


```python
udf = spark.read.parquet(DATASET_PATH)
print("Have %s rows" % udf.count())
n_expect = udf.count()

ds = spark_df_to_tf_dataset(
        udf,
        'part',
        spark_row_to_tf_element=lambda r: (r.x, r.id, r.y),
        tf_element_types=(tf.int64, tf.string, tf.float64))

n = 0
t = util.ThruputObserver(name='test_spark_df_to_tf_dataset_large')
with util.tf_data_session(ds) as (sess, iter_dataset):
  t.start_block()
  for actual in iter_dataset():
    n += 1
    t.update_tallies(n=1)
    for i in range(len(actual)):
      t.update_tallies(num_bytes=sys.getsizeof(actual[i]))
    t.maybe_log_progress()
  t.stop_block()

print("Read %s records" % n)
assert n == n_expect
```

    Have 10 rows
    getting shards
    10 [1, 6, 3, 5, 9, 4, 8, 7, 2, 0]


    2019-12-27 21:02:21,279	oarph 336 : Reading partition 3 
    2019-12-27 21:02:21,280	oarph 336 : Reading partition 0 
    2019-12-27 21:02:21,281	oarph 336 : Reading partition 1 
    2019-12-27 21:02:21,281	oarph 336 : Reading partition 6 
    2019-12-27 21:02:21,283	oarph 336 : Reading partition 8 
    2019-12-27 21:02:21,284	oarph 336 : Reading partition 4 
    2019-12-27 21:02:21,287	oarph 336 : Reading partition 5 
    2019-12-27 21:02:21,287	oarph 336 : Reading partition 2 
    2019-12-27 21:02:21,288	oarph 336 : Reading partition 7 
    2019-12-27 21:02:21,294	oarph 336 : Reading partition 9 
    2019-12-27 21:02:30,044	oarph 336 : Done reading partition 5, stats:
     Partition 5 [Pid:336 Id:140641112145760]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.73 seconds
    Total thru  786.52 KB
    Rate        90.11 KB / sec
    Hz          0.11456897938921241
    ----------  -------------------
    2019-12-27 21:02:30,047	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  --------------------------
    Thruput
    N thru                   1 (of 10)
    N chunks                 1
    Total time               9.28 seconds
    Total thru               786.52 KB
    Rate                     84.72 KB / sec
    Hz                       0.10771379278642114
    Progress
    Percent Complete         10.0
    Est. Time To Completion  1 minute and 23.55 seconds
    -----------------------  --------------------------
    2019-12-27 21:02:30,049	oarph 336 : 
    Partition 5 [Pid:336 Id:140641112145760]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.73 seconds
    Total thru  786.52 KB
    Rate        90.11 KB / sec
    Hz          0.11456897938921241
    ----------  -------------------
    
    2019-12-27 21:02:30,050	oarph 336 : Done reading partition 4, stats:
     Partition 4 [Pid:336 Id:140641112145704]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.76 seconds
    Total thru  786.52 KB
    Rate        89.8 KB / sec
    Hz          0.11417568922247091
    ----------  -------------------
    2019-12-27 21:02:30,063	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   2 (of 10)
    N chunks                 2
    Total time               9.29 seconds
    Total thru               1.57 MB
    Rate                     169.25 KB / sec
    Hz                       0.2151851417476836
    Progress
    Percent Complete         20.0
    Est. Time To Completion  37.18 seconds
    Latency (per chunk)
    Avg                      4 seconds and 647.16 milliseconds
    p50                      4 seconds and 647.16 milliseconds
    p95                      8 seconds and 820.19 milliseconds
    p99                      9 seconds and 191.13 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,067	oarph 336 : 
    Partition 4 [Pid:336 Id:140641112145704]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.76 seconds
    Total thru  786.52 KB
    Rate        89.8 KB / sec
    Hz          0.11417568922247091
    ----------  -------------------
    
    2019-12-27 21:02:30,140	oarph 336 : Done reading partition 9, stats:
     Partition 9 [Pid:336 Id:140641112142736]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.84 seconds
    Total thru  786.52 KB
    Rate        88.94 KB / sec
    Hz          0.11307729912366592
    ----------  -------------------
    2019-12-27 21:02:30,147	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   3 (of 10)
    N chunks                 3
    Total time               9.37 seconds
    Total thru               2.36 MB
    Rate                     251.83 KB / sec
    Hz                       0.3201814097500565
    Progress
    Percent Complete         30.0
    Est. Time To Completion  21.86 seconds
    Latency (per chunk)
    Avg                      3 seconds and 123.23 milliseconds
    p50                      75.37 milliseconds
    p95                      8 seconds and 363.01 milliseconds
    p99                      9 seconds and 99.69 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,150	oarph 336 : 
    Partition 9 [Pid:336 Id:140641112142736]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.84 seconds
    Total thru  786.52 KB
    Rate        88.94 KB / sec
    Hz          0.11307729912366592
    ----------  -------------------
    
    2019-12-27 21:02:30,167	oarph 336 : Done reading partition 3, stats:
     Partition 3 [Pid:336 Id:140641112114512]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.88 seconds
    Total thru  786.52 KB
    Rate        88.53 KB / sec
    Hz          0.11255886336132385
    ----------  -------------------
    2019-12-27 21:02:30,175	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   4 (of 10)
    N chunks                 4
    Total time               9.39 seconds
    Total thru               3.15 MB
    Rate                     335.05 KB / sec
    Hz                       0.42598915263832476
    Progress
    Percent Complete         40.0
    Est. Time To Completion  14.08 seconds
    Latency (per chunk)
    Avg                      2 seconds and 347.48 milliseconds
    p50                      47.79 milliseconds
    p95                      7 seconds and 902.59 milliseconds
    p99                      9 seconds and 7.61 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,180	oarph 336 : 
    Partition 3 [Pid:336 Id:140641112114512]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.88 seconds
    Total thru  786.52 KB
    Rate        88.53 KB / sec
    Hz          0.11255886336132385
    ----------  -------------------
    
    2019-12-27 21:02:30,191	oarph 336 : Done reading partition 1, stats:
     Partition 1 [Pid:336 Id:140641112114288]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.9 seconds
    Total thru  786.52 KB
    Rate        88.35 KB / sec
    Hz          0.11232812819197387
    ----------  -------------------
    2019-12-27 21:02:30,196	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   5 (of 10)
    N chunks                 5
    Total time               9.4 seconds
    Total thru               3.93 MB
    Rate                     418.21 KB / sec
    Hz                       0.5317270879443539
    Progress
    Percent Complete         50.0
    Est. Time To Completion  9.4 seconds
    Latency (per chunk)
    Avg                      1 second and 880.66 milliseconds
    p50                      20.22 milliseconds
    p95                      7 seconds and 442.16 milliseconds
    p99                      8 seconds and 915.52 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,197	oarph 336 : 
    Partition 1 [Pid:336 Id:140641112114288]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.9 seconds
    Total thru  786.52 KB
    Rate        88.35 KB / sec
    Hz          0.11232812819197387
    ----------  -------------------
    
    2019-12-27 21:02:30,227	oarph 336 : Done reading partition 6, stats:
     Partition 6 [Pid:336 Id:140641112116360]
    ----------  ------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.94 seconds
    Total thru  786.52 KB
    Rate        88.01 KB / sec
    Hz          0.1118984490328969
    ----------  ------------------
    2019-12-27 21:02:30,232	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   6 (of 10)
    N chunks                 6
    Total time               9.43 seconds
    Total thru               4.72 MB
    Rate                     500.18 KB / sec
    Hz                       0.6359411237617654
    Progress
    Percent Complete         60.0
    Est. Time To Completion  6.29 seconds
    Latency (per chunk)
    Avg                      1 second and 572.47 milliseconds
    p50                      25.87 milliseconds
    p95                      6 seconds and 981.74 milliseconds
    p99                      8 seconds and 823.44 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,233	oarph 336 : 
    Partition 6 [Pid:336 Id:140641112116360]
    ----------  ------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  8.94 seconds
    Total thru  786.52 KB
    Rate        88.01 KB / sec
    Hz          0.1118984490328969
    ----------  ------------------
    
    2019-12-27 21:02:30,568	oarph 336 : Done reading partition 8, stats:
     Partition 8 [Pid:336 Id:140641112117200]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.28 seconds
    Total thru  786.52 KB
    Rate        84.78 KB / sec
    Hz          0.10779677057703793
    ----------  -------------------
    2019-12-27 21:02:30,572	oarph 336 : Done reading partition 7, stats:
     Partition 7 [Pid:336 Id:140641112142008]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.28 seconds
    Total thru  786.52 KB
    Rate        84.77 KB / sec
    Hz          0.10777922258165155
    ----------  -------------------
    2019-12-27 21:02:30,606	oarph 336 : Done reading partition 0, stats:
     Partition 0 [Pid:336 Id:140641112145536]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.32 seconds
    Total thru  786.52 KB
    Rate        84.4 KB / sec
    Hz          0.10730318717766127
    ----------  -------------------
    2019-12-27 21:02:30,625	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   7 (of 10)
    N chunks                 7
    Total time               9.81 seconds
    Total thru               5.51 MB
    Rate                     561.01 KB / sec
    Hz                       0.7132781716136398
    Progress
    Percent Complete         70.0
    Est. Time To Completion  4.21 seconds
    Latency (per chunk)
    Avg                      1 second and 401.98 milliseconds
    p50                      31.52 milliseconds
    p95                      6 seconds and 612.41 milliseconds
    p99                      8 seconds and 749.57 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,686	oarph 336 : Done reading partition 2, stats:
     Partition 2 [Pid:336 Id:140641112115520]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.39 seconds
    Total thru  786.52 KB
    Rate        83.74 KB / sec
    Hz          0.10646848954044484
    ----------  -------------------
    2019-12-27 21:02:30,687	oarph 336 : 
    Partition 8 [Pid:336 Id:140641112117200]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.28 seconds
    Total thru  786.52 KB
    Rate        84.78 KB / sec
    Hz          0.10779677057703793
    ----------  -------------------
    
    2019-12-27 21:02:30,691	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   8 (of 10)
    N chunks                 8
    Total time               9.82 seconds
    Total thru               6.29 MB
    Rate                     641.03 KB / sec
    Hz                       0.8150258578040357
    Progress
    Percent Complete         80.0
    Est. Time To Completion  2.45 seconds
    Latency (per chunk)
    Avg                      1 second and 226.95 milliseconds
    p50                      25.87 milliseconds
    p95                      6 seconds and 167.16 milliseconds
    p99                      8 seconds and 660.52 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,693	oarph 336 : 
    Partition 7 [Pid:336 Id:140641112142008]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.28 seconds
    Total thru  786.52 KB
    Rate        84.77 KB / sec
    Hz          0.10777922258165155
    ----------  -------------------
    
    2019-12-27 21:02:30,697	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   9 (of 10)
    N chunks                 9
    Total time               9.82 seconds
    Total thru               7.08 MB
    Rate                     721.1 KB / sec
    Hz                       0.9168247665515971
    Progress
    Percent Complete         90.0
    Est. Time To Completion  1.09 second
    Latency (per chunk)
    Avg                      1 second and 90.72 milliseconds
    p50                      20.22 milliseconds
    p95                      5 seconds and 721.92 milliseconds
    p99                      8 seconds and 571.47 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,699	oarph 336 : 
    Partition 0 [Pid:336 Id:140641112145536]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.32 seconds
    Total thru  786.52 KB
    Rate        84.4 KB / sec
    Hz          0.10730318717766127
    ----------  -------------------
    
    2019-12-27 21:02:30,704	oarph 336 : Progress for 
    spark_tf_dataset [Pid:336 Id:140639257511920]
    -----------------------  ---------------------------------
    Thruput
    N thru                   10 (of 10)
    N chunks                 10
    Total time               9.82 seconds
    Total thru               7.87 MB
    Rate                     801.13 KB / sec
    Hz                       1.0185824147813816
    Progress
    Percent Complete         100.0
    Est. Time To Completion  0 seconds
    Latency (per chunk)
    Avg                      981.76 milliseconds
    p50                      16.82 milliseconds
    p95                      5 seconds and 276.68 milliseconds
    p99                      8 seconds and 482.43 milliseconds
    -----------------------  ---------------------------------
    2019-12-27 21:02:30,705	oarph 336 : 
    Partition 2 [Pid:336 Id:140641112115520]
    ----------  -------------------
    Thruput
    N thru      1
    N chunks    1
    Total time  9.39 seconds
    Total thru  786.52 KB
    Rate        83.74 KB / sec
    Hz          0.10646848954044484
    ----------  -------------------
    


    Read 10 records



```python

```