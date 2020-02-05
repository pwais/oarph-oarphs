---
layout: post
title: "ETL: Spark versus ..."
date: 2020-02-03 7:00
author: pwais
---

This post describes how [Spark](https://spark.apache.org/) functions as an [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) tool and compares Spark to other tools such as Ray, Beam, and Dask.  This post strives to be a living document.

# Spark in a Nutshell

## From MapReduce ... 

Spark is popularly known as an open source implementation of [MapReduce](http://stevekrenzel.com/finding-friends-with-mapreduce).  MapReduce was designed as a solution for running analytics on large, *distributed* datasets.  

Suppose we have a table of email addresses and we wish to compute the top 10 most frequent domains (e.g. `yahoo.com`) in the table.  If the table lives in an SQL database, we can write a simple SQL query with a `GROUP BY` to solve the problem.  But if the table is very large and stored across several machines, we'd need a MapReduce job: a Map from email address to domain (e.g.  `me@yahoo.com` -> `yahoo.com`) and a Reduce that groups extracted domains and counts them.  

In 2004, when MapReduce was first [published](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf), the MapReduce solution proposed above was novel.  Research on MapReduce focused on tuning performance and support for generic query execution (to obviate the need to write explicit `Map`s and `Reduce`s).  By 2012, Spark and other tools had provided [well-optimized SQL interfaces](https://pages.databricks.com/rs/094-YMS-629/images/1211.6176.pdf) to distributed data.  One could apply [Spark SQL](https://spark.apache.org/docs/2.4.4/sql-programming-guide.html) to the distributed setting described above and Spark would effectively auto-generate a MapReduce job for you.  (Today, Spark will compile the `Map` and `Reduce` operations [into highly-optimized bytecode](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html) at query time).

## ...to Interactive Cluster Computing

Spark is much more than a MapReduce platform.  At its core, Spark is a distributed execution engine:  

<center><img src="{{site.baseurl}}/assets/images/simple_spark_diagram.png" width="550px" style="border-radius: 8px; border: 8px solid #ddd;" /></center>

When you run a Spark job:

 * Your machine, the **Driver**, gains interactive control of JVM (and Python) processes on cluster **Worker** machines.  (Your local machine might host **Worker** processes when running in **Local Mode**).
 * Code from the Driver is transparently serialized (e.g. via [cloudpickle](https://github.com/cloudpipe/cloudpickle)) and sent as bytes to the Workers, who deserialize and execute the code.  The Workers might send back results, or they might simply hold resulting data in Worker memory as a distributed dataset.  (The Workers can also [spill to local disk](https://spark.apache.org/docs/2.4.4/rdd-programming-guide.html#rdd-persistence)).  The user's library code (e.g. Java JARs and Python packages) can be torrented out to Workers through the **SparkFiles** API (and usage of this API [can be automated using `oarphpy`]({{site.baseurl}}{% post_url 2020-02-03-atrick %})).
 * Each Worker can independently read and write to a distributed datastore like [AWS S3](https://aws.amazon.com/s3/).  This many-to-many strategy allows a job to achieve arbitrarily high levels of total I/O.
 * Workers can join and leave the cluster as one’s job is executing.  When workers leave (or a task fails), Spark will automatically try to re-compute any lost results using the available Workers.
 * When dealing with input data that is already distributed (e.g. in [Hadoop HDFS](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html), [Gluster](https://github.com/kubernetes/examples/tree/386580936e2183b72a08a6a964a85143790ba2a2/staging/spark/spark-gluster), etc), Spark will use data locality information to try to co-locate computation with data.  (This is a central feature of MapReduce: move computation to the data).
 * Above, we noted that Spark can transparently distribute user code (e.g. functions, classes, and libraries) with each job run.  If one needs to distribute a [dockerized runtime environment](https://docs.docker.com/get-started/) with user dependencies, Spark offers [integration with Kubernetes](https://spark.apache.org/docs/2.4.4/running-on-kubernetes.html), which will transparently distribute, cache docker images for, and set up dockerized Workers on a per-job basis.  This feature provides one of the easiest and most robust solutions for scaling a complete runtime environment from a single machine to a cluster.
 * The overhead of Spark itself?  Spark's task execution latency adds mere milliseconds to network latency, and Spark serializes task results with the most efficient solution available (and for Python uses standard `pickle` and not `cloudpickle`).  Spark carefully manages JVM memory, reserving some for user code and some for buffering intermediate results.  (The user can [tune](https://spark.apache.org/docs/2.4.4/tuning.html#memory-tuning) these settings on a per-job basis).  



# Spark versus ...


## Beam

[Beam](https://beam.apache.org/) seeks to be a software architecture that abstracts away the underlying execution of an ETL job.  Beam focuses on streaming workflows where [Hadoop](https://hadoop.apache.org/) was once the de-facto tool: crunching log files using custom `Map`s and `Reduce`s.  Beam offers a modern take and offers specialized support for streaming sources-- e.g. logs straight from a webapp or [Kafka cluster](https://kafka.apache.org/).  Beam is a favorite among Google Cloud sales engineers (i.e. their auto-scaling [Dataflow](https://cloud.google.com/dataflow) offering) and Google Research engineers who lack a Python-focused ETL tool due to Google's internal deprecation of MapReduce (public examples: [1](https://github.com/tensorflow/lingvo/blob/ed1f8f899c615e2efc30adf86bdcafdce6df9542/lingvo/tools/beam_utils.py) [2](https://github.com/tensorflow/transform) ).  Beam was first published as [Google Dataflow](http://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf).

Beam offers a [feature matrix](https://beam.apache.org/documentation/runners/capability-matrix/) to roughly compare it to others, including Spark.  There are a number of factors that make Spark preferable to Beam for ETL.

### Beam's Interactivity is an Extra

To iteratively develop a Beam `Pipeline` in a notebook environment, Beam requires using a [specialized Interactive Runner API](https://github.com/apache/beam/blob/master/sdks/python/apache_beam/runners/interactive/examples/Interactive%20Beam%20Example.ipynb). Cached data from partial execution, if any, is always spilled to disk and requires expensive text-based serialization.  Unlike Spark Dataframes, Beam `PCollection`s are incompatible with [pandas](https://pandas.pydata.org/).  

Notebook-based iterative development typically helps a user write not only correct code but also helps the user discover outlier data cases quickly and interactively.  A Beam user who simply needs working code might be fulfilled with the current offering.  However, a Beam user who needs to examine basic statistics (e.g. the range or mean) of a value during a full or partial run of a `Pipeline` must undertake much more effort than a Spark user, who can use either Spark's Dataframe API, or load a piece of data into `pandas` for fast analysis and graphing.

### Performance 

Beam's core abstractions are effective at isolating the user program from the execution engine... but at what cost?  These abstractions appear to impart a considerable performance penalty in [benchmark of common jobs](https://arxiv.org/pdf/1907.08302.pdf).

### Alpha-level Support For SQL

SQL is often a highly efficient substitute for Python in ETL jobs: SQL can result in far less code, SQL can be much more readable than a set of Python functions, SQL can be portable to other contexts (e.g. queries can also run in [Presto](https://prestodb.io/), [Hive](https://hive.apache.org/), or [BigQuery](https://cloud.google.com/bigquery)), and runtime optimization of SQL queries is both well-studied and built into most engines.  In 2019, Beam provided [alpha-quality SQL support](https://medium.com/weareservian/exploring-beam-sql-on-google-cloud-platform-b6c77f9b4af4) for its Java API *only*.  

### Per-job Dependencies 

While Beam does offer an affordance for [shipping your own Python library](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/) with a job, your library must have a `setup.py`, and the library cannot be updated live during job execution (as can be done easily with [`oarphpy.spark` in a Jupyter Notebook]({{site.baseurl}}{% post_url 2020-02-03-atrick %}))). Unlike Spark, Beam does not offer distribution of arbitrary binary files; the user would need to manually copy the file to [the job's scratch directory on distributed storage](https://beam.apache.org/releases/javadoc/2.4.0/org/apache/beam/sdk/options/PipelineOptions.html#getTempLocation--).  



## Ray

Before Spark grew into an ETL tool, the authors sought to build a general library for distributed systems using the [Akka Actor](https://doc.akka.io/docs/akka/current/typed/actors.html) model.  Akka today serves as Spark's networking layer and powers Driver-Worker RPC as well as fault tolerance.  

[Ray](https://github.com/ray-project/ray) began with similar aspirations and also leverages an [actor-based model](https://ray.readthedocs.io/en/latest/fault-tolerance.html) for fault tolerance and distributing compute.  Ray, like Spark, also originally targeted use cases in Machine Learning.  While Spark includes its own built-in distributed storage engine ([BlockManager](https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/storage/BlockManager.scala)) for RDDs and DataFrames, Ray relies on [Redis](https://redis.io/) to provide per-job distributed storage.

While Ray can indeed run [MapReduce jobs](https://github.com/ray-project/tutorial/blob/e7dc111be6889222fa665187f02fc132bea9928e/examples/map_reduce.ipynb), Ray lacks a few key features to make it a viable ETL tool:
 * **Data Locality**: Ray does not try to co-locate computation with data, so it cannot function as an efficient general MapReduce or database solution.

 * **DataFrames and SQL**: Ray has nascent DataFrame support, which is now part of the [Modin](https://github.com/modin-project/modin) project.  Support for SQL on data is currently in only a roughly planned state.

 * **Distributing Artifacts**: Like Spark, Ray leverages `cloudpickle` for distributing user code to workers.  However, there's no affordance for distributing a user library with an individual job.  The user must have a shared filesystem / temp store available, or must somehow leverage Ray's central Redis-powered [distributed object store](https://ray.readthedocs.io/en/latest/walkthrough.html#objects-in-ray).

Ray's strengths are in reinforcement learning and hyperparameter search: two use cases that are heavy in computation and light on data.  



## Dask

[Dask](https://dask.org/) seeks to offer the most performant multi-machine solutions for common Dataframe- and Matrix-based tasks in Data Science.  While Spark supports similar use cases through Spark Dataframes as well as [Spark MLlib](https://spark.apache.org/mllib/), Spark's APIs are bespoke while Dask builds behind existing Pandas and [Scikit-learn](https://scikit-learn.org/) APIs. 

While Dask might be an effective tool for data analysis itself, Dask lacks a few key features important to ETL:

 * **SQL** Dask can [read and write from standard databases](https://github.com/dask/dask/blob/master/dask/dataframe/io/sql.py) (as can Spark [through JDBC](https://spark.apache.org/docs/2.4.4/sql-data-sources-jdbc.html)), but Dask can't perform SQL on the data backing a DataFrame. One must use the Pandas API (which can make group aggregations and joins difficult to express).

 * **Data Locality**: Dask is [locality-aware](https://distributed.dask.org/en/latest/locality.html) once you get data into it, but it does not support HDFS-compatible distributed datastores as does Spark and most of the Hadoop ecosystem.

 * **Distributing Code**: Dask [Bags](https://docs.dask.org/en/latest/bag.html) support the same sort of `cloudpickle`-powered computation possible with Spark RDDs, but Dask lacks a facility for distributing custom user libraries.

That said, Dask offers a few features that Spark lacks:
 * No JVM required
 * Inclusion of [futures](https://docs.dask.org/en/latest/futures.html), which sometimes preferable for CPU-heavy use cases (e.g. jobs one might otherwise run on Ray).
 * Dask's [`delayed` API](https://docs.dask.org/en/latest/delayed.html) can make it [easier to run](https://matthewrocklin.com/blog/work/2017/03/22/dask-glm-1) distributed [stochastic gradient descent (SGD)](https://dask-glm.readthedocs.io/en/latest/_modules/dask_glm/algorithms.html#gradient_descent).  While Spark has similar support for parallel SGD, the RDD API complicates synchronization.

