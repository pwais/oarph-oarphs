---
layout: post
title: "A Trick for Rapid Iteration with Spark"
date: 2020-01-20
author: pwais
---

## ETL and Spark

Data wrangling is a key component of any machine learning or data science pipeline.  These pipelines <b>e</b>xtract raw data from logs or databases, <b>t</b>ransform the data into some desired format, and then <b>l</b>oad the results into a new data store or directly into a model.  ETL might be as little as a line of code or as much as a full-fledged software library.  Indeed, ETL is a key source of [technical debt](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) in machine learning systems; moreover, the success of project can sometimes depend on [how quickly engineers can iterate](http://www.catb.org/~esr/writings/cathedral-bazaar/cathedral-bazaar/ar01s04.html) on ETL code and leverage data available to them.

[Spark](https://spark.apache.org/) is one of many tools for ETL (and might be [one of the best]({% post_url 2020-01-20-python-etl-leaderboard %})).  Spark is popularly known as distributed computing tool, as it is one of several open source implementations of the [MapReduce paradigm](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf).  Indeed, parallel computing is key for ETL: executing a job on 10 cores often makes the job 10x faster.  

One differentiator between Spark and other tools is that Spark supports *interactive* ETL (e.g. in [Jupyter](https://jupyter.org/) notebooks) via [cloudpickle](https://github.com/cloudpipe/cloudpickle), an advanced code-centric `pickle` library for Python.  The `cloudpickle` package supports serializing `lambda` functions, code defined in the `__main__` module (e.g. a Jupyter notebook kernel or Python interpreter session), and dynamically generated (sub)classes.  When you run a Spark program, part of your code is serialized via `cloudpickle` in the local driver Python process, sent to (likely remote) worker Python processes, deserialized there and run.

But what if you want to use a library of semi-static Python code in your job?  What if you want to make local changes to that library (e.g. pre-commit modifications) and run that modified code in your Spark job?  Spark provides a [`pyFiles`](https://spark.apache.org/docs/latest/configuration.html#runtime-environment) configuration option for including Python Egg(s) with a job.  (Spark even provides caching and torrent-based distribution for these per-job data files).  However, traditional use of the `pyFiles` feature requires user configuration and/or a custom build step.  This article demonstrates how to *automatically* include with a Spark job the library containing the calling code using `oarphpy`.


## Automatic User Library Inclusion in Spark



`oarphpy.spark` provides a utility that does the following:
 * 



[link](test2.md)

```
block
```

```python
def foo():
  return 5
```

```c++
struct moof {
  int a;
};
```

<table>
    <tr>
        <td>Foo</td>
    </tr>
</table>

