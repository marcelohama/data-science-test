# data-science-sample

Some samples using python3 to do analytics with PySpark+Docker.

## Pre-Requisites

- OSX 11.7.10 (Big Sur)
- Docker Desktop v4.24.0

## Steps to Initialize

PYSPARK

	$ docker run -p 8888:8888 jupyter/pyspark-notebook

	http://127.0.0.1:8888/?token=<token>

NOTEBOOK PYTHON

	import pyspark
	
	sc = pyspark.SparkContext('local[*]')
	
	txt = 'Hello PySpark'
	
	print(txt)
