import streamlit as st
import findspark
findspark.init()

import os
import pyspark
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext,Row

from pyspark.sql.functions import regexp_replace, trim, col, lower, concat_ws,udf,when
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer,StopWordsRemover
from pyspark.sql.functions import *

from pyspark.ml import *
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.param import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import rand 
from time import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row

from transformers import pipeline

conf = SparkConf().setMaster("local[*]").setAppName("streamlit-sentiment-prediction-marianmt")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

st.text_area("Yorum giriniz:", key="yorum")

# You can access the value at any point with:
girdi = st.session_state.yorum

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-tr-en")

eng_sentence = pipe(girdi)[0]['translation_text']

eng_sentence

d = [{'reviewText': eng_sentence}]
df = spark.createDataFrame(d)

log_reg = PipelineModel.load("./models/base_log_reg_model_248/")

model_preditions=log_reg.transform(df)

model_preditions = model_preditions.withColumn("Sentiment_Prediction", when(col("prediction") =='0','Negative Review').otherwise('Positive Review'))

aaa = model_preditions.select("Sentiment_Prediction","probability").take(1)
aaa
