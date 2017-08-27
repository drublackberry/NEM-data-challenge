# Databricks notebook source
# mount data from S3
ACCESS_KEY="MY_ACCESS_KEY"
SECRET_KEY="MY_SECRET_KEY"
ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")
AWS_BUCKET_NAME = "hoberlab"
MOUNT_NAME = "hoberlab"
# next command just needed on the first run
dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)

# Load the data into a DF
file_to_load = "/mnt/hoberlab/nem/table_data"  # could be a directory and it is interpreted as one file
# on local filestore "/FileStore/tables/zu62a9pc1500284962393/output_1.csv"
df = sqlContext.read.format("csv").options(delimiter=',', header='true',inferschema='true').load(file_to_load)
display(df)

# COMMAND ----------

from pyspark.sql.functions import date_format, col, concat, udf
from pyspark.sql.types import TimestampType
from datetime import datetime

# covert the time into single datetime
strptime_udf = udf(lambda x: datetime.strptime(x, "%Y-%m-%d%H:%M:%S"), TimestampType())
df_time = df.select(col("asset"), col("variable"), col("value"), strptime_udf(concat(date_format(col("date"), "YYYY-MM-dd"), col("time"))).alias('timestamp')).cache()

# COMMAND ----------

# NOTE: this cell is expensive to run (AWS + DBU costs) and provides a check only (disabled by default)

#from pyspark.sql.functions import min, max
## explore the data
#print ("Number of different assets = {}".format(df_time.select("asset").distinct().count()))
#print ("Number of different variables = {}".format(df_time.select("variable").distinct().count()))
#t_df = df_time.select("timestamp").distinct().cache()
#print ("Number of different timestamps = {}".format(t_df.count()))
#print ("Start time = {}".format(t_df.select(min("timestamp").alias('t_min')).take(1)[0].t_min))
#print ("Stop time = {}".format(t_df.select(max("timestamp").alias('t_max')).take(1)[0].t_max))

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import avg, second, minute

# perform moving average on window
win = Window.partitionBy("asset").partitionBy("variable").orderBy("timestamp").rowsBetween(-9,0)  # timestamp of each interval is the superior limit by guidelines
df_time_ma = df_time.select("asset", "variable", "timestamp", avg("value").over(win).alias("ma")).cache()

# COMMAND ----------

# visualize that moving average performed well
import matplotlib.pyplot as plt
from pyspark.sql.functions import dayofyear, hour

the_asset = df_time_ma.select('asset').distinct().take(1)[0].asset
the_variable = df_time_ma.select('variable').distinct().take(1)[0].variable
the_day = (sorted([x.dayofyear for x in (df_time_ma.select(dayofyear('timestamp').alias('dayofyear')).distinct().take(5))]))[0] # min doesn't work on javalist
the_hours = 12
the_title = ("Data for asset {}, variable {} for day {}, {} hours".format(the_asset, the_variable, the_day, the_hours))

# currently exports to pandas for visualization and export in CSV format, later on the pyspark dataframe is exported in CSV
test_df = df_time_ma.filter(df_time_ma.asset==the_asset).filter(df_time_ma.variable==the_variable).filter(dayofyear('timestamp')==the_day).filter(hour('timestamp')<=the_hours).cache()
test_df_1s = test_df.toPandas()
test_df_60s = test_df.filter(second(df_time_ma.timestamp)==0).toPandas()
test_df_10m = test_df.filter(minute(df_time_ma.timestamp)%10==0).filter(second(df_time_ma.timestamp)==0).toPandas()

plt.figure(figsize=(12,4))
plt.plot(test_df_1s.timestamp, test_df_1s.ma, 'b')
plt.plot(test_df_60s.timestamp, test_df_60s.ma, 'r')
plt.plot(test_df_10m.timestamp, test_df_10m.ma, 'g')
plt.grid()
plt.title(the_title)
plt.legend(['1s', '60s', '10m'])
display(plt.gcf())

# COMMAND ----------

from itertools import chain
from pyspark.sql.functions import create_map, lit, round
from bs4 import BeautifulSoup  # has been imported using PyPi

# load translation dictionary
def import_translations(filehandler):
    soup = BeautifulSoup(filehandler, "html5lib")
    translations = soup.assets.findAll('translation')
    return {t.src.string:t.dest.string for t in translations}

# copy from dbfs into worker local disk
dbutils.fs.cp("dbfs:/mnt/hoberlab/nem/translations.xml", "file:/tmp/translations.xml")
with open("/tmp/translations.xml",'r') as f:
    translate_dict = import_translations(f)

# filter for 10 minute sampling
sdf = df_time_ma.filter(second(df_time_ma.timestamp)==0).filter(minute(df_time_ma.timestamp)%10==0)  # use df on 10 minutes interval

# add identifier
translate_map = create_map([lit(x) for x in chain(*translate_dict.items())])
sdf = sdf.withColumn("identifier", translate_map.getItem(col("variable")))

# final selection of columns with timestamp format
strftime_udf = udf(lambda x: datetime.strftime(x, "%Y-%m-%d %H:%M:%S"))
sdf = sdf.select('asset', concat('identifier', lit('_'), 'asset').alias('id'), strftime_udf(col('timestamp')).alias('timestamp'), round('ma',6).alias('value'))

# pivoting for desired format
sdf = sdf.groupby('timestamp').pivot('id').min().orderBy('timestamp')  # use min, max is irrelevant since only one item
display(sdf)

# write to csv onto S3 bucket, force 1 file
sdf.coalesce(1).write.csv("/mnt/hoberlab/nem/output", header=True)

# COMMAND ----------


