# Ben Noriega
# CS777
# 02.22.23
# Term Project

import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession 

if __name__ == "__main__":
    print("Starting Term Project")
    print()

    sc = SparkContext(appName="Assignment-5")
    spark = SparkSession.builder.getOrCreate()

    corpus = sc.textFile(sys.argv[1], 1)
    print(corpus.take(1))