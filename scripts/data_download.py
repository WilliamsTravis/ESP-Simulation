"""
Download sample forecasts and observations from the Colorado Basin River
    Forecast Center.

Created on Tue Oct 23 10:12:22 2018

@author: User
"""
import os
import pandas as pd
import requests
import sys
from tqdm import tqdm
from bs4 import BeautifulSoup

if 'scripts' not in sys.path:
    sys.path.insert(0, 'scripts')
os.chdir('c:\\users\\user\\github\\ESP_simulation')  # How to generalize this?

# In[]
# Forecasts and observations from sample forecast sites
# In thousand acre feet
sites = ['MPHC2', 'DOLC2', 'CAMC2', 'DOLU1']  # add site abbr. to get new data
urls = [["https://www.cbrfc.noaa.gov/wsup/graph/espplot_data.py?id=" +
         site + "&year=" + str(y) +
         "&bper=7&eper=10&qpf=0&quantiles=1&observed=1&average=1" for
         y in range(2013, 2019)] for site in sites]
urls = [u for su in urls for u in su]
labels = [[site + "_" + str(y) for y in range(2013, 2019)] for site in sites]
labels = [l for sl in labels for l in sl]

# Build a list of data frames and names for saving
dfs = []
for url in tqdm(urls, position=0):
    r = requests.get(url)
    content = BeautifulSoup(r.text, 'html.parser').text
    rows = [x for x in content.split('\n')]  # List of Lists
    header = rows[0].split(",")
    rows = [r.split(",") for r in rows[1:]]
    df = pd.DataFrame(rows)
    df.columns = header
    dfs.append(df)

# Write to file
for i in range(len(dfs)):
    filename = labels[i]
    dfs[i].to_csv("data\\" + filename + ".csv", index=False)


# In[] Historical Volumes for MPHC2 and DOLC2
urls = ["https://www.cbrfc.noaa.gov/rmap/wsup/point.php?rfc=cbrfc&id=" +
        site +
        "&wyear=2017&mode=obs&qpf=0&showesp=1&showunapp=0&showoff=1&" +
        "showobs=1&showmm=1&showhvol=0&mode=historical" for site in sites]
labels = [site + "_historical" for site in sites]

for i in range(len(urls)):
    r = requests.get(urls[i])
    content = BeautifulSoup(r.text, 'html.parser')
    pre = content.find('pre').text
    rows = [x for x in pre.split('\n')]
    header = rows[0].split()
    header.insert(0, 'index')
    rows = rows[1:]
    rows = [r.split() for r in rows]
    df = pd.DataFrame(rows)
    df.columns = header
    df = df.drop(['index'], axis=1)
    df = df.dropna()
    df['year'] = df['year'].astype(int)
    df['vol'] = df['vol'].astype(float)
    df = df.sort_values('year')
    df.to_csv("data\\" + labels[i] + ".csv", index=False)
