#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import kragle as kg
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from sklearn.cluster import KMeans

#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d


# In[2]:


import subprocess
subprocess.run(["java", "-D HADOOP_USER_NAME=ldeek_klclustering"])
os.environ["HADOOP_USER_NAME"] = "ldeek_klclustering"


# In[3]:


SINGLE_DATEINT = 20191206
SINGLE_COUNTRY = 'BR'
COUNTRY_LIMIT = 'WHERE country_iso_code = \'%s\'' % (SINGLE_COUNTRY)
TITLE_RANK_LIMIT = 10000 # Max for BR is 59,678

load_data_query = '''
(
    SELECT
        raw.steering_site_name,
        raw.title_id,
        CAST(raw.playtime_minutes_steered * 1.0 / total.total_playtime_minutes_steered AS DOUBLE) 
            AS playtime_minutes_share,
        raw.rank_by_playtime_minutes_steered,
        raw.country_iso_code,
        raw.country_dateint
    FROM
    (
        SELECT 
            site_meta.steering_site_name,
            site_meta.title_id,
            site_meta.playtime_minutes_steered,
            site_meta.rank_by_playtime_minutes_steered,
            site_meta.country_iso_code,
            site_meta.country_dateint
        FROM 
            ldeek.clustering_steering_site_metadata site_meta
            JOIN ldeek.clustering_similarity_benchmark benchmark ON (
                site_meta.country_iso_code = benchmark.region_code AND
                site_meta.country_dateint = benchmark.country_dateint AND
                site_meta.title_id = benchmark.title_id
            )
        WHERE
            benchmark.country_dateint = %i AND
            benchmark.rank_by_playtime_minutes_steered <= %i AND
            benchmark.region_code = \'%s\'
    ) raw
    JOIN
    (
        SELECT 
            site_meta.steering_site_name,
            SUM(site_meta.playtime_minutes_steered) AS total_playtime_minutes_steered,
            site_meta.country_iso_code,
            site_meta.country_dateint
        FROM 
            ldeek.clustering_steering_site_metadata site_meta
            JOIN ldeek.clustering_similarity_benchmark benchmark ON (
                site_meta.country_iso_code = benchmark.region_code AND
                site_meta.country_dateint = benchmark.country_dateint AND
                site_meta.title_id = benchmark.title_id
            )
        WHERE
            benchmark.country_dateint = %i AND
            benchmark.rank_by_playtime_minutes_steered <= %i AND
            benchmark.region_code = \'%s\'
        GROUP BY 1,3,4
    ) total ON (
        raw.steering_site_name = total.steering_site_name AND
        raw.country_iso_code = total.country_iso_code AND
        raw.country_dateint = total.country_dateint
    )
    GROUP BY 1,2,3,4,5,6
)
UNION
(
    SELECT
        raw.steering_site_name,
        raw.title_id,
        CAST(raw.playtime_minutes_steered * 1.0 / total.total_playtime_minutes_steered AS DOUBLE) 
            AS playtime_minutes_share,
        raw.rank_by_playtime_minutes_steered,
        raw.country_iso_code,
        raw.country_dateint
    FROM
    (
        SELECT
            'benchmark' AS steering_site_name,
            title_id,
            playtime_minutes_steered,
            rank_by_playtime_minutes_steered,
            region_code AS country_iso_code,
            country_dateint
        FROM
            ldeek.clustering_similarity_benchmark
        WHERE
            country_dateint = %i AND
            rank_by_playtime_minutes_steered <= %i AND
            region_code = \'%s\'
    ) raw 
    JOIN
    (
        SELECT
            'benchmark' AS steering_site_name,
            SUM(playtime_minutes_steered) AS total_playtime_minutes_steered,
            region_code AS country_iso_code,
            country_dateint
        FROM
            ldeek.clustering_similarity_benchmark
        WHERE
            country_dateint = %i AND
            rank_by_playtime_minutes_steered <= %i AND
            region_code = \'%s\'
        GROUP BY 1,3,4
    ) total ON (
        raw.steering_site_name = total.steering_site_name AND
        raw.country_iso_code = total.country_iso_code AND
        raw.country_dateint = total.country_dateint
    )
    GROUP BY 1,2,3,4,5,6
)
ORDER BY steering_site_name DESC, title_id DESC
'''

load_data_query = (load_data_query) % (SINGLE_DATEINT, TITLE_RANK_LIMIT,SINGLE_COUNTRY,
                                       SINGLE_DATEINT, TITLE_RANK_LIMIT, SINGLE_COUNTRY,
                                       SINGLE_DATEINT, TITLE_RANK_LIMIT, SINGLE_COUNTRY,
                                       SINGLE_DATEINT, TITLE_RANK_LIMIT, SINGLE_COUNTRY)
print(load_data_query)


# In[4]:


SparkContext().sparkUser()


# In[5]:


conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '14g'),
                                        ('spark.driver.memory', '14g'),
                                        ('spark.memory.fraction', '0.35'),
                                        ('spark.yarn.executor.memoryOverhead', '2000'),
                                        ('spark.driver.maxResultSize', '4g'),
                                        ('spark.kryoserializer.buffer.max', '2047m'), 
                                        ('spark.sql.execution.arrow.enabled', 'false')])


# In[6]:


spark.sparkContext.stop()
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark


# In[7]:


training_data = spark.sql(load_data_query)
print('Extraction query for clustering data info complete.')


# In[8]:


all_data = training_data.select("*").toPandas()


# In[9]:


data = all_data
data.rename({0: 'steering_site_name',
             1: 'title_id',
             2: 'playtime_minutes_share',
             3: 'rank_by_playtime_minutes_steered',
             4: 'country_iso_code',
             5: 'country_dateint'},
            axis=1, inplace=True)


# In[10]:


"""
Print out debugging messages.
"""
print("***Amount of data: ", all_data.shape)
print(all_data.loc[0]) # print(all_data.head(1))


# In[11]:


SITE_NAMES_ARRAY = data.steering_site_name.unique()
print("Unique site count: ", len(SITE_NAMES_ARRAY))


# In[12]:


"""
Create variable to split the dataset into separate steering sites.
"""

# .sort_values(by='title_id',ascending=False)
# .reset_index(drop=True)

SITE_DATA_DICT = {
    site: all_data.loc[data['steering_site_name']==site]
    for site in SITE_NAMES_ARRAY
}


# In[13]:


"""
Define the similarity metrics.
"""

SKEW_KL_ALPHA = 0.99


# Replaces zeros with non-zero values.
def replace_zeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


# Computed as D(p || q) = sum_y p(y) * log(p(y) / q(y))
def kl_divergence(p, q):
    
    # Sum across all title view shares.
    ratio = np.divide(p, q, out=np.zeros_like(p), where=q!=0)
    KL = np.sum(p * np.log(replace_zeroes(ratio)))
    return KL


# Computed as s_alpha(p, q) = D(q || alpha*p + (1-alpha)q) = sum_y q(y) * log(q(y)/(alpha*p(y)+(1-alpha)q(y)))
def skew_divergence(p, q):
    
    # Sum across all title view shares.
    den = (SKEW_KL_ALPHA * p) + (1.0 - SKEW_KL_ALPHA) * q
    ratio = np.divide(q, den, out=np.zeros_like(q), where=den!=0)
    skew = np.sum(q * np.log(replace_zeroes(ratio)))
    return skew


# In[14]:


"""
Compute the similarity metrics across sites.
"""

n = len(SITE_NAMES_ARRAY)
sites_left = SITE_NAMES_ARRAY

site_p_array = []
site_q_array = []
kl_array = []
skew_array = []

kl_metrics = np.zeros((n,n), dtype=float)
skew_metrics = np.zeros((n,n), dtype=float)

# Compute similarity across all sites.
for site_p in SITE_NAMES_ARRAY:
    site_p_index = np.argwhere(SITE_NAMES_ARRAY==site_p).item(0)
    
    # Ignore the diagonal zero values.
    #sites_left = np.delete(sites_left, np.argwhere(sites_left == site_p))
    for site_q in SITE_NAMES_ARRAY: #sites_left:
        site_q_index = np.argwhere(SITE_NAMES_ARRAY==site_q).item(0)
        
        p = SITE_DATA_DICT[site_p].playtime_minutes_share.values
        q = SITE_DATA_DICT[site_q].playtime_minutes_share.values
        
        kl = -1 #kl_divergence(p, q)
        skew = skew_divergence(p, q)
        
        kl_metrics[site_p_index, site_q_index] = kl
        skew_metrics[site_p_index, site_q_index] = skew
        
        site_p_array.append(site_p)
        site_q_array.append(site_q)
        kl_array.append(kl)
        skew_array.append(skew)
        
SIMILARITY_METRICS = pd.DataFrame({'site_p':site_p_array, 
                                   'site_q':site_q_array, 
                                   'kl':kl_array, 
                                   'skew':skew_array})


# In[15]:


# Validate all data exists as it should.
print("site_p total: ", len(SIMILARITY_METRICS.site_p.unique()))
print("site_q total: ", len(SIMILARITY_METRICS.site_q.unique()))

# site_p should only exclude the last index.
a = set(SIMILARITY_METRICS.site_p.values.flatten())
b = set(SITE_NAMES_ARRAY.flatten())
missing = b.difference(a)
if len(missing) > 0:
    print("site_p missing sites: ", missing)
    print("Index should be %i, and is: %i" %
      (len(SITE_NAMES_ARRAY) - 1, np.argwhere(SITE_NAMES_ARRAY==list(missing)[0]).item(0))) # 'gvt.jpa001.norc'

# site_q should only exclude the last index.
a = set(SIMILARITY_METRICS.site_q.values.flatten())
missing = b.difference(a)
if len(missing) > 0:
    print("site_q missing sites: ", missing)
    print("Index should be 0, and is: ", 
      np.argwhere(SITE_NAMES_ARRAY==list(missing)[0]).item(0)) # 'claro-br-net.rbr001.norc'


# In[16]:


"""
Update font for all graphing
"""
font = {'size'   : 10}
plt.rc('font', **font)


# In[17]:


"""
Plot the values to get some understanding
"""
plt.imshow(skew_metrics, cmap='hot', interpolation='nearest')
plt.show()


# In[26]:


ax = sns.heatmap(skew_metrics)
plt.xlabel('Site Indices')
plt.ylabel('Site Indices')
plt.title('Skew Divergence Heatmap')
#plt.show()
plt.savefig('3d_heatmap.png', bbox_inches="tight")


# In[19]:


SITE_NAMES_INT_MAPPING = dict([(y,x+1) for x,y in enumerate(SITE_NAMES_ARRAY)])


# In[20]:


"""
Benchmark global region.
"""

benchmark_site = 'benchmark' # gru001.ix-cr-02 highest streaming site
benchmark_site_metrics = SIMILARITY_METRICS.loc[SIMILARITY_METRICS['site_p']==benchmark_site].reset_index(drop=True)

# Sort sites in descending order of similarity.
benchmark_site_metrics = benchmark_site_metrics.sort_values(by='skew', ascending=False).reset_index(drop=True)
metrics_subset = benchmark_site_metrics#[0:9]

print("Number of site_q: ", len(metrics_subset.site_q))
site_pos = np.arange(len(metrics_subset.site_q))
plt.figure(figsize=(20,5))
plt.bar(site_pos, 
        metrics_subset['skew'], align='center', alpha=0.8)
plt.xticks(site_pos, 
           metrics_subset.site_q, 
           rotation='vertical')
plt.ylabel('Skew divergence metric')
plt.title('Skew in BR against %s' % (benchmark_site))
#plt.savefig('highly_uncorrelated.png', bbox_inches="tight")


# In[29]:


"""
Apply 2D k-means against benchmark and plot the new graph.
"""

benchmark_site = 'benchmark'
kmeans_benchmark = KMeans(n_clusters=5).fit(benchmark_site_metrics['skew'].values.reshape(-1,1))
centroids_benchmark = kmeans_benchmark.cluster_centers_
print(centroids_benchmark)

plt.figure(figsize=(5,5))
site_pos = np.arange(len(benchmark_site_metrics.site_q))
plt.scatter(site_pos,
            benchmark_site_metrics['skew'], 
            c= kmeans_benchmark.labels_.astype(float), 
            s=50, 
            alpha=0.5)
#plt.xticks(site_pos, benchmark_site_metrics['site_q'], rotation='vertical')
plt.ylabel('Skew Divergence')
plt.xlabel('Open Connect Site IDs')
#plt.title('Clustering of Metric in BR against %s' % (benchmark_site))
plt.savefig('2d_clustering.png', bbox_inches="tight")


# In[22]:


"""
Highest streaming site
"""

benchmark_site = 'gru001.ix-cr-02'
ix_site_metrics = SIMILARITY_METRICS.loc[SIMILARITY_METRICS['site_p']==benchmark_site].reset_index(drop=True)

# Sort sites in descending order of similarity.
ix_site_metrics = ix_site_metrics.sort_values(by='skew', ascending=False).reset_index(drop=True)
metrics_subset = ix_site_metrics#[0:200]

site_pos = np.arange(len(metrics_subset.site_q))
plt.figure(figsize=(20,5))
plt.bar(site_pos, 
        metrics_subset['skew'],
        align='center', 
        alpha=0.8)
plt.xticks(site_pos, 
           metrics_subset.site_q, 
           rotation='vertical')
plt.ylabel('Skew divergence metric')
plt.title('Skew in BR against %s' % (benchmark_site))


# In[23]:


"""
Some ISP site
"""

benchmark_site = 'algar.rao001.norc'
isp_site_metrics = SIMILARITY_METRICS.loc[SIMILARITY_METRICS['site_p']==benchmark_site].reset_index(drop=True)

# Sort sites in descending order of similarity.
isp_site_metrics = isp_site_metrics.sort_values(by='skew', ascending=False).reset_index(drop=True)
metrics_subset = isp_site_metrics#[0:200]

site_pos = np.arange(len(metrics_subset.site_q))
plt.figure(figsize=(20,5))
plt.bar(site_pos, 
        metrics_subset['skew'], 
        align='center', 
        alpha=0.8)
plt.xticks(site_pos, 
           metrics_subset.site_q, 
           rotation='vertical')
plt.ylabel('Skew divergence metric')
plt.title('Skew in BR against %s' % (benchmark_site))


# In[24]:


"""
Do a k-means on the divergence metrics
"""

# Map the site names to integer values
site_p_mapping = np.array([SITE_NAMES_INT_MAPPING[x] for x in SIMILARITY_METRICS['site_p']])
site_q_mapping = np.array([SITE_NAMES_INT_MAPPING[x] for x in SIMILARITY_METRICS['site_q']])
skew_values = SIMILARITY_METRICS['skew']

dimensions = pd.DataFrame({'site_p':site_p_mapping, 
                           'site_q':site_q_mapping,
                           'skew':skew_values})

kmeans = KMeans(n_clusters=10).fit(dimensions)
centroids = kmeans.cluster_centers_
print(centroids)


# In[25]:


print(centroids[0][0])

for centroid in centroids:
    a = SITE_NAMES_INT_MAPPING[centroid[0]]
    b = SITE_NAMES_INT_MAPPING[centroid[1]]
    print("Centroid: %s - %s" % (a, b))

    


# In[28]:


"""
3D Points
https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
"""

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for 3D scattered points.
xdata = site_p_mapping
ydata = site_q_mapping
zdata = skew_values
#ax.scatter3D(xdata, ydata, zdata, c=kmeans.labels_.astype(float), s=50, alpha=0.5, cmap='Greens');
ax.scatter3D(xdata, ydata, zdata, c=zdata, s=50, alpha=0.5, cmap='Greens');

# Add the centroids to the scatter plot.
#ax.scatter3D(centroids[:, 0], centroids[:, 1], c='red', s=50)

ax.view_init(60, 35)
plt.xlabel('Site Indices')
plt.ylabel('Site Indices')
#plt.zlabel('Skew Divergence metric')
#plt.show()
plt.savefig('3d_divergence_plot.png', bbox_inches="tight")
#fig


# In[ ]:


"""
3D Surface plot
https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
"""

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_surface(xdata, ydata, zdata, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

#ax.set_title('3D');
ax.view_init(60, 35)
fig


# In[ ]:





# In[ ]:





# In[ ]:




