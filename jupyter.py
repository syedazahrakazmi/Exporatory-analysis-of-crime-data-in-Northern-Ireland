#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().system('pip install pysal==2.7.0')


# In[ ]:


get_ipython().system('pip install spreg')


# In[ ]:


get_ipython().system('pip install mapclassify')


# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


get_ipython().system('pip install numpy')


# In[ ]:


get_ipython().system('pip install fiona')


# In[26]:


get_ipython().system(' pip install geopandas')


# In[ ]:


get_ipython().system('pip install fiona')


# In[ ]:


get_ipython().system('pip install spreg')


# In[ ]:


get_ipython().system('pip install pysal spreg')


# In[ ]:


get_ipython().system('pip install spreg')


# In[28]:


import os
import pandas as pd
import numpy as np
import fiona
import shapely
from shapely.geometry import Point, Polygon
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
from geopandas import GeoDataFrame
from scipy import stats
from IPython.display import IFrame
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster
import pysal
from pysal.spreg import ols
from pysal.spreg import ml_error
from pysal.spreg import ml_lag


# In[ ]:


#Read the data


# In[9]:


imd=pd.read_csv('C:\\Users\\zahra\\Desktop\\programming_projectpp\\crime_data_NI.csv')
# use TextFileReader iterable with chunks of 100,000 rows.
tp = pd.read_csv('C:\\Users\\zahra\\Desktop\\programming_projectpp\\crime_data_NI.csv', iterator=True, chunksize=100000)  
crime_data = pd.concat(tp, ignore_index=True)  
# print data shape
crime_data.shape


# In[10]:


imd.describe()


# In[ ]:


#Data cleaning for Crime Data


# In[11]:


# view the basic stats on columns
crime_data.info()


# In[12]:


crime_data.drop(['Context', 'Falls within','Reported by',  'Location','Last outcome category' ], inplace=True, axis=1)
# show dataframe columns
crime_data.columns


# In[13]:


# Now check the name of the fields and rename the columns by more python recognized names...
colnames = ['ID','Month','lat','lon','code','name','type']
crime_data.columns = colnames
crime_data.head(4)


# In[14]:


# print all crime variables in the "type" column
crimes = crime_data['type'].sort_values().unique()
crimes, len(crimes)


# In[15]:


# for examples, lets check out the column 'crime type'
crime_data.type.value_counts()


# In[ ]:


#Explore the most prevalent crime types


# In[16]:


crime_data_group = crime_data.groupby('type').size().reset_index(name='count')
crime_data_sort = crime_data_group.sort_values(['count'], ascending=False).reset_index(drop=True)
crime_data_sort


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
from geopandas import GeoDataFrame
in_type=[crime_data_sort['type'][i]for i in range(3)]
fillcolors = ['#ff0000','#0000ff','#00ff00']
nlst = crime_data[crime_data.type.isin(in_type)].copy() 
nlst.shape


# In[22]:


nlst['lon'].notnull().count() # check whether they all have longitude data
# Your code here to replace the ???
nlst['lat'].notnull().count() # check whether they all have latitude data


# In[29]:


new_gdb = gpd.GeoSeries(nlst[['lon', 'lat']].apply(Point, axis=1), crs="+init=epsg:4326")
bbox = new_gdb.total_bounds
titles=["Kernel Density: "+in_type[i] for i in range(3)]
    
fig, axs = plt.subplots(2, 2, figsize = (12,12))

ax1 = plt.subplot2grid((8,8), (0,0), rowspan=3, colspan=3) 
ax2 = plt.subplot2grid((8,8), (4,0), rowspan=3, colspan=3)
ax3 = plt.subplot2grid((8,8), (0,4), rowspan=3, colspan=3)


fig.tight_layout(pad = 0.4, w_pad = 4.0, h_pad = 4.0)
ax1.set_title(titles[0], fontsize =16)
ax2.set_title(titles[1], fontsize =16)
ax3.set_title(titles[2], fontsize =16)

ax1.set_xlim(bbox[0], bbox[2])
ax1.set_ylim(bbox[1], bbox[3]) 
ax2.set_xlim(bbox[0], bbox[2])
ax2.set_ylim(bbox[1], bbox[3]) 
ax3.set_xlim(bbox[0], bbox[2])
ax3.set_ylim(bbox[1], bbox[3]) 

# ^The above code sets the x and y limits for each function. Without this, the density maps
# are very small and only take up about 20% of the graph space.
gdfnew1 = nlst[nlst['type']==in_type[0]]
gdfnew2 = nlst[nlst['type']==in_type[1]]
gdfnew3 = nlst[nlst['type']==in_type[2]]

sns.kdeplot(gdfnew1.lon, gdfnew1.lat, shade = True, cmap = "Reds", ax=ax1) 
sns.kdeplot(gdfnew2.lon, gdfnew2.lat, shade = True, cmap = "Blues", ax=ax2)
sns.kdeplot(gdfnew3.lon, gdfnew3.lat, shade = True, cmap = "Greens", ax=ax3)

sns.set(style = "whitegrid") # aesthetics
sns.despine(left=True) # aesthetics
sns.set_context("paper") # aesthetics
plt.axis('equal')
plt.show()   


# In[24]:


# Set up geodataframe, initially with CRS = WGS84 (since that matches the lon and lat co-ordinates)
crs = {'init':'epsg:4326'}
geometry = [shapely.geometry.Point(xy) for xy in zip(crime_data['lon'], crime_data['lat'])]
geo_df = gpd.GeoDataFrame(crime_data,
                               crs = crs,
                               geometry = geometry)
# Convert geometry to OSGB36 EPSG: 27700
geo_df_new = geo_df.to_crs(epsg = 27700)
# convert the .csv file into .shp file 
geo_df_new.to_file(driver='ESRI Shapefile', filename='crime_data.shp')


# In[25]:


# Plot map
fig, ax = plt.subplots(1,
                       figsize = (12,12),
                       dpi = 72,
                       facecolor = 'pink')

ax.set_position([0,0,1,1])   # Puts axis to edge of figure
ax.set_axis_off()            # Turns axis off so facecolour applies to axis area as well as bit around the outside
ax.get_xaxis().set_visible(False)   # Turns the x axis off so that 'invisible' axis labels don't take up space
ax.get_yaxis().set_visible(False)
lims = plt.axis('equal')
geo_df_new.plot(ax=ax)
plt.show()


# In[ ]:


crime_data.head(4)


# In[ ]:


#importing the polygon map for LGDs


# In[ ]:


# draw the LSOA map and set its Coordinate Reference Systems (CRS) into EPSG: 27700.
shapelgd=gpd.read_file('C:\\Users\\zahra\\Desktop\\programming_projectpp\\shapelgd.shp')
f, ax = plt.subplots(1, figsize=(12, 12))
ax = lsoa.plot(axes=ax);
shapelgd.crs = {'init' :'epsg:27700'}
plt.show()


# In[ ]:


# check the lsoa is a GeoDataFrame
type(shapelgd)


# In[ ]:


# check the labels for columns in lsoa
shapelgd.columns


# In[ ]:


# make the columns for LSOA data more readable
# rename the indicator with full title to help you interpret the columns/indicators
lsoa.drop(['objectid', 'lsoa11nmw','st_areasha', 'st_lengths', 'IMD2015_LS', 'IMD2015__1', 'IMD2015_Lo'], inplace=True, axis=1)
colnames2 = ['code','name','IMDindex','Income','Employment','Education', 'Health','Subcrime','BarriersHou','LivEnviron','Affect_child', 'Affect_old', 'child_young','Adult_skill','Geobarrier','widerbarrier','indoors', 'outdoors','population','depend_child', 'peo_middle', 'peo_over60', 'work_age','geometry']
lsoa.columns = colnames2
lsoa.head(4)


# In[ ]:


# to visulize both crime points data and lsoa vector data on a map
f, ax = plt.subplots(1, figsize=(12, 12))
ax.set_axis_off()
plt.axis('equal')
lsoa.plot(ax=ax, cmap='OrRd', linewidth=0.5, edgecolor='white')
geo_df_new.plot(column='type', markersize=3, categorical=True, legend=True, ax=ax)
# the legend was set by default to take the first column as labels.


# In[ ]:


#chloropleth for deprivation index


# In[30]:


# Make a Choropleth maps on the deprivation index.
shapelgd.plot(column='pc_emp_dep', cmap='OrRd', scheme='quantiles')


# In[ ]:


# Please plot series quantile map for 3 sub domain of deprivation indexes you want to explore
# e.g. Income, Employment, Education, etc.
# Your code here
shapelgd.plot(column='popn_dens', cmap='OrRd', scheme='quantiles')


# In[ ]:


shapelgd.plot(column='pc_emp_dep', cmap='OrRd', scheme='quantiles')


# In[ ]:


shapelgd.plot(column='age18_64', cmap='OrRd', scheme='quantiles')


# In[ ]:


# Make a Choropleth map on crime incidents per lsoa.
lsoa.plot(column='Numbers', cmap='coolwarm', scheme='quantiles')


# In[ ]:


#crime heat map


# In[31]:


crime_data['lat'] = crime_data['lat'].astype(float)
crime_data['lon'] = crime_data['lon'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
heat_df = crime_data[['lat', 'lon']]
heat_df = heat_df.dropna(axis=0, subset=['lat','lon'])

# List comprehension to make out list of lists
heat_data = [[row['lat'],row['lon']] for index, row in heat_df.iterrows()]

heatmap_map = folium.Map([51.50632, -0.1271448], zoom_start=12)

# Plot it on the map
hm=plugins.HeatMap(heat_data)
heatmap_map.add_child(hm)
# You save a map as an html file 
heatmap_map.save("heatmap.html")


# In[ ]:




