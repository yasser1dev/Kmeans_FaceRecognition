#!/usr/bin/env python
# coding: utf-8

# <h1>Face Recognition using K-means clustering<h1>

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.cluster import KMeans
from time import perf_counter 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mydb = mysql.connector.connect(
  host="localhost",
  user="login",
  passwd="pass",
  database="Db_Name"
)
mycursor = mydb.cursor()

mycursor.execute("SELECT face_data FROM facedata")

face_data = mycursor.fetchall()

mycursor.execute("SELECT name FROM facedata")

names = mycursor.fetchall()


# In[3]:


def getVectors(listest):
    listX=[] 
    for i in listest: 
        string=i
        for j in string:
            liststr=j.split(";")
            if "" in liststr:
                liststr.remove("")
            listfloat=[float(i) for i in liststr]
            listX.append(listfloat)
    return listX

def getVector(string):
    listX=[] 
    liststr=string.split(";")
    if "" in liststr:
        liststr.remove("")
    listfloat=[float(i) for i in liststr]
    listX.append(listfloat)
    return listX


# In[14]:


y=getVectors(face_data)
data=np.array(y[0:len(y)])

Fdf=pd.DataFrame()
dataFace={'Player':names[8:],'DataFace':face_data[8:]}
Fdf=pd.DataFrame(dataFace)



# In[15]:


origin=[75],[-75]
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.quiver(*origin,data[:,0],data[:,1],color=['r','r','r','g','c','y','m','K'],scale=0.4)
plt.show()


# In[16]:


plt.xlim(-0.4,0.4)
plt.ylim(-0.4,0.4)
plt.scatter(data[:,0],data[:,1],s=7)
plt.show()


# In[17]:


Kmean = KMeans(n_clusters=5)
Kmean.fit(data)


# In[18]:


centers=np.array(Kmean.cluster_centers_[:5])
#len(Kmean.cluster_centers_)




# In[22]:


centers=np.array(Kmean.cluster_centers_[:5])
plt.xlim(-0.3,0.3)
plt.ylim(-0.3,0.3)
plt.scatter(data[:,0],data[:,1],s=7,cmap='viridis')
plt.scatter(centers[:,0],centers[:,1],s=100,color=['r','k','b','y','m'])
plt.show()


# In[25]:


labels=Kmean.labels_



# In[29]:


df=pd.DataFrame()
dataX={'Names':names,'Clusters':Kmean.labels_}
df=pd.DataFrame(dataX)


# In[30]:

#New data face
string="-0.188544;0.095237;0.074243;-0.025494;-0.156491;0.064710;0.001180;-0.028229;0.064593;-0.043912;0.213778;-0.058394;-0.316834;-0.045117;-0.083976;0.140793;-0.160744;-0.126002;-0.162082;-0.086415;0.066026;-0.000921;-0.005023;0.074203;-0.115162;-0.286073;-0.015599;-0.131666;0.095471;-0.182737;-0.040339;-0.060210;-0.166168;-0.055323;-0.057478;0.019376;-0.049639;-0.115201;0.217509;-0.082100;-0.186580;-0.022452;0.034335;0.305159;0.185156;0.010131;0.064125;-0.023581;0.132554;-0.308604;0.064937;0.149767;0.135676;0.026239;0.026742;-0.192097;-0.016477;0.197890;-0.187218;0.158758;0.040050;-0.111802;0.004390;-0.042628;0.208341;0.091674;-0.097854;-0.125249;0.188304;-0.118678;0.051524;0.098207;-0.136022;-0.206991;-0.229943;0.046298;0.464189;0.151835;-0.129045;0.033149;-0.137372;0.013541;0.055108;0.029263;-0.124913;0.016775;-0.150708;0.142713;0.204265;-0.056071;-0.039482;0.195231;0.054817;-0.108707;0.084381;0.052328;-0.100235;-0.035316;-0.074722;0.007963;0.085539;-0.167964;0.067743;0.004829;-0.125409;0.107590;-0.015552;0.027830;-0.057694;-0.077954;-0.128881;0.041763;0.233357;-0.220671;0.213550;0.162669;0.050805;0.157627;0.099203;0.046564;0.012320;-0.051109;-0.087758;-0.055520;0.109958;-0.023378;0.185327;0.053302;"
new_data=getVector(string)


# In[31]:


t1_start = perf_counter() 
labels = Kmean.predict(new_data)
t1_stop = perf_counter() 


# In[32]:


print("Time elapsed: ", t1_stop - t1_start)


# In[33]:

#show pridected value
labels


# In[ ]:




