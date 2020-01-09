#!/usr/bin/env python
# coding: utf-8

# In[63]:



import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file
from sklearn.datasets import load_iris


# In[64]:


data = load_iris()
columns =  list(['sepal_length','sepal_width','petal_length','petal_width'])
df = pd.DataFrame(data.data, columns = columns)
df.head()


# In[65]:


q1 = df.quantile(q=0.25)
q2 = df.quantile(q=0.5)
q3 = df.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr


# In[66]:


def outliers(col):
    cat = col.name
    return col[(col > upper.loc[cat]) | (col < lower.loc[cat])]
out = df.apply(outliers).dropna()
print(out)


# In[67]:


if not out.empty:
    outx = []
    outy = []
    for keys in out.index:
        outx.append(keys[0])
        outy.append(out.loc[keys[0]].loc[keys[1]])


# In[68]:


p = figure(tools="", background_fill_color="#efefef", x_range=columns, toolbar_location=None)


qmin = df.quantile(q=0.00)
qmax = df.quantile(q=1.00)
upper = [min([x,y]) for (x,y) in zip(list(qmax),upper)]
lower = [max([x,y]) for (x,y) in zip(list(qmin),lower)]


p.segment(columns, upper, columns, q3, line_color="black")
p.segment(columns, lower, columns, q1, line_color="black")


p.vbar(columns, 0.7, q2, q3, fill_color="#E08E79", line_color="black")
p.vbar(columns, 0.7, q1, q2, fill_color="#3B8686", line_color="black")


p.rect(columns, lower, 0.2, 0.01, line_color="black")
p.rect(columns, upper, 0.2, 0.01, line_color="black")


# In[69]:



if not out.empty:
    p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = "white"
p.grid.grid_line_width = 2
p.xaxis.major_label_text_font_size="12pt"

output_file("boxplot1.html", title="boxplot1.py example")

show(p)


# In[70]:


import matplotlib.pyplot as plt 

li = []
for i in df.columns:
    li.append(df[i])
plt.boxplot(li)
plt.show()

