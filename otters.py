# STUDY OF LEAD AND PESTICIDES ACCUMULATION IN THE BONES OF EURASIAN OTTERS IN UNITED KINGDOM USING PYTHON 3.9.14 #
## Course: Data science and machine learning for the biosciences 
## Anusha Mohan Kumar (2350551), University of the West of England


## 1. MODULES REQUIRED TO RUN THE CODE##
import pandas as pd
import researchpy as rp
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



## 2. READING AND FORMATTING THE DATA ##
# Reading the datafile (csv format) to python and was assigned to python object called nanotters 
nanotters = pd.read_csv("otters.csv")
nanotters

# missing values in the data is replaced with 0 and the object was named as otters
otters = nanotters.fillna(0)
otters.head()

# organochloride pesticides were summed togther as total_pesticides
otters ["total_pesticides"] = otters ["dieldrin"]+ otters["dde"] + otters["tde"]+ otters["pcb105"]+ otters["pcb118"]+ otters["pcb128"]+ otters["pcb138"]+ otters["pcb153"]+ otters ["pcb156"]+ otters["pcb170"]+ otters["pcb180"]+ otters["pcb187"]+ otters["hcbenz"]
otters.head()



## 3 DIFFERENCES IN LEAD ACCUMULATION BETWEEN INDIVIDUALS ##
# T-test was carried out for BonePb level in males and females to find the influence of sex on lead accumulation
summary_raw, results_raw = rp.ttest(
    group1 = otters['BonePb'][otters['Sex'] == 'Male'], group1_name= "Male",
    group2= otters['BonePb'][otters['Sex'] == 'Female'], group2_name= "Female",
)
summary = pd.DataFrame(summary_raw)
summary
results = pd.DataFrame(results_raw)
results

# Visual representation of variation of lead content between male and female
sns.catplot(data=otters, x="Sex", y="BonePb", kind="box")
# Visualizing the influence of age of otters on lead accumulation in their bones
sns.catplot(data=otters, x="Age", y="BonePb", col="Sex", kind="box")


## 4. SPATIAL AND TEMPORAL VARIATION OF LEAD IN THE BONES OF THE OTTERS ##
# Using relplot from seaborn the variation in lead in different regions was studied and the change in the content of the lead in bones over the years
sns.relplot(
    data = otters, x="Year", y="BonePb", col = "region", kind = "scatter", hue = "Sex"
)
sns.relplot(data=otters, x="Year", y="BonePb", kind = "line")


## 5. PCA FOR THE CONTAMINANT LOAD IN OTTERS ##
# subset of data was created for organochlorine contaminants
X = pd.DataFrame(
    otters, columns=["dieldrin", "dde", "tde", "pcb105", "pcb118", "pcb128", "pcb138", "pcb153", "pcb156", "pcb170", "pcb180", "pcb187", "hcbenz"]
)
X.head()

# sex of otters was assigned to y variable for visualizing the spread of data 
ystring = otters ["Sex"]
ystring.head()

# replacing string (i.e female and male) as integer (0 and 1 respectively)
replacement = {
    "Female": "0",
    "Male": "1",
}
y = ystring.replace(replacement, regex=True)
y.head()

# Since different x variables are in different units, standard scaler is used to weigh the values of all components equally in two directions.
scaler = StandardScaler()
scaler.fit(X)
X_scaled_raw = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_raw, columns=X.columns)
X_scaled

# Principal component analysis was carried out on the scaled x values of different pesticide contaminants. Number of principal components defined are 10 (n_components) and projects the original data into the PCA space
pca = PCA(n_components=10) 
X_new = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_) #pca.explained_variance_ratio_  gives the percentage of variance explained by each components, the first principal explains 53.51% , second 16.48% and so on.

# Nearly 98% of the variation was explained by 10 principal components which is graphically presented 
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth = 2, color = "blue")
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Each variables importance is measured by magnitude of the values in every components which is obtained by the code pca.components_
components_raw = (pca.components_ )
components = pd.DataFrame (components_raw)
components
# Index column: Principal component 1 is represented by 0 and so on 
# Index row: Variables(13 contaminants) are assigned the number from 0 to 12 


# biplot for visualizing the pca 
# Source :  Author: Serafeim Loukas, serafeim.loukas@epfl.ch https://medium.com/geekculture/pca-clearly-explained-when-why-how-to-use-it-and-feature-importance-a-guide-in-python-37596289571c
def biplot(score, coeff , y):
    xs = score[:,0]    # projection on PC1
    ys = score[:,1]    # projection on PC2
    n = coeff.shape[0] # number of variables 
    
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r']
    markers=['o','x']
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], c = colors[s], marker=markers[s]) 
    for i in range(n):
        plt.arrow(0, 0,coeff[i,0], coeff [i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', ha = 'center', va = 'center',fontsize=10)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-2,5])
    plt.ylim([-5,5])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)
 
mpl.rcParams.update(mpl.rcParamsDefault)
biplot(X_new[:,0:2], np.transpose(pca.components_[0:2, :]), y)
plt.show()
