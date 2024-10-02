```python
# load conda environment
import sys
sys.path.append("/homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/")

from ml_mmpa import master_functions

# data process
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles, venn2


from scipy import stats

import seaborn as sns

# text process for assays 
import re

import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets, decomposition
from sklearn.manifold import TSNE

#chem

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Descriptors3D, Draw, rdMolDescriptors, Draw, PandasTools
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat, GetTanimotoDistMat
from rdkit.Chem.Draw import IPythonConsole

import sympy as sp
```

    RDKit WARNING: [10:39:44] Enabling RDKit 2019.09.2 jupyter extensions


### Plan:

Phase 1:

1. import data
2. standartise it
3. find overlap of data youre interested in 
4. plot and compute paired t-test


Phase 2:

<!-- Look for any diferences between the two forming compound sets, ones that increase property and ones that decrease property -->

1. Define 

### 1. import data



```python
# import master data
inhibition = pd.read_csv('data/CO-ADD_InhibitionData_r03_01-02-2020_CSV.csv', low_memory=False)
# s_aureus_master = pd.read_csv('../data/master_s_aureus.csv', dtype=np.unicode_ , sep=';')
# import master data
dose_response = pd.read_csv('data/CO-ADD_DoseResponseData_r03_01-02-2020_CSV.csv', low_memory=False)
# s_aureus_master = pd.read_csv('../data/master_s_aureus.csv', dtype=np.unicode_ , sep=';')

```


```python
### Import curated datasets:

e_coli_wild_perm = pd.read_pickle('data_curated/e_coli_wild_perm.pkl')
e_coli_wild_efflux = pd.read_pickle('data_curated/e_coli_wild_efflux.pkl')
e_coli_wild = pd.read_pickle('data_curated/e_coli_wild.pkl')
e_coli_s_aureus = pd.read_pickle('data_curated/e_coli_s_aureus.pkl')

# import substrate and evader

efflux_substrate = e_coli_wild_efflux[(e_coli_wild_efflux['INHIB_AVE_efflux']>65)&(e_coli_wild_efflux['INHIB_AVE_wild']<65)]

efflux_evader = e_coli_wild_efflux[(e_coli_wild_efflux['INHIB_AVE_wild']>65)&(e_coli_wild_efflux['INHIB_AVE_efflux']>65)]

sub_and_evade= pd.read_pickle('data_curated/sub_and_evade.pkl')
rest_of_ecoli_efflux= pd.read_pickle('data_curated/rest_of_ecoli_efflux.pkl')


### Import MMPA result:

# efflux_mmpa_index = pd.read_csv('out/index_co_add_wild_efflux_final.csv')

efflux_mmpa_index = pd.read_pickle('data_curated/efflux_mmpa_index.pkl')

efflux_mmpa_index_len_stat = pd.read_pickle('data_curated/efflux_mmpa_index_len_stat.pkl')

ecoli_wild_index=pd.read_csv('data_curated/index_inhib_wild_final.csv')


## relevant transforms:

substrate_transforms = pd.read_pickle('data_curated/substrate_transforms.pkl')

evader_transforms = pd.read_pickle('data_curated/evader_transforms.pkl')


#import classed compounds:

efflux_substrate = pd.read_pickle('data_curated/efflux_substrate.pkl')
efflux_evader= pd.read_pickle('data_curated/efflux_evader.pkl')
wt_only= pd.read_pickle('data_curated/wt_only.pkl')
inactive= pd.read_pickle('data_curated/inactive.pkl')
sub_and_evade= pd.read_pickle('data_curated/sub_and_evade.pkl')


# permeation

om_permeating = pd.read_pickle('data_curated/om_permeating.pkl')

om_non_permeating = pd.read_pickle('data_curated/om_non_permeating.pkl')


### OM bias


efflux_evaders_om_corrected = pd.read_pickle('data_curated/efflux_evaders_om_corrected.pkl')

efflux_substrates_om_corrected = pd.read_pickle('data_curated/efflux_substrates_om_corrected.pkl')

sub_and_evade_om_corrected = efflux_evaders_om_corrected.append(efflux_substrates_om_corrected).reset_index(drop=True)

### logD

sub_and_evade_om_corrected


sub_and_evade_logd = pd.read_csv('data_curated/sub_and_evade_PE.csv')

### Overlap

comp_a_lhs_overlap = pd.read_pickle('data_curated/comp_a_lhs_overlap.pkl')

```


```python
sub_and_evade_om_corrected[['SMILES']].to_csv('sub_and_evade.csv', index=True)
```

### 2. standartise it


Define all datasets: wild / tolC / lpxc / s.aureus

(where duplicated find mean)


```python
# e_coli_wild_grouped = inhibition[(inhibition['ORGANISM']=='Escherichia coli') & (inhibition['STRAIN']=='ATCC 25922')].groupby('SMILES').mean().reset_index()

e_coli_wild = inhibition[(inhibition['ORGANISM']=='Escherichia coli') & (inhibition['STRAIN']=='ATCC 25922')][['SMILES', 'INHIB_AVE']].groupby('SMILES').mean().reset_index()

e_coli_efflux = inhibition[(inhibition['ORGANISM']=='Escherichia coli') & (inhibition['STRAIN']=='tolC; MB5747')][['SMILES', 'INHIB_AVE']].groupby('SMILES').mean().reset_index()

e_coli_pore = inhibition[(inhibition['ORGANISM']=='Escherichia coli') & (inhibition['STRAIN']=='lpxC; MB4902')][['SMILES', 'INHIB_AVE']].groupby('SMILES').mean().reset_index()

s_aureus_wild = inhibition[(inhibition['ORGANISM']=='Staphylococcus aureus')][['SMILES', 'INHIB_AVE']].groupby('SMILES').mean().reset_index()
```

### 3. find overlap of data youre interested in 

wild / tolc

wild / lpxs

wild / s.aureus


```python
e_coli_wild_efflux = e_coli_wild[['SMILES', 'INHIB_AVE']].merge(e_coli_efflux[['SMILES', 'INHIB_AVE']],  on='SMILES', suffixes=('_wild', '_efflux'))
e_coli_wild_perm = e_coli_wild[['SMILES', 'INHIB_AVE']].merge(e_coli_pore[['SMILES', 'INHIB_AVE']], on='SMILES', suffixes=('_wild', '_lpxC'))
e_coli_s_aureus = e_coli_wild[['SMILES', 'INHIB_AVE']].merge(s_aureus_wild[['SMILES', 'INHIB_AVE']], on='SMILES', suffixes=('_ecoli', '_saureus'))


e_coli_wild_efflux = e_coli_wild_efflux.dropna().drop_duplicates(subset=['SMILES'])
e_coli_wild_perm = e_coli_wild_perm.dropna().drop_duplicates(subset=['SMILES'])
e_coli_s_aureus = e_coli_s_aureus.dropna().drop_duplicates(subset=['SMILES'])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-2a7cea0a1011> in <module>
    ----> 1 e_coli_wild_efflux = e_coli_wild[['SMILES', 'INHIB_AVE']].merge(e_coli_efflux[['SMILES', 'INHIB_AVE']],  on='SMILES', suffixes=('_wild', '_efflux'))
          2 e_coli_wild_perm = e_coli_wild[['SMILES', 'INHIB_AVE']].merge(e_coli_pore[['SMILES', 'INHIB_AVE']], on='SMILES', suffixes=('_wild', '_lpxC'))
          3 e_coli_s_aureus = e_coli_wild[['SMILES', 'INHIB_AVE']].merge(s_aureus_wild[['SMILES', 'INHIB_AVE']], on='SMILES', suffixes=('_ecoli', '_saureus'))
          4 
          5 


    NameError: name 'e_coli_efflux' is not defined



```python
# drop that one big annoying value

e_coli_wild_perm = e_coli_wild_perm[e_coli_wild_perm.SMILES != 'S(O)(=O)(=O)c1ccccc1\\C(\\c(cc(C)c(c2Br)O)c2)=C(\\C=C3C)/C=C(C3=O)Br']

e_coli_wild_efflux = e_coli_wild_efflux[e_coli_wild_efflux.SMILES != 'S(O)(=O)(=O)c1ccccc1\\C(\\c(cc(C)c(c2Br)O)c2)=C(\\C=C3C)/C=C(C3=O)Br']

e_coli_s_aureus = e_coli_s_aureus[e_coli_s_aureus.SMILES != 'S(O)(=O)(=O)c1ccccc1\\C(\\c(cc(C)c(c2Br)O)c2)=C(\\C=C3C)/C=C(C3=O)Br']
```

### 4. Plot and compute paired t-test (Wild vs TolC histogram FIGURE)

let's look at some plots between the overlapped datasets and calculate paired t-tests to determine if the distributions are different:


```python
# e_coli_wild_efflux[['INHIB_AVE_wild', 'INHIB_AVE_efflux']].plot.hist(bins=200, alpha=0.5, figsize=[10,7])


sns.set(rc={"figure.figsize":(10, 7)})

sns.set_style("ticks")

sns.histplot(e_coli_wild_efflux[['INHIB_AVE_efflux', 'INHIB_AVE_wild']], alpha=0.5, bins=150)
sns.despine()

plt.legend(labels = ['Wild Type', '$\Delta TolC$'],  fontsize=22)

plt.xlim([-120, 120])

plt.xlabel('Growth Inhibition based on $OD_{600}$ (%)', fontsize=22);

plt.ylabel('Number of Compounds',  fontsize=22);

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


plt.tight_layout()
# plt.savefig('figures/hist_wild_tolc.png', dpi=600)

```


    
![png](co_add_chaser_files/co_add_chaser_12_0.png)
    



```python
e_coli_wild_efflux.INHIB_AVE_efflux.mean(), e_coli_wild_efflux.INHIB_AVE_wild.mean()
```




    (6.611431868216318, 4.068315397968456)




```python
e_coli_wild_efflux.INHIB_AVE_efflux.std(), e_coli_wild_efflux.INHIB_AVE_wild.std()
```




    (17.071170693888337, 9.732757492784494)




```python
e_coli_wild_efflux.INHIB_AVE_efflux.std()*4 + 6.6, e_coli_wild_efflux.INHIB_AVE_wild.std()*4+4.07
```




    (74.88468277555334, 43.001029971137974)




```python
stats.ttest_rel(e_coli_wild_efflux['INHIB_AVE_wild'], e_coli_wild_efflux['INHIB_AVE_efflux'])
```




    Ttest_relResult(statistic=-44.09489438970853, pvalue=0.0)




```python
sns.histplot(data = e_coli_wild_perm, x='INHIB_AVE_lpxC')
sns.histplot(data = e_coli_wild_perm, x='INHIB_AVE_wild')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b2520665e10>




    
![png](co_add_chaser_files/co_add_chaser_17_1.png)
    



```python
len(e_coli_wild_perm)
```




    80620




```python
e_coli_wild_perm.plot.hist(bins=200, alpha=0.5, figsize=[10,7])

# e_coli_wild_efflux[['INHIB_AVE_wild', 'INHIB_AVE_efflux']].plot.hist(bins=200, alpha=0.5, figsize=[10,7])

sns.set(rc={"figure.figsize":(10, 7)})

sns.set_style("ticks")

sns.histplot(e_coli_wild_perm[['INHIB_AVE_lpxC', 'INHIB_AVE_wild']], alpha=0.5, bins=150)

sns.despine()

plt.legend(labels=['lpxC', 'Wild Type']  ,fontsize=22)

plt.xlim([-120, 120])

plt.xlabel('Growth Inhibition based on $OD_{600}$ (%)', fontsize=22);

plt.ylabel('Number of Compounds',  fontsize=22);

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


plt.tight_layout()
plt.savefig('figures/hist_wild_lpxc.png', dpi=600)

```


    
![png](co_add_chaser_files/co_add_chaser_19_0.png)
    



```python
stats.ttest_rel(e_coli_wild_perm['INHIB_AVE_wild'], e_coli_wild_perm['INHIB_AVE_lpxC'])
```




    Ttest_relResult(statistic=-40.55675268375553, pvalue=0.0)




```python
e_coli_s_aureus.plot.hist(bins=200, alpha=0.5, figsize=[10,7])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b20955c5450>




    
![png](co_add_chaser_files/co_add_chaser_21_1.png)
    



```python
stats.ttest_rel(e_coli_s_aureus['INHIB_AVE_ecoli'], e_coli_s_aureus['INHIB_AVE_saureus'])
```




    Ttest_relResult(statistic=-137.98270899888263, pvalue=0.0)



The goal is to assign a single value to the compound, two ways to go about that:

we will stick with absolute difference:  find lpxs - wild

### Save the curated datasets:



```python
# e_coli_wild_efflux['SMILES'] = e_coli_wild_efflux['SMILES'].apply(Chem.CanonSmiles)
```


```python
# e_coli_wild_perm.to_pickle('data_curated/e_coli_wild_perm.pkl')
# e_coli_wild_efflux.to_pickle('data_curated/e_coli_wild_efflux.pkl')
# e_coli_wild.to_pickle('data_curated/e_coli_wild.pkl')
# e_coli_s_aureus.to_pickle('data_curated/e_coli_s_aureus.pkl')
```

### Dev


```python
# e_coli_wild_perm_only_pos = e_coli_wild_perm[(e_coli_wild_perm['INHIB_AVE_lpxC']>0) & ( e_coli_wild_perm['INHIB_AVE_wild']>0)]
# len(e_coli_wild_perm_only_pos)
```




    44897




```python
# e_coli_wild_perm_only_pos['perm_diff'] =  e_coli_wild_perm_only_pos['INHIB_AVE_lpxC'] - e_coli_wild_perm_only_pos['INHIB_AVE_wild']
# e_coli_wild_perm_only_pos['perm_diff'].plot.hist(bins=100, alpha=0.5, figsize=[10,7])
```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.





    <matplotlib.axes._subplots.AxesSubplot at 0x2b91e8544150>




    
![png](co_add_chaser_files/co_add_chaser_29_2.png)
    


Another way to do that is to look at direction and take the right value like so: lpxc > wild




```python
# e_coli_wild_perm_direction = e_coli_wild_perm[(e_coli_wild_perm['INHIB_AVE_lpxC']> e_coli_wild_perm['INHIB_AVE_wild'])]
```


```python
# e_coli_wild_perm_direction['abs_diff']  = e_coli_wild_perm_direction.INHIB_AVE_lpxC - e_coli_wild_perm_direction.INHIB_AVE_wild
```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
# e_coli_wild_perm_direction['abs_diff'].plot.hist(bins=100, alpha=0.5, figsize=[10,7])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b91e81df3d0>




    
![png](co_add_chaser_files/co_add_chaser_33_1.png)
    



```python
# e_coli_wild_perm_direction[e_coli_wild_perm_direction['abs_diff']>0].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>INHIB_AVE_wild</th>
      <th>INHIB_AVE_lpxC</th>
      <th>abs_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>B(C1)(CC(C2)CC3CC12)C3.n(cc4Br)cc(c4)Br</td>
      <td>12.08</td>
      <td>12.83</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B(C1)(CC(C2)CC3CC12)C3.n(cccc4Cc5ccccc5)c4</td>
      <td>2.94</td>
      <td>3.51</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B(CC1CC2CC3C1)(C3)C2.N#Cc(cc4)ccn4</td>
      <td>7.55</td>
      <td>7.56</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B(c1ccccc1)(OC(C)CC2P(c3ccccc3)(c4ccccc4)=O)O2</td>
      <td>8.95</td>
      <td>18.56</td>
      <td>9.61</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B1(c2ccccc2)OC(C)CC(P(c3ccccc3)c4ccccc4)O1</td>
      <td>10.17</td>
      <td>28.85</td>
      <td>18.68</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BrC(/C(OC1=C2C(C=CC=C3)=C3C=C1)=NBr)(C2C4=C(C=...</td>
      <td>-25.63</td>
      <td>-10.82</td>
      <td>14.81</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BrC(/C(OC1=C2C(C=CC=C3)=C3C=C1)=NBr)(C2C4=CC(O...</td>
      <td>-16.78</td>
      <td>5.16</td>
      <td>21.94</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BrC(/C(OC1=C2C(C=CC=C3)=C3C=C1)=NBr)(C2C4=CC([...</td>
      <td>-20.79</td>
      <td>0.14</td>
      <td>20.93</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BrC(/C(OC1=C2C(C=CC=C3)=C3C=C1)=NBr)(C2C4=CC=C...</td>
      <td>-19.75</td>
      <td>0.20</td>
      <td>19.95</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BrC(/C(OC1=C2C(C=CC=C3)=C3C=C1)=NBr)(C2C4=CC=C...</td>
      <td>-19.01</td>
      <td>-16.49</td>
      <td>2.52</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC(OC)...</td>
      <td>-7.21</td>
      <td>13.42</td>
      <td>20.63</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC([N+...</td>
      <td>-14.11</td>
      <td>15.55</td>
      <td>29.66</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC=C(C...</td>
      <td>-14.95</td>
      <td>-2.42</td>
      <td>12.53</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC=C(C...</td>
      <td>-8.93</td>
      <td>13.98</td>
      <td>22.91</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC=C(C...</td>
      <td>-19.60</td>
      <td>-2.05</td>
      <td>17.55</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC=CC=...</td>
      <td>-9.86</td>
      <td>12.69</td>
      <td>22.55</td>
    </tr>
    <tr>
      <th>27</th>
      <td>BrC(/C(OC1=C2C=CC3=C1C=CC=C3)=NBr)(C2C4=CC=CC=...</td>
      <td>-16.66</td>
      <td>-6.55</td>
      <td>10.11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BrC(/C=N\NC1=NC(N2CCOCC2)=NC(N3CCCC3)=N1)=C/C4...</td>
      <td>18.02</td>
      <td>24.17</td>
      <td>6.15</td>
    </tr>
    <tr>
      <th>57</th>
      <td>BrC1=C2C(NC(C(C)C2)=O)=C3C(N=C(CCCC4)C4=C3N)=N1</td>
      <td>-12.78</td>
      <td>14.57</td>
      <td>27.35</td>
    </tr>
    <tr>
      <th>58</th>
      <td>BrC1=C2C(NC(CC2C)=O)=C3C(N=C(CCCC4)C4=C3N)=N1</td>
      <td>-6.57</td>
      <td>20.01</td>
      <td>26.58</td>
    </tr>
  </tbody>
</table>
</div>



<!-- ### 5. Find difference in pre and post interested dataset -->



```python
# e_coli_wild_perm['abs_diff']  = e_coli_wild_perm.INHIB_AVE_lpxC - e_coli_wild_perm.INHIB_AVE_wild
```


```python
# e_coli_wild_perm['abs_diff'].plot.hist(bins=100, alpha=0.5, figsize=[10,7])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b20842cc650>




    
![png](co_add_chaser_files/co_add_chaser_37_1.png)
    



```python
# def calc_feats(df):

#     table=pd.DataFrame()
# #     df=df.dropna() 
    
#     for i,mol in enumerate(df):
# #         Chem.SanitizeMol(mol)
# #         table.loc[i,'SMILES']=Chem.MolToSmiles(mol)
# #         table.loc[i,'Mol']=mol
#         table.loc[i,'MolWt']=Descriptors.MolWt(mol)
#         table.loc[i,'LogP']=Descriptors.MolLogP(mol)
#         table.loc[i,'NumHAcceptors']=Descriptors.NumHAcceptors(mol)
#         table.loc[i,'NumHDonors']=Descriptors.NumHDonors(mol)
#         table.loc[i,'NumHeteroatoms']=Descriptors.NumHeteroatoms(mol)
#         table.loc[i,'NumRotatableBonds']=Descriptors.NumRotatableBonds(mol)
#         table.loc[i,'NumHeavyAtoms']=Descriptors.HeavyAtomCount (mol)
#         table.loc[i,'NumAliphaticCarbocycles']=Descriptors.NumAliphaticCarbocycles(mol)
#         table.loc[i,'NumAliphaticHeterocycles']=Descriptors.NumAliphaticHeterocycles(mol)
#         table.loc[i,'NumAliphaticRings']=Descriptors.NumAliphaticRings(mol)
#         table.loc[i,'NumAromaticCarbocycles']=Descriptors.NumAromaticCarbocycles(mol)
#         table.loc[i,'NumAromaticHeterocycles']=Descriptors.NumAromaticHeterocycles(mol)
#         table.loc[i,'NumAromaticRings']=Descriptors.NumAromaticRings(mol)
#         table.loc[i,'RingCount']=Descriptors.RingCount(mol)
#         table.loc[i,'FractionCSP3']=Descriptors.FractionCSP3(mol)
#         table.loc[i,'TPSA']=Descriptors.TPSA(mol)
    
#     return table


```


```python
e_coli_wild_efflux_features = calc_feats(e_coli_wild_efflux['Mol'])

```


```python
e_coli_wild_efflux_features.to_pickle('data_curated/e_coli_wild_efflux_features.pkl')
```


```python
e_coli_wild_efflux_features['sub_class'] = e_coli_wild_efflux['sub_class']
```


```python
#pca

table = e_coli_wild_efflux_features

descriptors = table[['MolWt', 'LogP','NumHeteroatoms','RingCount','FractionCSP3', 'TPSA']].values #The non-redundant molecular descriptors chosen for PCA

# descriptors=table.iloc[:,2:]

descriptors_std = StandardScaler().fit_transform(descriptors) #Important to avoid scaling problems between our different descriptors
pca = PCA()
descriptors_2d = pca.fit_transform(descriptors_std)
descriptors_pca= pd.DataFrame(descriptors_2d) # Saving PCA values to a new table
descriptors_pca.index = table.index
descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
descriptors_pca.head(5) #Displays the PCA table

scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1'])) 
scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

# And we add the new values to our PCA table
descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]


descriptors_pca['sub_class'] = e_coli_wild_efflux_features['sub_class']



# plt.rcParams['axes.linewidth'] = 1.5


cmap = sns.diverging_palette(133, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(13,8))



sns.scatterplot(x='PC1',y='PC2',data=descriptors_pca, alpha=0.7, hue='sub_class',style='sub_class', s=30)#, palette=["C0", "C1", "C2", "k"])


pca_lab= ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontsize=16,fontweight='bold')
plt.ylabel(pca_lab[1],fontsize=16,fontweight='bold')

plt.tick_params ('both',width=2,labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

plt.tight_layout()
plt.show()

print('same but in contours, for ease of read')

cmap = sns.diverging_palette(133, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(13,8))

sns.kdeplot(x='PC1',y='PC2',data=descriptors_pca, hue='sub_class' , levels=5,)


pca_lab= ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontsize=16,fontweight='bold')
plt.ylabel(pca_lab[1],fontsize=16,fontweight='bold')

plt.tick_params ('both',width=2,labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

plt.tight_layout()
plt.show()
```


```python
sns.jointplot(data = e_coli_wild_efflux_features,  x = 'MolWt',  y = 'LogP', hue='sub_class',)
```




    <seaborn.axisgrid.JointGrid at 0x2b71911889d0>




    
![png](co_add_chaser_files/co_add_chaser_43_1.png)
    



```python
e_coli_wild_efflux[['SMILES', 'abs_diff']].to_csv('co_add_wild_efflux.csv', index=False)
```


```python
#### import the mmpa:


```


```python
efflux_mmpa_index_len = master_functions.clean_mmpa_pairs_len(efflux_mmpa_index) # filter pairs by len LHS & RHS vs CORE
```

    Initial number of transofrms: 1406980 
    Number fo transforms disqualified based on length discrepancy: 526528 
    Remaining number of transforms: 880452



```python
efflux_mmpa_index_len_stat = master_functions.stat_it_2(efflux_mmpa_index_len)
```


```python
efflux_mmpa_index_len_stat.to_pickle("data_curated/efflux_mmpa_index_len_stat.pkl")
```


```python
efflux_mmpa_index_len_stat.measurement_delta.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b2c31784090>




    
![png](co_add_chaser_files/co_add_chaser_49_1.png)
    



```python
efflux_mmpa_neg_pos = master_functions.zero_in(efflux_mmpa_index_len_stat, pos_only=False, cutoff=0.05) #  is this filtering for positive?
```

    Number of unique transforms where p-val < 0.05 is 5197
    Split between 2566 positive transforms and 2631 negative transforms



```python
efflux_mmpa_neg_pos = master_functions.split_transition(efflux_mmpa_neg_pos, 'smirks')
```


```python
efflux_mmpa_neg_pos.measurement_delta.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b2c31723750>




    
![png](co_add_chaser_files/co_add_chaser_52_1.png)
    



```python
efflux_mmpa_pos = efflux_mmpa_neg_pos[efflux_mmpa_neg_pos['measurement_delta']>0]
```


```python
efflux_mmpa_pos[efflux_mmpa_pos['RHS']=='[*:1][H]']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>smirks</th>
      <th>dof</th>
      <th>t-stat</th>
      <th>p-val (t-test)</th>
      <th>measurement_delta</th>
      <th>std</th>
      <th>sem</th>
      <th>LHS</th>
      <th>RHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>157</th>
      <td>[*:1]c1nc2ccccc2o1&gt;&gt;[*:1][H]</td>
      <td>6</td>
      <td>3.049545</td>
      <td>0.022525</td>
      <td>35.437143</td>
      <td>30.744866</td>
      <td>11.620467</td>
      <td>[*:1]c1nc2ccccc2o1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>476</th>
      <td>[*:1]OCc1ccc(Cl)cc1&gt;&gt;[*:1][H]</td>
      <td>9</td>
      <td>3.132617</td>
      <td>0.012071</td>
      <td>24.295000</td>
      <td>24.525031</td>
      <td>7.755496</td>
      <td>[*:1]OCc1ccc(Cl)cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>542</th>
      <td>[*:1]NC(=O)c1cccc(C)c1&gt;&gt;[*:1][H]</td>
      <td>2</td>
      <td>18.650861</td>
      <td>0.002862</td>
      <td>22.823333</td>
      <td>2.119536</td>
      <td>1.223715</td>
      <td>[*:1]NC(=O)c1cccc(C)c1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>559</th>
      <td>[*:1]NC(=O)c1ccc(Cl)cc1&gt;&gt;[*:1][H]</td>
      <td>2</td>
      <td>5.621207</td>
      <td>0.030220</td>
      <td>22.516667</td>
      <td>6.938014</td>
      <td>4.005664</td>
      <td>[*:1]NC(=O)c1ccc(Cl)cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>751</th>
      <td>[*:1]C(=O)Nc1nnc(C)s1&gt;&gt;[*:1][H]</td>
      <td>2</td>
      <td>5.034021</td>
      <td>0.037269</td>
      <td>19.633333</td>
      <td>6.755223</td>
      <td>3.900130</td>
      <td>[*:1]C(=O)Nc1nnc(C)s1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>[*:1]C(=O)Nc1ccc(I)cc1&gt;&gt;[*:1][H]</td>
      <td>2</td>
      <td>20.418814</td>
      <td>0.002390</td>
      <td>15.080000</td>
      <td>1.279179</td>
      <td>0.738535</td>
      <td>[*:1]C(=O)Nc1ccc(I)cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1191</th>
      <td>[*:1]CCCCCCCC&gt;&gt;[*:1][H]</td>
      <td>4</td>
      <td>8.840509</td>
      <td>0.000904</td>
      <td>13.916000</td>
      <td>3.519834</td>
      <td>1.574118</td>
      <td>[*:1]CCCCCCCC</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>[*:1]OC(C)C&gt;&gt;[*:1][H]</td>
      <td>16</td>
      <td>2.987736</td>
      <td>0.008699</td>
      <td>12.592941</td>
      <td>17.378383</td>
      <td>4.214877</td>
      <td>[*:1]OC(C)C</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>[*:1]NC(=O)c1ccccc1&gt;&gt;[*:1][H]</td>
      <td>13</td>
      <td>3.304501</td>
      <td>0.005698</td>
      <td>12.444286</td>
      <td>14.090553</td>
      <td>3.765859</td>
      <td>[*:1]NC(=O)c1ccccc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>[*:1]NC(=O)c1ccc(C)cc1&gt;&gt;[*:1][H]</td>
      <td>7</td>
      <td>2.981994</td>
      <td>0.020457</td>
      <td>12.155000</td>
      <td>11.529043</td>
      <td>4.076132</td>
      <td>[*:1]NC(=O)c1ccc(C)cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1660</th>
      <td>[*:1]C(=O)NCC1CCCO1&gt;&gt;[*:1][H]</td>
      <td>8</td>
      <td>2.491253</td>
      <td>0.037449</td>
      <td>9.833333</td>
      <td>11.841433</td>
      <td>3.947144</td>
      <td>[*:1]C(=O)NCC1CCCO1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>[*:1]C#Cc1ccccc1&gt;&gt;[*:1][H]</td>
      <td>3</td>
      <td>4.983685</td>
      <td>0.015531</td>
      <td>7.552500</td>
      <td>3.030890</td>
      <td>1.515445</td>
      <td>[*:1]C#Cc1ccccc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>[*:1]c1ccc(I)cc1&gt;&gt;[*:1][H]</td>
      <td>14</td>
      <td>3.251720</td>
      <td>0.005793</td>
      <td>7.482667</td>
      <td>8.912282</td>
      <td>2.301141</td>
      <td>[*:1]c1ccc(I)cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2053</th>
      <td>[*:1]C(C)c1ccccc1&gt;&gt;[*:1][H]</td>
      <td>4</td>
      <td>3.289522</td>
      <td>0.030229</td>
      <td>6.658000</td>
      <td>4.525806</td>
      <td>2.024002</td>
      <td>[*:1]C(C)c1ccccc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2097</th>
      <td>[*:1]c1cccc(OC)c1&gt;&gt;[*:1][H]</td>
      <td>26</td>
      <td>2.114801</td>
      <td>0.044190</td>
      <td>6.504074</td>
      <td>15.980778</td>
      <td>3.075502</td>
      <td>[*:1]c1cccc(OC)c1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2210</th>
      <td>[*:1]c1cccc(C)c1&gt;&gt;[*:1][H]</td>
      <td>33</td>
      <td>2.120839</td>
      <td>0.041537</td>
      <td>5.853235</td>
      <td>16.092658</td>
      <td>2.759868</td>
      <td>[*:1]c1cccc(C)c1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2530</th>
      <td>[*:1]C(F)(F)F&gt;&gt;[*:1][H]</td>
      <td>196</td>
      <td>2.248762</td>
      <td>0.025640</td>
      <td>2.849315</td>
      <td>17.784026</td>
      <td>1.267059</td>
      <td>[*:1]C(F)(F)F</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2563</th>
      <td>[*:1]C&gt;&gt;[*:1][H]</td>
      <td>4828</td>
      <td>4.442076</td>
      <td>0.000009</td>
      <td>1.065756</td>
      <td>16.672484</td>
      <td>0.239923</td>
      <td>[*:1]C</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2564</th>
      <td>[*:1]Sc1nnc2n(N)cnn12&gt;&gt;[*:1][H]</td>
      <td>2</td>
      <td>inf</td>
      <td>0.000000</td>
      <td>0.790000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>[*:1]Sc1nnc2n(N)cnn12</td>
      <td>[*:1][H]</td>
    </tr>
  </tbody>
</table>
</div>




```python
features_all_pos, l_feats_pos, r_feats_pos = master_functions.calculate_fractions_mk4(efflux_mmpa_pos)
```

      1%|          | 21/2566 [00:00<00:12, 201.22it/s]

    Generating molecular objects from pre-defined substructures
    Calcualting LHS+RHS matches


    100%|██████████| 2566/2566 [00:12<00:00, 212.06it/s]



```python
to_drop = ['arene', 'heteroarene', 'alkyne', 'benzene ring', 'amine', 'azaarene', 'alkene', 'aryl halide', 'alkyl halide', 'leaving group', 'alkenyl halide']

features_all_dropped = features_all_pos.drop(to_drop, axis = 1)

l_feats_dropped =  l_feats_pos.drop(to_drop, axis = 1)
r_feats_dropped =  r_feats_pos.drop(to_drop, axis = 1)

# fractions_to_drop=['fr_ketone_Topliss', 'fr_Al_OH_noTert', 'fr_Ar_N', 'fr_methoxy', 'fr_C_O', 'fr_phenol_noOrthoHbond' ]
# fractions above are kept as significant but not looked at on the exchange

fr_sig_descriptors = master_functions.find_sig_feats_mk2(l_feats_dropped, r_feats_dropped, 0.01)
```

    Found significant fractions:  18



```python
#fr_sig_descriptors.remove('fr_NH0')
fractions_to_drop=[]

# res_neg= master_functions.results_arr(features_all.drop(columns=['fr_NH0']), fr_sig_descriptors, r_feats, l_feats, fractions_to_drop )

res_neg_neg= master_functions.results_arr(features_all_dropped, fr_sig_descriptors, r_feats_dropped, l_feats_dropped, fractions_to_drop )
```

    aniline has negative correlation 
    phenol has positive correlation 
    tertiary amine has negative correlation 
    enamine has negative correlation 
    iminyl has negative correlation 
    N-acylcarbamate or urea (mixed imide) has negative correlation 
    first_gain
    [('secondary carboxamide', 'carboxamide'), 'iminyl', 'aniline']
    [71.43, 57.14, 14.29]
    percentage_loss 100
    carboxamide has negative correlation 
    lactam has positive correlation 
    oxime ether has positive correlation 
    second double loss
    ['α,β-unsaturated carbonyl', ('1,2-diol', 'carboxylic acid'), 'nitro']
    [-15.79, -10.53, -5.26]
    alkanol has positive correlation 
    second double loss
    ['ether', ('aryl chloride', 'aryl bromide'), 'carbonyl']
    [-17.3, -8.02, -5.91]
    α,β-unsaturated carbonyl has negative correlation 
    carbonyl has negative correlation 
    first_gain
    [('aniline', 'ether'), 'aryl bromide', 'alkanol']
    [10.08, 6.59, 5.81]
    1,2-diol has positive correlation 
    ketone has negative correlation 
    alkyl fluoride has positive correlation 
    aryl bromide has negative correlation 
    aryl iodide has negative correlation 
    sulfide has positive correlation 
    second double loss
    ['aniline', ('secondary amine', 'ether'), 'aryl bromide']
    [-26.42, -24.53, -18.87]



```python
res_neg_neg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Main fraction</th>
      <th>Correlation</th>
      <th>$\overline{\Delta P}$</th>
      <th>sem</th>
      <th>std</th>
      <th>dof</th>
      <th>Opposite fraction 1</th>
      <th>% of opposite 1</th>
      <th>Opposite fraction 2</th>
      <th>% of opposite 2</th>
      <th>Opposite fraction 3</th>
      <th>% of opposite 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>1,2-diol</td>
      <td>Positive</td>
      <td>41.75</td>
      <td>4.41</td>
      <td>20.67</td>
      <td>22</td>
      <td>ether</td>
      <td>-22.73</td>
      <td>secondary amine</td>
      <td>-9.09</td>
      <td>aniline</td>
      <td>-4.55</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ketone</td>
      <td>Negative</td>
      <td>-30.28</td>
      <td>4.34</td>
      <td>22.14</td>
      <td>26</td>
      <td>tertiary amine</td>
      <td>15.38</td>
      <td>(carboxamide, ether, aryl chloride)</td>
      <td>7.69</td>
      <td>tertiary carboxamide</td>
      <td>7.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phenol</td>
      <td>Positive</td>
      <td>26.03</td>
      <td>1.10</td>
      <td>16.68</td>
      <td>228</td>
      <td>ether</td>
      <td>-17.98</td>
      <td>aryl chloride</td>
      <td>-9.21</td>
      <td>aryl bromide</td>
      <td>-8.33</td>
    </tr>
    <tr>
      <th>9</th>
      <td>alkanol</td>
      <td>Positive</td>
      <td>25.56</td>
      <td>1.08</td>
      <td>16.56</td>
      <td>237</td>
      <td>ether</td>
      <td>-17.30</td>
      <td>(aryl chloride, aryl bromide)</td>
      <td>-8.02</td>
      <td>carbonyl</td>
      <td>-5.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>enamine</td>
      <td>Negative</td>
      <td>-23.71</td>
      <td>2.70</td>
      <td>13.77</td>
      <td>26</td>
      <td>carboxamide</td>
      <td>26.92</td>
      <td>secondary carboxamide</td>
      <td>23.08</td>
      <td>hydrazone</td>
      <td>3.85</td>
    </tr>
    <tr>
      <th>8</th>
      <td>oxime ether</td>
      <td>Positive</td>
      <td>20.57</td>
      <td>2.60</td>
      <td>11.35</td>
      <td>19</td>
      <td>α,β-unsaturated carbonyl</td>
      <td>-15.79</td>
      <td>(1,2-diol, carboxylic acid)</td>
      <td>-10.53</td>
      <td>nitro</td>
      <td>-5.26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>iminyl</td>
      <td>Negative</td>
      <td>-20.28</td>
      <td>1.63</td>
      <td>12.38</td>
      <td>58</td>
      <td>lactam</td>
      <td>12.07</td>
      <td>sulfide</td>
      <td>8.62</td>
      <td>primary amine</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>10</th>
      <td>α,β-unsaturated carbonyl</td>
      <td>Negative</td>
      <td>-20.10</td>
      <td>1.96</td>
      <td>13.03</td>
      <td>44</td>
      <td>carboxamide</td>
      <td>22.73</td>
      <td>secondary carboxamide</td>
      <td>20.45</td>
      <td>oxime ether</td>
      <td>6.82</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sulfide</td>
      <td>Positive</td>
      <td>19.98</td>
      <td>1.22</td>
      <td>8.87</td>
      <td>53</td>
      <td>aniline</td>
      <td>-26.42</td>
      <td>(secondary amine, ether)</td>
      <td>-24.53</td>
      <td>aryl bromide</td>
      <td>-18.87</td>
    </tr>
    <tr>
      <th>0</th>
      <td>aniline</td>
      <td>Negative</td>
      <td>-19.08</td>
      <td>0.68</td>
      <td>10.51</td>
      <td>236</td>
      <td>ether</td>
      <td>23.73</td>
      <td>aryl chloride</td>
      <td>7.20</td>
      <td>sulfide</td>
      <td>6.78</td>
    </tr>
    <tr>
      <th>11</th>
      <td>carbonyl</td>
      <td>Negative</td>
      <td>-18.58</td>
      <td>0.85</td>
      <td>13.58</td>
      <td>258</td>
      <td>(aniline, ether)</td>
      <td>10.08</td>
      <td>aryl bromide</td>
      <td>6.59</td>
      <td>alkanol</td>
      <td>5.81</td>
    </tr>
    <tr>
      <th>7</th>
      <td>lactam</td>
      <td>Positive</td>
      <td>18.49</td>
      <td>3.12</td>
      <td>10.34</td>
      <td>11</td>
      <td>iminyl</td>
      <td>-63.64</td>
      <td>secondary carboxamide</td>
      <td>-27.27</td>
      <td>nitrile</td>
      <td>-9.09</td>
    </tr>
    <tr>
      <th>15</th>
      <td>aryl bromide</td>
      <td>Negative</td>
      <td>-18.41</td>
      <td>0.54</td>
      <td>11.51</td>
      <td>453</td>
      <td>nitro</td>
      <td>10.82</td>
      <td>aryl chloride</td>
      <td>8.61</td>
      <td>phenol</td>
      <td>5.74</td>
    </tr>
    <tr>
      <th>16</th>
      <td>aryl iodide</td>
      <td>Negative</td>
      <td>-18.13</td>
      <td>1.40</td>
      <td>11.83</td>
      <td>71</td>
      <td>aryl chloride</td>
      <td>19.72</td>
      <td>(alkanol, phenol, nitro)</td>
      <td>14.08</td>
      <td>aryl bromide</td>
      <td>14.08</td>
    </tr>
    <tr>
      <th>6</th>
      <td>carboxamide</td>
      <td>Negative</td>
      <td>-16.60</td>
      <td>0.93</td>
      <td>9.25</td>
      <td>98</td>
      <td>aniline</td>
      <td>17.35</td>
      <td>aryl bromide</td>
      <td>13.27</td>
      <td>secondary amine</td>
      <td>9.18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>N-acylcarbamate or urea (mixed imide)</td>
      <td>Negative</td>
      <td>-16.38</td>
      <td>5.54</td>
      <td>14.65</td>
      <td>7</td>
      <td>(secondary carboxamide, carboxamide)</td>
      <td>71.43</td>
      <td>iminyl</td>
      <td>57.14</td>
      <td>aniline</td>
      <td>14.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tertiary amine</td>
      <td>Negative</td>
      <td>-15.58</td>
      <td>1.70</td>
      <td>10.04</td>
      <td>35</td>
      <td>phenol</td>
      <td>25.71</td>
      <td>alkanol</td>
      <td>20.00</td>
      <td>secondary amine</td>
      <td>17.14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>alkyl fluoride</td>
      <td>Positive</td>
      <td>11.89</td>
      <td>1.11</td>
      <td>5.67</td>
      <td>26</td>
      <td>aryl chloride</td>
      <td>-26.92</td>
      <td>carbonyl</td>
      <td>-23.08</td>
      <td>aryl bromide</td>
      <td>-15.38</td>
    </tr>
  </tbody>
</table>
</div>




```python
master_functions.plot_feats(res_neg_neg)
```


    
![png](co_add_chaser_files/co_add_chaser_59_0.png)
    


# Phase 2: Efflux


## Define 'active compound'



```python
# 4 log_2 folds away

# below 3 std away from average
```


```python
# z-score:
e_coli_wild_efflux['wild_stds'] = stats.zscore(e_coli_wild_efflux.INHIB_AVE_wild)
e_coli_wild_efflux['tolc_stds'] = stats.zscore(e_coli_wild_efflux.INHIB_AVE_efflux)
```


```python
threshold = 4

def label_it(row):
    if row['wild_stds'] >=threshold:
        return 'active'
    if row['wild_stds'] <threshold:
        return 'inactive'
    
e_coli_wild_efflux['wild_class'] = e_coli_wild_efflux.apply(label_it, axis=1)
```


```python
def label_it(row):
    if row['tolc_stds'] >=threshold:
        return 'active'
    if row['tolc_stds'] <threshold:
        return 'inactive'
    
    
e_coli_wild_efflux['tolc_class'] = e_coli_wild_efflux.apply(label_it, axis=1)
```


```python
def label_substrate(row):
    if row['tolc_class'] == 'active' and row['wild_class'] == 'inactive':
        return 'Efflux Substrate'
    if row['tolc_class'] == 'active' and row['wild_class'] == 'active':
        return 'Efflux Evader'
    if row['tolc_class'] == 'inactive' and row['wild_class'] == 'inactive':
        return 'Inactive'
    if row['tolc_class'] == 'inactive' and row['wild_class'] == 'active':
        return 'WT-only Active'
```


```python
e_coli_wild_efflux['Class'] = e_coli_wild_efflux.apply(label_substrate, axis=1)
```


```python
e_coli_wild_efflux.Class.value_counts()
```




    Inactive            72724
    Efflux Substrate      760
    Efflux Evader         200
    WT-only Active         53
    Name: Class, dtype: int64



## Scatter plot WT vs tolC


```python
sns.set(rc={"figure.figsize":(10, 8)})
sns.set_style("ticks")

sns.scatterplot(data = e_coli_wild_efflux, x='INHIB_AVE_wild', y='INHIB_AVE_efflux', hue='Class', s=30)

sns.despine()

plt.legend(fontsize=20)

# plt.xlim([-120, 120])

plt.xlabel('E.coli WT Growth Inhibition (%)', fontsize=22);

plt.ylabel('E.coli $\Delta TolC$ Growth Inhibition (%)',  fontsize=22);

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.axvline(x=43.02,  color='red', linestyle='--', alpha=0.5)
plt.axhline(y=74.98,  color='red', linestyle='--', alpha=0.5)

plt.tight_layout()

# plt.savefig('figures/wild_tolc_class_scatter.png', dpi=600)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-11-5923647c4196> in <module>
          2 sns.set_style("ticks")
          3 
    ----> 4 sns.scatterplot(data = e_coli_wild_efflux, x='INHIB_AVE_wild', y='INHIB_AVE_efflux', hue='Class', s=30)
          5 
          6 sns.despine()


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/_decorators.py in inner_f(*args, **kwargs)
         44             )
         45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 46         return f(**kwargs)
         47     return inner_f
         48 


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/relational.py in scatterplot(x, y, hue, style, size, data, palette, hue_order, hue_norm, sizes, size_order, size_norm, markers, style_order, x_bins, y_bins, units, estimator, ci, n_boot, alpha, x_jitter, y_jitter, legend, ax, **kwargs)
        803         x_bins=x_bins, y_bins=y_bins,
        804         estimator=estimator, ci=ci, n_boot=n_boot,
    --> 805         alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend,
        806     )
        807 


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/relational.py in __init__(self, data, variables, x_bins, y_bins, estimator, ci, n_boot, alpha, x_jitter, y_jitter, legend)
        585         )
        586 
    --> 587         super().__init__(data=data, variables=variables)
        588 
        589         self.alpha = alpha


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/_core.py in __init__(self, data, variables)
        602     def __init__(self, data=None, variables={}):
        603 
    --> 604         self.assign_variables(data, variables)
        605 
        606         for var, cls in self._semantic_mappings.items():


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/_core.py in assign_variables(self, data, variables)
        666             self.input_format = "long"
        667             plot_data, variables = self._assign_variables_longform(
    --> 668                 data, **variables,
        669             )
        670 


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/_core.py in _assign_variables_longform(self, data, **kwargs)
        900 
        901                 err = f"Could not interpret value `{val}` for parameter `{key}`"
    --> 902                 raise ValueError(err)
        903 
        904             else:


    ValueError: Could not interpret value `Class` for parameter `hue`



```python
e_coli_wild_efflux[e_coli_wild_efflux['wild_stds']>4].sort_values(by='wild_stds')
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2896             try:
    -> 2897                 return self._engine.get_loc(key)
       2898             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'wild_stds'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-5-58ffacf6e674> in <module>
    ----> 1 e_coli_wild_efflux[e_coli_wild_efflux['wild_stds']>4].sort_values(by='wild_stds')
    

    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2993             if self.columns.nlevels > 1:
       2994                 return self._getitem_multilevel(key)
    -> 2995             indexer = self.columns.get_loc(key)
       2996             if is_integer(indexer):
       2997                 indexer = [indexer]


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2897                 return self._engine.get_loc(key)
       2898             except KeyError:
    -> 2899                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2900         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2901         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'wild_stds'



```python
e_coli_wild_efflux[e_coli_wild_efflux['wild_stds']>=4].sort_values(by='wild_stds')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>INHIB_AVE_wild</th>
      <th>INHIB_AVE_efflux</th>
      <th>Mol</th>
      <th>fps</th>
      <th>abs_diff</th>
      <th>sub_class</th>
      <th>wild_stds</th>
      <th>tolc_stds</th>
      <th>wild_class</th>
      <th>tolc_class</th>
      <th>evader_class</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36040</th>
      <td>CCCCCCCCCCCNC(=O)c1ccc(NC(=O)c2cccc(C)c2)cc1</td>
      <td>43.02</td>
      <td>85.55</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>42.53</td>
      <td>increase</td>
      <td>4.002149</td>
      <td>4.624118</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>25810</th>
      <td>Br.O=C(CN(C1=NCCCCC1)c1ccc(Br)cc1)c1ccc(C2CCCC...</td>
      <td>43.06</td>
      <td>88.91</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>45.85</td>
      <td>increase</td>
      <td>4.006259</td>
      <td>4.820942</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>29584</th>
      <td>CC1(c2cccc([N+](=O)[O-])c2)Nc2ccccc2C(=O)N1/N=...</td>
      <td>43.13</td>
      <td>76.34</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>33.21</td>
      <td>increase</td>
      <td>4.013451</td>
      <td>4.084608</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>18822</th>
      <td>O=C(NN=C1CSc2sc(=O)sc2SC1)c1ccccc1O</td>
      <td>43.32</td>
      <td>57.73</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>14.41</td>
      <td>increase</td>
      <td>4.032973</td>
      <td>2.994459</td>
      <td>active</td>
      <td>inactive</td>
      <td>Unclassified</td>
      <td>Unclassified</td>
    </tr>
    <tr>
      <th>38895</th>
      <td>Oc1c(I)cc(I)c2cccnc12</td>
      <td>43.38</td>
      <td>99.77</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>56.39</td>
      <td>increase</td>
      <td>4.039138</td>
      <td>5.457107</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>28235</th>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>101.08</td>
      <td>95.52</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>-5.56</td>
      <td>decrease</td>
      <td>9.967611</td>
      <td>5.208147</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>21558</th>
      <td>CCN1CCN(c2cc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)CC1</td>
      <td>101.15</td>
      <td>101.88</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>0.73</td>
      <td>increase</td>
      <td>9.974803</td>
      <td>5.580708</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>67523</th>
      <td>O=[N+]([O-])c1ccc(C=[N+]([O-])CCO)o1</td>
      <td>101.33</td>
      <td>101.32</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...</td>
      <td>-0.01</td>
      <td>decrease</td>
      <td>9.993298</td>
      <td>5.547904</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
    <tr>
      <th>21276</th>
      <td>CSC1SCC2C(=O)N(C)C(C(C)C)C(=O)OCC(NC(=O)c3cnc4...</td>
      <td>101.44</td>
      <td>66.39</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>-35.05</td>
      <td>decrease</td>
      <td>10.004600</td>
      <td>3.501750</td>
      <td>active</td>
      <td>inactive</td>
      <td>Unclassified</td>
      <td>Unclassified</td>
    </tr>
    <tr>
      <th>71989</th>
      <td>COC1=C(N)C(=O)c2nc(-c3nc(C(=O)O)c(C)c(-c4ccc(O...</td>
      <td>102.37</td>
      <td>102.22</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>-0.15</td>
      <td>decrease</td>
      <td>10.100154</td>
      <td>5.600624</td>
      <td>active</td>
      <td>active</td>
      <td>Evader</td>
      <td>Evader</td>
    </tr>
  </tbody>
</table>
<p>253 rows × 13 columns</p>
</div>




```python
# 4x log_2 folds

# e_coli_wild_efflux['log_2_fold_change_wild'] = np.log2(e_coli_wild_efflux.INHIB_AVE_wild / e_coli_wild_efflux.INHIB_AVE_wild.mean())
# e_coli_wild_efflux['log_2_fold_change_efflux'] = np.log2(e_coli_wild_efflux.INHIB_AVE_efflux / e_coli_wild_efflux.INHIB_AVE_efflux.mean())

```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/pandas/core/series.py:856: RuntimeWarning: divide by zero encountered in log2
      result = getattr(ufunc, method)(*inputs, **kwargs)
    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/pandas/core/series.py:856: RuntimeWarning: invalid value encountered in log2
      result = getattr(ufunc, method)(*inputs, **kwargs)


## Define substrate, evader, rest, wt-only


```python
e_coli_wild_efflux['SMILES'] = e_coli_wild_efflux['SMILES'].apply(Chem.CanonSmiles)
```


```python
efflux_substrate = e_coli_wild_efflux[e_coli_wild_efflux['Class']=='Efflux Substrate']

efflux_evader = e_coli_wild_efflux[e_coli_wild_efflux['Class']=='Efflux Evader']

wt_only = e_coli_wild_efflux[e_coli_wild_efflux['Class']=='WT-only Active']

inactive = e_coli_wild_efflux[e_coli_wild_efflux['Class']=='Inactive']

sub_and_evade = efflux_evader.append(efflux_substrate).reset_index(drop=True)

# efflux_substrate.to_pickle('data_curated/efflux_substrate.pkl')
# efflux_evader.to_pickle('data_curated/efflux_evader.pkl')
# wt_only.to_pickle('data_curated/wt_only.pkl')
# inactive.to_pickle('data_curated/inactive.pkl')
# sub_and_evade.to_pickle('data_curated/sub_and_evade.pkl')


```

## OM Bias


```python
# Efflux evaders that are also permeators:

efflux_evaders_om_corrected = efflux_evader[efflux_evader['SMILES'].isin(om_permeating['SMILES'])]
```

Out of 200 efflux evades 186 are also permeators


```python
# Efflux substrates that are also non-permeators:

efflux_substrates_om_corrected = efflux_substrate[efflux_substrate['SMILES'].isin(om_non_permeating['SMILES'])]
```

Out of 760 efflux substrates 206 are non-permeators


```python
#drop 'em

efflux_substrates_om_corrected = efflux_substrate[~efflux_substrate['SMILES'].isin(om_non_permeating['SMILES'])]
```


```python
efflux_evaders_om_corrected.to_pickle('data_curated/efflux_evaders_om_corrected.pkl')

efflux_substrates_om_corrected.to_pickle('data_curated/efflux_substrates_om_corrected.pkl')
```

## TSNE of evader vs substrate


```python
sub_and_evade_om_corrected_tsne = master_functions.tsne_no_plot(sub_and_evade_om_corrected['fps'], perp=50)

fig, ax = plt.subplots(figsize=(12,12))
sns.scatterplot(x='TC1',y='TC2',data=sub_and_evade_om_corrected_tsne, s=30 ,alpha=0.9, hue=sub_and_evade_om_corrected['Class']) 

fig, ax = plt.subplots(figsize=(12,12))

sns.kdeplot(x='TC1',y='TC2',data=sub_and_evade_om_corrected_tsne,alpha=0.7, hue=sub_and_evade_om_corrected['Class'], levels = 4)
```

    [t-SNE] Computing 151 nearest neighbors...
    [t-SNE] Indexed 740 samples in 0.026s...
    [t-SNE] Computed neighbors for 740 samples in 0.552s...
    [t-SNE] Computed conditional probabilities for sample 740 / 740
    [t-SNE] Mean sigma: 1.234652
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 61.141048
    [t-SNE] KL divergence after 1000 iterations: 0.610763





    <matplotlib.axes._subplots.AxesSubplot at 0x2b2236dea250>




    
![png](co_add_chaser_files/co_add_chaser_85_2.png)
    



    
![png](co_add_chaser_files/co_add_chaser_85_3.png)
    


## TSNE of evades + substrates + rest of compounds(sample)


```python
sub_evade_rest_sample = sub_and_evade_om_corrected.append(inactive.sample(3000)).reset_index(drop=True)

sub_evade_rest_sample_tsne = master_functions.tsne_no_plot(sub_evade_rest_sample['fps'], perp=30)
```

    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 3740 samples in 3.440s...
    [t-SNE] Computed neighbors for 3740 samples in 103.436s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 3740
    [t-SNE] Computed conditional probabilities for sample 2000 / 3740
    [t-SNE] Computed conditional probabilities for sample 3000 / 3740
    [t-SNE] Computed conditional probabilities for sample 3740 / 3740
    [t-SNE] Mean sigma: 1.709593
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 77.862297
    [t-SNE] KL divergence after 1000 iterations: 1.211201



```python
sub_evade_rest_sample.Class.value_counts()
```




    Inactive            3000
    Efflux Substrate     554
    Efflux Evader        186
    Name: Class, dtype: int64




```python
fig, ax = plt.subplots(figsize=(12,12))

sns.scatterplot(x='TC1',y='TC2',data=sub_evade_rest_sample_tsne, s=30 ,alpha=0.9, hue=sub_evade_rest_sample['Class']) 

fig, ax = plt.subplots(figsize=(10,8))

sns.set(font_scale=1.5)

sns.set_style("ticks")

sns.kdeplot(x='TC1',y='TC2',data=sub_evade_rest_sample_tsne,alpha=0.7, hue=sub_evade_rest_sample['Class'], levels = 3, linewidths=3 )

sns.despine()

plt.tight_layout()

plt.savefig('figures/tsne_all_contour.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_89_0.png)
    



    
![png](co_add_chaser_files/co_add_chaser_89_1.png)
    


# LogD v MW


```python
sub_and_evade_logd['Class'] = sub_and_evade_om_corrected['Class']
```


```python
sub_and_evade_logd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>SMILES</th>
      <th>logS</th>
      <th>logS @ pH7.4</th>
      <th>logD</th>
      <th>2C9 pKi</th>
      <th>logP</th>
      <th>MW</th>
      <th>HBD</th>
      <th>HBA</th>
      <th>TPSA</th>
      <th>Flexibility</th>
      <th>Rotatable Bonds</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>OB1OCc2ccccc21</td>
      <td>5.188</td>
      <td>2.2370</td>
      <td>0.07439</td>
      <td>4.217</td>
      <td>0.07439</td>
      <td>133.9</td>
      <td>1</td>
      <td>2</td>
      <td>29.46</td>
      <td>0.00000</td>
      <td>0</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>BrC(/C=N/Nc1nc(N2CCOCC2)nc(N2CCOCC2)n1)=C/c1cc...</td>
      <td>2.053</td>
      <td>0.4994</td>
      <td>2.27200</td>
      <td>5.529</td>
      <td>2.78000</td>
      <td>474.4</td>
      <td>1</td>
      <td>9</td>
      <td>88.00</td>
      <td>0.18180</td>
      <td>6</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>c1cc(sc1C(=C2CN3CCC2CC3)c4ccc(s4)Cl)Cl</td>
      <td>1.303</td>
      <td>0.8745</td>
      <td>3.51100</td>
      <td>5.096</td>
      <td>4.87400</td>
      <td>356.3</td>
      <td>0</td>
      <td>1</td>
      <td>3.24</td>
      <td>0.08333</td>
      <td>2</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>2.361</td>
      <td>2.2380</td>
      <td>1.63100</td>
      <td>4.581</td>
      <td>3.76600</td>
      <td>295.1</td>
      <td>1</td>
      <td>2</td>
      <td>37.30</td>
      <td>0.18750</td>
      <td>3</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>O=C(CCl)C(=O)Nc1ccccc1</td>
      <td>4.326</td>
      <td>2.9250</td>
      <td>1.00300</td>
      <td>3.932</td>
      <td>1.00300</td>
      <td>197.6</td>
      <td>1</td>
      <td>3</td>
      <td>46.17</td>
      <td>0.30770</td>
      <td>4</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>735</th>
      <td>735</td>
      <td>c1ccc2c(c1)ccc3c2nc4ccccn34</td>
      <td>1.606</td>
      <td>1.6420</td>
      <td>4.15400</td>
      <td>4.902</td>
      <td>4.15400</td>
      <td>218.3</td>
      <td>0</td>
      <td>2</td>
      <td>17.30</td>
      <td>0.00000</td>
      <td>0</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>736</th>
      <td>736</td>
      <td>O=C(CSc1ccc2ccccc2n1)N/N=C/c1ccc(O)cc1O</td>
      <td>1.119</td>
      <td>2.5010</td>
      <td>2.21900</td>
      <td>4.954</td>
      <td>2.21900</td>
      <td>353.4</td>
      <td>3</td>
      <td>6</td>
      <td>94.81</td>
      <td>0.22220</td>
      <td>6</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>737</th>
      <td>737</td>
      <td>Cc1c2ccncc2c(C)c2c1[nH]c1ccccc12</td>
      <td>1.294</td>
      <td>0.9868</td>
      <td>4.80000</td>
      <td>5.346</td>
      <td>4.80000</td>
      <td>246.3</td>
      <td>1</td>
      <td>2</td>
      <td>28.68</td>
      <td>0.00000</td>
      <td>0</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>738</th>
      <td>738</td>
      <td>Cc1cc(C)c(CSc2nnc(C)s2)c(C)c1</td>
      <td>1.607</td>
      <td>2.4660</td>
      <td>3.86300</td>
      <td>4.569</td>
      <td>3.86300</td>
      <td>264.4</td>
      <td>0</td>
      <td>2</td>
      <td>25.78</td>
      <td>0.16670</td>
      <td>3</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>739</th>
      <td>739</td>
      <td>COc1cc([C@@H]2c3cc4c(cc3[C@@H](OC3OC5CO[C@@H](...</td>
      <td>1.052</td>
      <td>2.1080</td>
      <td>1.28600</td>
      <td>5.984</td>
      <td>1.28600</td>
      <td>656.7</td>
      <td>3</td>
      <td>13</td>
      <td>160.80</td>
      <td>0.11320</td>
      <td>6</td>
      <td>Efflux Substrate</td>
    </tr>
  </tbody>
</table>
<p>740 rows × 14 columns</p>
</div>




```python
sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Substrate']['logS @ pH7.4'].mean()
```




    2.193845270758123




```python
sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Evader']['logS @ pH7.4'].mean()
```




    2.7207848387096774




```python
sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Substrate'].TPSA.mean()
```




    76.03005415162455




```python
sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Evader'].TPSA.mean()
```




    96.14951612903226




```python
sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Evader']['Flexibility'].mean(), sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Substrate']['Flexibility'].mean()
```




    (0.1854570430107527, 0.19158178700361012)




```python
sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Evader']['Rotatable Bonds'].mean(), sub_and_evade_logd[sub_and_evade_logd['Class']=='Efflux Substrate']['Rotatable Bonds'].mean()
```




    (4.887096774193548, 5.729241877256317)




```python
# sns.set(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.4, color_codes=False, rc=None)


fg = sns.jointplot(data = sub_and_evade_logd, x= 'MW', y='logD', hue='Class', alpha=0.7, height = 9).plot_joint(sns.kdeplot,n_levels=4, linewidths=2.5);

## axes labels

fg.ax_joint.set_xlabel('Molecular Weight', fontsize=25)

fg.ax_joint.set_ylabel('LogD$_{7.4}$', fontsize=25)

##tick params

fg.ax_joint.tick_params(labelsize=22)

sns.despine()


plt.savefig('figures/logd_mw.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_99_0.png)
    



```python
fig, ax = plt.subplots(figsize=(8,8))


sns.set(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.3, color_codes=False, rc=None)

fig = sns.scatterplot(data = sub_and_evade_logd, x= 'MW', y='logD', hue='Class', alpha=0.6)
fig = sns.kdeplot(data = sub_and_evade_logd, x= 'MW', y='logD', hue='Class', alpha=0.6, levels=1, linewidths=2)

sns.despine()
```


    
![png](co_add_chaser_files/co_add_chaser_100_0.png)
    


## PCA of evader vs substrate


```python
sub_and_evade_features = master_functions.calc_feats(sub_and_evade_om_corrected['Mol'])
sub_and_evade_features['Class'] = sub_and_evade_om_corrected['Class']
```


```python
sub_and_evade_features
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MolWt</th>
      <th>LogP</th>
      <th>NumHAcceptors</th>
      <th>NumHDonors</th>
      <th>NumHeteroatoms</th>
      <th>NumRotatableBonds</th>
      <th>NumHeavyAtoms</th>
      <th>NumAliphaticCarbocycles</th>
      <th>NumAliphaticHeterocycles</th>
      <th>NumAliphaticRings</th>
      <th>NumAromaticCarbocycles</th>
      <th>NumAromaticHeterocycles</th>
      <th>NumAromaticRings</th>
      <th>RingCount</th>
      <th>FractionCSP3</th>
      <th>TPSA</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>133.943</td>
      <td>-0.09570</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.142857</td>
      <td>29.46</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>1</th>
      <td>474.363</td>
      <td>2.37850</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.400000</td>
      <td>88.00</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>2</th>
      <td>392.804</td>
      <td>6.06570</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.375000</td>
      <td>3.24</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>3</th>
      <td>295.054</td>
      <td>3.47940</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.100000</td>
      <td>37.30</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197.621</td>
      <td>1.43300</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>46.17</td>
      <td>Efflux Evader</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>735</th>
      <td>254.720</td>
      <td>4.06250</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.000000</td>
      <td>17.30</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>736</th>
      <td>353.403</td>
      <td>2.88840</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.055556</td>
      <td>94.81</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>737</th>
      <td>246.313</td>
      <td>4.48614</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.117647</td>
      <td>28.68</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>738</th>
      <td>264.419</td>
      <td>4.06408</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.384615</td>
      <td>25.78</td>
      <td>Efflux Substrate</td>
    </tr>
    <tr>
      <th>739</th>
      <td>656.662</td>
      <td>2.75290</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.468750</td>
      <td>160.83</td>
      <td>Efflux Substrate</td>
    </tr>
  </tbody>
</table>
<p>740 rows × 17 columns</p>
</div>




```python
sub_and_evade_features_200 = master_functions.calcualte_features_single(sub_and_evade_om_corrected, 'SMILES')
sub_and_evade_features_200['Class'] = sub_and_evade_om_corrected['Class']
```

      1%|▏         | 10/740 [00:00<00:08, 90.64it/s]

    Computing features: 


    100%|██████████| 740/740 [00:09<00:00, 81.68it/s] 
    100%|██████████| 740/740 [00:00<00:00, 1179.93it/s]



```python
sub_and_evade_features_200 = sub_and_evade_features_200.dropna(axis=1)
```


```python
#pca

table = sub_and_evade_features_200

# descriptors = table[['MolWt', 'LogP','NumHeteroatoms','RingCount','FractionCSP3', 'TPSA']].values #The non-redundant molecular descriptors chosen for PCA

# descriptors=table.iloc[:,:-1]

descriptors  = table.iloc[:,:-87]

descriptors  = table.iloc[:,:-180]


descriptors_std = StandardScaler().fit_transform(descriptors) #Important to avoid scaling problems between our different descriptors
pca = PCA()
descriptors_2d = pca.fit_transform(descriptors_std)
descriptors_pca= pd.DataFrame(descriptors_2d) # Saving PCA values to a new table
descriptors_pca.index = table.index
descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
descriptors_pca.head(5) #Displays the PCA table

scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1'])) 
scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

# And we add the new values to our PCA table
descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]


descriptors_pca['Class'] = sub_and_evade_features['Class']



# plt.rcParams['axes.linewidth'] = 1.5


cmap = sns.diverging_palette(133, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,5))

sns.scatterplot(x='PC1',y='PC2',data=descriptors_pca, alpha=0.7, hue='Class', s=20)#, palette=["C0", "C1", "C2", "k"])


pca_lab = ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontsize=16,fontweight='bold')
plt.ylabel(pca_lab[1],fontsize=16,fontweight='bold')

plt.tick_params ('both',width=2,labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

plt.tight_layout()

# plt.savefig('figures/pca_evade_substrate.png', dpi=600)

plt.show()

print('same but in contours, for ease of read')

cmap = sns.diverging_palette(133, 10, as_cmap=True)


############ kdeplot


fig, ax = plt.subplots(figsize=(10,7))

sns.set_style("ticks")

# sns.set(font_scale=2)

sns.kdeplot(x='PC1',y='PC2',data=descriptors_pca, hue='Class' , levels=3,)


pca_lab= ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontweight='bold',fontsize=22)
plt.ylabel(pca_lab[1],fontweight='bold', fontsize=22)

plt.tick_params ('both',width=2,labelsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

# plt.legend()

plt.tight_layout()

# plt.savefig('figures/pca_evade_substrate_contour.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_106_0.png)
    


    same but in contours, for ease of read



    
![png](co_add_chaser_files/co_add_chaser_106_2.png)
    



```python
#pca

table = sub_and_evade_features

descriptors = table[['MolWt', 'LogP','NumHeteroatoms','RingCount','FractionCSP3', 'TPSA']].values #The non-redundant molecular descriptors chosen for PCA

# descriptors=table.iloc[:,:-1]

descriptors_std = StandardScaler().fit_transform(descriptors) #Important to avoid scaling problems between our different descriptors
pca = PCA()
descriptors_2d = pca.fit_transform(descriptors_std)
descriptors_pca= pd.DataFrame(descriptors_2d) # Saving PCA values to a new table
descriptors_pca.index = table.index
descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
descriptors_pca.head(5) #Displays the PCA table

scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1'])) 
scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

# And we add the new values to our PCA table
descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]


descriptors_pca['Class'] = sub_and_evade_features['Class']



# plt.rcParams['axes.linewidth'] = 1.5


cmap = sns.diverging_palette(133, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,5))

sns.scatterplot(x='PC1',y='PC2',data=descriptors_pca, alpha=0.7, hue='Class', s=20)#, palette=["C0", "C1", "C2", "k"])


pca_lab = ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontsize=16,fontweight='bold')
plt.ylabel(pca_lab[1],fontsize=16,fontweight='bold')

plt.tick_params ('both',width=2,labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

plt.tight_layout()

# plt.savefig('figures/pca_evade_substrate.png', dpi=600)

plt.show()

print('same but in contours, for ease of read')

cmap = sns.diverging_palette(133, 10, as_cmap=True)


############ kdeplot


fig, ax = plt.subplots(figsize=(10,7))

sns.set_style("ticks")

# sns.set(font_scale=2)

sns.kdeplot(x='PC1',y='PC2',data=descriptors_pca, hue='Class' , levels=3,)


pca_lab= ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontweight='bold',fontsize=22)
plt.ylabel(pca_lab[1],fontweight='bold', fontsize=22)

plt.tick_params ('both',width=2,labelsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

# plt.legend()

plt.tight_layout()

# plt.savefig('figures/pca_evade_substrate_contour.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_107_0.png)
    


    same but in contours, for ease of read



    
![png](co_add_chaser_files/co_add_chaser_107_2.png)
    


## PCA of evader _ substrates_ rest of compounds (sample)


```python
rest_vs_evade_sub = sub_and_evade_om_corrected.append(inactive.sample(3000), sort=False).reset_index(drop=True)
```


```python
rest_vs_evade_sub_features = master_functions.calc_feats(rest_vs_evade_sub['Mol'])
rest_vs_evade_sub_features['Class'] = rest_vs_evade_sub['Class']
```


```python
rest_vs_evade_sub_features.Class.value_counts()
```




    Inactive            3000
    Efflux Substrate     554
    Efflux Evader        186
    Name: Class, dtype: int64



# Physicochemical feature comaprison


```python
rest_vs_evade_sub_features[rest_vs_evade_sub_features['Class']=='Efflux Substrate'].LogP.mean()
```




    3.826753321299641




```python
rest_vs_evade_sub_features[rest_vs_evade_sub_features['Class']=='Efflux Evader'].LogP.mean()
```




    2.338045215053765




```python
rest_vs_evade_sub_features[rest_vs_evade_sub_features['Class']=='Efflux Substrate'].MolWt.mean()
```




    422.1415559566788




```python
rest_vs_evade_sub_features[rest_vs_evade_sub_features['Class']=='Efflux Evader'].MolWt.mean()
```




    383.9766021505377




```python
#pca

table = rest_vs_evade_sub_features

descriptors = table[['MolWt', 'LogP', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds']].values #The non-redundant molecular descriptors chosen for PCA

# Molecular weight, hydrogen bond donors and acceptors, topological polar surface area, rotatable bounds
# and heavy atom count are the major contributing factors to PC1, whereas LogP, 
# logD and the fraction of sp3 hybridized carbon atoms (Fsp3) are major contributors to PC2.

# descriptors=table.iloc[:,2:]

descriptors_std = StandardScaler().fit_transform(descriptors) #Important to avoid scaling problems between our different descriptors
pca = PCA()
descriptors_2d = pca.fit_transform(descriptors_std)
descriptors_pca= pd.DataFrame(descriptors_2d) # Saving PCA values to a new table
descriptors_pca.index = table.index
descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
descriptors_pca.head(5) #Displays the PCA table

scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1'])) 
scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

# And we add the new values to our PCA table
descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]


descriptors_pca['Class'] = rest_vs_evade_sub['Class']

# plt.rcParams['axes.linewidth'] = 1.5

cmap = sns.diverging_palette(133, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(10,5))

sns.set_style("ticks")

sns.scatterplot(x='PC1',y='PC2',data=descriptors_pca, alpha=1, hue='Class', s=5)#, palette=["C0", "C1", "C2", "k"])


pca_lab= ('PC1 '+str([np.round(pca.explained_variance_ratio_[0]*100, 1)]), 'PC2 '+str([np.round(pca.explained_variance_ratio_[1]*100, 1)]))


plt.xlabel(pca_lab[0],fontsize=16,fontweight='bold')
plt.ylabel(pca_lab[1],fontsize=16,fontweight='bold')

plt.tick_params ('both',width=2,labelsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

#ax.legend(handles=handles[1:], labels=labels[1:])

#plt.legend(loc='lower right',frameon=False,prop={'size': 22},ncol=1)

plt.tight_layout()
plt.show()

print('same but in contours, for ease of read')

cmap = sns.diverging_palette(133, 10, as_cmap=True)


############# kdeplot



fig, ax = plt.subplots(figsize=(9,9))


# sns.set(font_scale=1.5)

sns.set_style("ticks")


sns.kdeplot(x='PC1',y='PC2',data=descriptors_pca, hue='Class' , levels=3, linewidths=2.5)


pca_lab= ('PC1 - '+str(np.round(pca.explained_variance_ratio_[0]*100, 1)) + '%', 'PC2 - '+str(np.round(pca.explained_variance_ratio_[1]*100, 1))  + '%')



plt.xlabel(pca_lab[0],fontsize=24)
plt.ylabel(pca_lab[1],fontsize=24)

plt.tick_params('both',labelsize=21)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()

major_ticks = np.arange(-7, 10, 3)

ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)

plt.xlim([-7,10])
plt.ylim([-7,7])

plt.tight_layout()
# plt.show()

plt.savefig('figures/pca_all_contour.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_117_0.png)
    


    same but in contours, for ease of read



    
![png](co_add_chaser_files/co_add_chaser_117_2.png)
    



```python
# sns.set(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.4, color_codes=False, rc=None)


# fg = sns.jointplot(data = sub_and_evade_logd, x= 'MW', y='logD', hue='Class', alpha=0.7, height = 9).plot_joint(sns.kdeplot,n_levels=4, linewidths=2.5);


fig, ax = plt.subplots(figsize=(11,9))

fg = sns.jointplot(x='PC1',y='PC2',data=descriptors_pca, hue='Class', size=10, alpha=0.5, height = 9).plot_joint(sns.kdeplot,n_levels=4, linewidths=2.5);


# sns.scatterplot(x='PC1',y='PC2',data=descriptors_pca, hue='Class', size=10, alpha=0.5)
# sns.kdeplot(x='PC1',y='PC2',data=descriptors_pca, n_levels=4, linewidths=2.5, hue='Class');


## axes labels


pca_lab= ('PC1 - '+str(np.round(pca.explained_variance_ratio_[0]*100, 1)) + '%', 'PC2 - '+str(np.round(pca.explained_variance_ratio_[1]*100, 1))  + '%')

plt.xlabel(pca_lab[0],fontsize=24)
plt.ylabel(pca_lab[1],fontsize=24)

plt.tick_params('both',labelsize=21)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()



sns.despine()


# plt.savefig('figures/logd_mw.png', dpi=600)
```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/seaborn/axisgrid.py:2073: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)



    
![png](co_add_chaser_files/co_add_chaser_118_1.png)
    



    
![png](co_add_chaser_files/co_add_chaser_118_2.png)
    


## Mol vs LogP


```python
list(sub_and_evade_features_200.columns)
```




    ['BalabanJ',
     'BertzCT',
     'Chi0',
     'Chi0n',
     'Chi0v',
     'Chi1',
     'Chi1n',
     'Chi1v',
     'Chi2n',
     'Chi2v',
     'Chi3n',
     'Chi3v',
     'Chi4n',
     'Chi4v',
     'EState_VSA1',
     'EState_VSA10',
     'EState_VSA11',
     'EState_VSA2',
     'EState_VSA3',
     'EState_VSA4',
     'EState_VSA5',
     'EState_VSA6',
     'EState_VSA7',
     'EState_VSA8',
     'EState_VSA9',
     'ExactMolWt',
     'FpDensityMorgan1',
     'FpDensityMorgan2',
     'FpDensityMorgan3',
     'FractionCSP3',
     'HallKierAlpha',
     'HeavyAtomCount',
     'HeavyAtomMolWt',
     'Ipc',
     'Kappa1',
     'Kappa2',
     'Kappa3',
     'LabuteASA',
     'MaxAbsEStateIndex',
     'MaxEStateIndex',
     'MinAbsEStateIndex',
     'MinEStateIndex',
     'MolLogP',
     'MolMR',
     'MolWt',
     'NHOHCount',
     'NOCount',
     'NumAliphaticCarbocycles',
     'NumAliphaticHeterocycles',
     'NumAliphaticRings',
     'NumAromaticCarbocycles',
     'NumAromaticHeterocycles',
     'NumAromaticRings',
     'NumHAcceptors',
     'NumHDonors',
     'NumHeteroatoms',
     'NumRadicalElectrons',
     'NumRotatableBonds',
     'NumSaturatedCarbocycles',
     'NumSaturatedHeterocycles',
     'NumSaturatedRings',
     'NumValenceElectrons',
     'PEOE_VSA1',
     'PEOE_VSA10',
     'PEOE_VSA11',
     'PEOE_VSA12',
     'PEOE_VSA13',
     'PEOE_VSA14',
     'PEOE_VSA2',
     'PEOE_VSA3',
     'PEOE_VSA4',
     'PEOE_VSA5',
     'PEOE_VSA6',
     'PEOE_VSA7',
     'PEOE_VSA8',
     'PEOE_VSA9',
     'RingCount',
     'SMR_VSA1',
     'SMR_VSA10',
     'SMR_VSA2',
     'SMR_VSA3',
     'SMR_VSA4',
     'SMR_VSA5',
     'SMR_VSA6',
     'SMR_VSA7',
     'SMR_VSA8',
     'SMR_VSA9',
     'SlogP_VSA1',
     'SlogP_VSA10',
     'SlogP_VSA11',
     'SlogP_VSA12',
     'SlogP_VSA2',
     'SlogP_VSA3',
     'SlogP_VSA4',
     'SlogP_VSA5',
     'SlogP_VSA6',
     'SlogP_VSA7',
     'SlogP_VSA8',
     'SlogP_VSA9',
     'TPSA',
     'VSA_EState1',
     'VSA_EState10',
     'VSA_EState2',
     'VSA_EState3',
     'VSA_EState4',
     'VSA_EState5',
     'VSA_EState6',
     'VSA_EState7',
     'VSA_EState8',
     'VSA_EState9',
     'fr_Al_COO',
     'fr_Al_OH',
     'fr_Al_OH_noTert',
     'fr_ArN',
     'fr_Ar_COO',
     'fr_Ar_N',
     'fr_Ar_NH',
     'fr_Ar_OH',
     'fr_COO',
     'fr_COO2',
     'fr_C_O',
     'fr_C_O_noCOO',
     'fr_C_S',
     'fr_HOCCN',
     'fr_Imine',
     'fr_NH0',
     'fr_NH1',
     'fr_NH2',
     'fr_N_O',
     'fr_Ndealkylation1',
     'fr_Ndealkylation2',
     'fr_Nhpyrrole',
     'fr_SH',
     'fr_aldehyde',
     'fr_alkyl_carbamate',
     'fr_alkyl_halide',
     'fr_allylic_oxid',
     'fr_amide',
     'fr_amidine',
     'fr_aniline',
     'fr_aryl_methyl',
     'fr_azide',
     'fr_azo',
     'fr_barbitur',
     'fr_benzene',
     'fr_benzodiazepine',
     'fr_bicyclic',
     'fr_diazo',
     'fr_dihydropyridine',
     'fr_epoxide',
     'fr_ester',
     'fr_ether',
     'fr_furan',
     'fr_guanido',
     'fr_halogen',
     'fr_hdrzine',
     'fr_hdrzone',
     'fr_imidazole',
     'fr_imide',
     'fr_isocyan',
     'fr_isothiocyan',
     'fr_ketone',
     'fr_ketone_Topliss',
     'fr_lactam',
     'fr_lactone',
     'fr_methoxy',
     'fr_morpholine',
     'fr_nitrile',
     'fr_nitro',
     'fr_nitro_arom',
     'fr_nitro_arom_nonortho',
     'fr_nitroso',
     'fr_oxazole',
     'fr_oxime',
     'fr_para_hydroxylation',
     'fr_phenol',
     'fr_phenol_noOrthoHbond',
     'fr_phos_acid',
     'fr_phos_ester',
     'fr_piperdine',
     'fr_piperzine',
     'fr_priamide',
     'fr_prisulfonamd',
     'fr_pyridine',
     'fr_quatN',
     'fr_sulfide',
     'fr_sulfonamd',
     'fr_sulfone',
     'fr_term_acetylene',
     'fr_tetrazole',
     'fr_thiazole',
     'fr_thiocyan',
     'fr_thiophene',
     'fr_unbrch_alkane',
     'fr_urea',
     'qed',
     'Class']




```python

fig, ax = plt.subplots(figsize=(10,7))

sns.scatterplot(data = sub_and_evade_features_200, x='MolWt', y='MaxAbsEStateIndex', hue='Class', hue_order=[ 'Efflux Evader', 'Efflux Substrate', 'Inactive'], alpha=0.8 )
plt.tight_layout()

# plt.xlim([0, 1200])
# plt.ylim([-5, 12])

# plt.savefig('figures/mol_logp_kde.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_121_0.png)
    



```python

fig, ax = plt.subplots(figsize=(10,7))

sns.kdeplot(data = sub_and_evade_features_200, x='MolWt', y='MinAbsEStateIndex', hue='Class', hue_order=[ 'Efflux Evader', 'Efflux Substrate', 'Inactive'], alpha=0.8, levels=1 )
plt.tight_layout()

# plt.xlim([0, 1200])
# plt.ylim([-5, 12])

# plt.savefig('figures/mol_logp_kde.png', dpi=600)
```


    
![png](co_add_chaser_files/co_add_chaser_122_0.png)
    


## Make smiles canonical so we can compare them:



```python
# make smiles canonical so we can compare them:

efflux_mmpa_index['compound_structure_B'] = efflux_mmpa_index.compound_structure_B.apply(Chem.CanonSmiles)
efflux_mmpa_index['compound_structure_A'] = efflux_mmpa_index.compound_structure_A.apply(Chem.CanonSmiles)

efflux_evader['SMILES'] = efflux_evader.SMILES.apply(Chem.CanonSmiles)
efflux_substrate['SMILES'] = efflux_substrate.SMILES.apply(Chem.CanonSmiles)

efflux_evader = efflux_evader.drop_duplicates(subset=['SMILES'])

efflux_substrate = efflux_substrate.drop_duplicates(subset=['SMILES'])

rest_of_ecoli_efflux['SMILES'] = rest_of_ecoli_efflux.SMILES.apply(Chem.CanonSmiles)

e_coli_wild_efflux['SMILES'] = e_coli_wild_efflux['SMILES'].apply(Chem.CanonSmiles)
```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys


# Transforms leading to efflux evaders


```python
# find efflux evaders 

evader_transforms = efflux_mmpa_index[(efflux_mmpa_index['compound_structure_B'].isin(efflux_evader.SMILES)) & (efflux_mmpa_index['compound_structure_A'].isin(rest_of_ecoli_efflux.SMILES))]

evader_transforms = master_functions.clean_mmpa_pairs_len(evader_transforms)

evader_transforms.to_pickle('data_curated/evader_transforms.pkl')
```

    Initial number of transofrms: 2880 
    Number fo transforms disqualified based on length discrepancy: 2268 
    Remaining number of transforms: 612



```python
len(evader_transforms.compound_structure_A.unique())
```




    418




```python
len(evader_transforms.compound_structure_B.unique())
```




    78




```python
len(evader_transforms)
```




    612



# Transforms leading to efflux substrates


```python
substrate_transforms = efflux_mmpa_index[(efflux_mmpa_index['compound_structure_B'].isin(efflux_substrate.SMILES)) & (efflux_mmpa_index['compound_structure_A'].isin(rest_of_ecoli_efflux.SMILES)) ]

substrate_transforms = master_functions.clean_mmpa_pairs_len(substrate_transforms)

substrate_transforms.to_pickle('data_curated/substrate_transforms.pkl')
```

    Initial number of transofrms: 10619 
    Number fo transforms disqualified based on length discrepancy: 2694 
    Remaining number of transforms: 7925



```python
len(substrate_transforms.compound_structure_A.unique())
```




    2799




```python
len(substrate_transforms.compound_structure_B.unique())
```




    488




```python
len(substrate_transforms)
```




    7925




```python
# e_coli_wild_efflux[e_coli_wild_efflux['SMILES'].isin(substrate_transforms.compound_structure_A)].INHIB_AVE_wild.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b3ce01c1e90>




    
![png](co_add_chaser_files/co_add_chaser_135_1.png)
    



```python
# e_coli_wild_efflux[e_coli_wild_efflux['SMILES'].isin(substrate_transforms.compound_structure_A)].INHIB_AVE_efflux.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b3ce0187310>




    
![png](co_add_chaser_files/co_add_chaser_136_1.png)
    


# Transfrom compound_A ovelap


```python
evader_transforms
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B</th>
      <th>idsmiles_A</th>
      <th>idsmiles_B</th>
      <th>smirks</th>
      <th>common_core</th>
      <th>measurement_A</th>
      <th>measurement_B</th>
      <th>measurement_delta</th>
      <th>LHS</th>
      <th>RHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5578</th>
      <td>O/C(=C\C=N\c1ccc(Br)cc1)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>752</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>15.84</td>
      <td>30.93</td>
      <td>15.09</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5581</th>
      <td>Cc1ccc(/N=C/C=C(\O)c2ccc(Br)cc2)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>753</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccc(C)cc1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>-1.73</td>
      <td>30.93</td>
      <td>32.66</td>
      <td>[*:1]/C=N\c1ccc(C)cc1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5583</th>
      <td>O/C(=C\C=N\c1cccc(Cl)c1)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>755</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>-21.10</td>
      <td>30.93</td>
      <td>52.03</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>13581</th>
      <td>COc1ccc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc1</td>
      <td>Cc1ccc(N2C(=O)NC(=O)/C(=C\c3ccc([N+](=O)[O-])o...</td>
      <td>31110</td>
      <td>31119</td>
      <td>[*:1]c1ccc(OC)cc1&gt;&gt;[*:1]c1ccc([N+](=O)[O-])o1</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
      <td>-2.80</td>
      <td>-6.37</td>
      <td>-3.57</td>
      <td>[*:1]c1ccc(OC)cc1</td>
      <td>[*:1]c1ccc([N+](=O)[O-])o1</td>
    </tr>
    <tr>
      <th>13602</th>
      <td>COc1cc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc(...</td>
      <td>Cc1ccc(N2C(=O)NC(=O)/C(=C\c3ccc([N+](=O)[O-])o...</td>
      <td>31112</td>
      <td>31119</td>
      <td>[*:1]c1cc(I)c(O)c(OC)c1&gt;&gt;[*:1]c1ccc([N+](=O)[O...</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
      <td>8.95</td>
      <td>-6.37</td>
      <td>-15.32</td>
      <td>[*:1]c1cc(I)c(O)c(OC)c1</td>
      <td>[*:1]c1ccc([N+](=O)[O-])o1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1404516</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28235</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>32.75</td>
      <td>47.27</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCC</td>
    </tr>
    <tr>
      <th>1404517</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404520</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28235</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>32.75</td>
      <td>19.03</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCC</td>
    </tr>
    <tr>
      <th>1404521</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1405306</th>
      <td>O=C([O-])Cn1cnnn1.[K+]</td>
      <td>O=[N+]([O-])c1nonc1NCn1cnnn1</td>
      <td>73296</td>
      <td>64759</td>
      <td>[*:1]CC(=O)[O-]&gt;&gt;[*:1]CNc1nonc1[N+](=O)[O-]</td>
      <td>[*:1]n1cnnn1</td>
      <td>-13.82</td>
      <td>-2.55</td>
      <td>11.27</td>
      <td>[*:1]CC(=O)[O-]</td>
      <td>[*:1]CNc1nonc1[N+](=O)[O-]</td>
    </tr>
  </tbody>
</table>
<p>612 rows × 11 columns</p>
</div>




```python
len(evader_transforms.merge(evader_transforms, on = ['compound_structure_A']).compound_structure_A.unique())
```




    418




```python
comp_a_evader_overlap = evader_transforms[evader_transforms.compound_structure_A.isin(substrate_transforms.compound_structure_A)]
```


```python
len(comp_a_evader_overlap)
```




    164




```python
len(comp_a_evader_overlap.compound_structure_B.unique())
```




    30




```python
comp_a_substrate_overlap = substrate_transforms[substrate_transforms.compound_structure_A.isin(evader_transforms.compound_structure_A)]
```


```python
len(comp_a_substrate_overlap.compound_structure_B.unique())
```




    71




```python
len(comp_a_substrate_overlap)
```




    377




```python
plt.figure(figsize=(10,10))


venn2([ set(evader_transforms.compound_structure_A.to_list()), 
        set(substrate_transforms.compound_structure_A.to_list())],
        set_labels=('Evader Comp_A', 'Substrate Comp_A')
     )
```




    <matplotlib_venn._common.VennDiagram at 0x2affa94fb610>




    
![png](co_add_chaser_files/co_add_chaser_146_1.png)
    



```python
len(substrate_transforms.compound_structure_A.unique())
```




    2799



## How are overlapping transforms split?


```python
len(comp_a_evader_overlap[comp_a_evader_overlap['compound_structure_A'].isin(evader_transforms.compound_structure_A)])
```




    89




```python
len(comp_a_evader_substrate_overlap.compound_structure_B.unique())
```




    96




```python
len(substrate_transforms[substrate_transforms['compound_structure_B'].isin(comp_a_evader_substrate_overlap.compound_structure_B)].compound_structure_B.unique())
```




    96




```python
len(comp_a_substrate_evader_overlap[comp_a_substrate_evader_overlap['compound_structure_B'].isin(evader_transforms.compound_structure_B)].compound_structure_B.unique())
```




    46




```python
len(evader_transforms[evader_transforms['compound_structure_B'].isin(comp_a_evader_substrate_overlap.compound_structure_B)].compound_structure_B.unique())
```




    0



# TSNE of RHS and LHS of overlap


```python
comp_a_substrate_overlap['mol_b'] = comp_a_substrate_overlap.compound_structure_B.apply(Chem.MolFromSmiles)
comp_a_substrate_overlap['label'] = 'substrate_overlap_B'
comp_a_substrate_overlap = comp_a_substrate_overlap.dropna(subset=['mol_b'])

comp_a_substrate_overlap['fps']=comp_a_substrate_overlap.mol_b.apply(MACCSkeys.GenMACCSKeys)

unique_substrate =  comp_a_substrate_overlap.drop_duplicates(subset=['compound_structure_B'])


```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
comp_a_evader_overlap['mol_b'] = comp_a_evader_overlap.compound_structure_B.apply(Chem.MolFromSmiles)
comp_a_evader_overlap['label'] = 'evader_overlap_B'
comp_a_evader_overlap = comp_a_evader_overlap.dropna(subset=['mol_b'])

comp_a_evader_overlap['fps']=comp_a_evader_overlap.mol_b.apply(MACCSkeys.GenMACCSKeys)

unique_evader =  comp_a_evader_overlap.drop_duplicates(subset=['compound_structure_B'])

```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
append = unique_evader.append(unique_substrate).reset_index(drop=True)
```


```python
append.label.value_counts()
```




    substrate_overlap_B    71
    evader_overlap_B       30
    Name: label, dtype: int64




```python
len(append)
```




    101




```python
append_tsne =  master_functions.tsne_no_plot(append['fps'], 5)
```

    [t-SNE] Computing 16 nearest neighbors...
    [t-SNE] Indexed 101 samples in 0.001s...
    [t-SNE] Computed neighbors for 101 samples in 0.007s...
    [t-SNE] Computed conditional probabilities for sample 101 / 101
    [t-SNE] Mean sigma: 0.341923
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 61.464149
    [t-SNE] KL divergence after 1000 iterations: 0.468673



```python
append_tsne =  master_functions.tsne_no_plot(append['fps'], 5)

fig, ax = plt.subplots(figsize=(12,12))
sns.scatterplot(x='TC1',y='TC2',data=append_tsne, s=20 ,alpha=0.7, hue=append['label']) 

fig, ax = plt.subplots(figsize=(12,12))

sns.kdeplot(x='TC1',y='TC2',data=append_tsne,alpha=0.7, hue=append['label'], levels = 3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ae654c56a50>




    
![png](co_add_chaser_files/co_add_chaser_161_1.png)
    



    
![png](co_add_chaser_files/co_add_chaser_161_2.png)
    


# Physchem of overlap


```python
unique_evader_transforms = comp_a_evader_overlap.drop_duplicates(subset=['compound_structure_A'])

unique_evader_transforms['Mol_A'] = unique_evader_transforms.compound_structure_A.apply(Chem.MolFromSmiles)
unique_evader_transforms['Mol_B'] = unique_evader_transforms.compound_structure_B.apply(Chem.MolFromSmiles)


unique_evader_transforms_comp_a = comp_a_evader_overlap.drop_duplicates(subset=['compound_structure_A'])[['compound_structure_A']]
unique_evader_transforms_comp_b = comp_a_evader_overlap.drop_duplicates(subset=['compound_structure_A'])[['compound_structure_B']]
```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.



```python
unique_substrate_transforms = comp_a_substrate_overlap.drop_duplicates(subset=['compound_structure_A'])

unique_substrate_transforms['Mol_A'] = unique_substrate_transforms.compound_structure_A.apply(Chem.MolFromSmiles)
unique_substrate_transforms['Mol_B'] = unique_substrate_transforms.compound_structure_B.apply(Chem.MolFromSmiles)

# unique_substrate_transforms_comp_a = comp_a_substrate_overlap.drop_duplicates(subset=['compound_structure_A'])[['compound_structure_A']]

# unique_substrate_transforms_comp_b = comp_a_substrate_overlap.drop_duplicates(subset=['compound_structure_A'])[['compound_structure_B']]
```

    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.



```python
unique_substrate_transforms_comp_b_feats = master_functions.calc_feats(unique_substrate_transforms.Mol_B)
```


```python
unique_substrate_transforms_comp_b_feats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MolWt</th>
      <th>LogP</th>
      <th>NumHAcceptors</th>
      <th>NumHDonors</th>
      <th>NumHeteroatoms</th>
      <th>NumRotatableBonds</th>
      <th>NumHeavyAtoms</th>
      <th>NumAliphaticCarbocycles</th>
      <th>NumAliphaticHeterocycles</th>
      <th>NumAliphaticRings</th>
      <th>NumAromaticCarbocycles</th>
      <th>NumAromaticHeterocycles</th>
      <th>NumAromaticRings</th>
      <th>RingCount</th>
      <th>FractionCSP3</th>
      <th>TPSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>195.174</td>
      <td>1.64820</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>72.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>195.174</td>
      <td>1.64820</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>72.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>195.174</td>
      <td>1.64820</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>72.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>195.174</td>
      <td>1.64820</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>72.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>195.174</td>
      <td>1.64820</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.111111</td>
      <td>72.60</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>645</th>
      <td>333.100</td>
      <td>4.34550</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.083333</td>
      <td>54.46</td>
    </tr>
    <tr>
      <th>646</th>
      <td>320.218</td>
      <td>3.55597</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.583333</td>
      <td>28.78</td>
    </tr>
    <tr>
      <th>647</th>
      <td>320.218</td>
      <td>3.55597</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.583333</td>
      <td>28.78</td>
    </tr>
    <tr>
      <th>648</th>
      <td>320.218</td>
      <td>3.55597</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.583333</td>
      <td>28.78</td>
    </tr>
    <tr>
      <th>649</th>
      <td>320.218</td>
      <td>3.55597</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.583333</td>
      <td>28.78</td>
    </tr>
  </tbody>
</table>
<p>650 rows × 16 columns</p>
</div>




```python
e_coli_wild_efflux_feats = calc_feats(comp_a_evader_overlap)

e_coli_wild_efflux_feats = e_coli_wild_efflux_feats.dropna()


```


    ---------------------------------------------------------------------------

    ArgumentError                             Traceback (most recent call last)

    <ipython-input-19-a8adcfbc2436> in <module>
    ----> 1 e_coli_wild_efflux_feats = calc_feats(comp_a_evader_overlap)
          2 
          3 e_coli_wild_efflux_feats = e_coli_wild_efflux_feats.dropna()
          4 


    <ipython-input-12-c652eab93235> in calc_feats(df)
          8 #         table.loc[i,'SMILES']=Chem.MolToSmiles(mol)
          9 #         table.loc[i,'Mol']=mol
    ---> 10         table.loc[i,'MolWt']=Descriptors.MolWt(mol)
         11         table.loc[i,'LogP']=Descriptors.MolLogP(mol)
         12         table.loc[i,'NumHAcceptors']=Descriptors.NumHAcceptors(mol)


    ~/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/rdkit/Chem/Descriptors.py in <lambda>(*x, **y)
         65 
         66 
    ---> 67 MolWt = lambda *x, **y: _rdMolDescriptors._CalcMolWt(*x, **y)
         68 MolWt.version = _rdMolDescriptors._CalcMolWt_version
         69 MolWt.__doc__ = """The average molecular weight of the molecule


    ArgumentError: Python argument types in
        rdkit.Chem.rdMolDescriptors._CalcMolWt(str)
    did not match C++ signature:
        _CalcMolWt(RDKit::ROMol mol, bool onlyHeavy=False)



```python
master_functions.clean_mmpa_pairs_len(evader_transforms)
```

    Initial number of transofrms: 2385 
    Number fo transforms disqualified based on length discrepancy: 1948 
    Remaining number of transforms: 437





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B</th>
      <th>idsmiles_A</th>
      <th>idsmiles_B</th>
      <th>smirks</th>
      <th>common_core</th>
      <th>measurement_A</th>
      <th>measurement_B</th>
      <th>measurement_delta</th>
      <th>LHS</th>
      <th>RHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5578</th>
      <td>O/C(=C\C=N\c1ccc(Br)cc1)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>752</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>15.84</td>
      <td>30.93</td>
      <td>15.09</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5581</th>
      <td>Cc1ccc(/N=C/C=C(\O)c2ccc(Br)cc2)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>753</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccc(C)cc1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>-1.73</td>
      <td>30.93</td>
      <td>32.66</td>
      <td>[*:1]/C=N\c1ccc(C)cc1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5583</th>
      <td>O/C(=C\C=N\c1cccc(Cl)c1)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>755</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>-21.10</td>
      <td>30.93</td>
      <td>52.03</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>13581</th>
      <td>COc1ccc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc1</td>
      <td>Cc1ccc(N2C(=O)NC(=O)/C(=C\c3ccc([N+](=O)[O-])o...</td>
      <td>31110</td>
      <td>31119</td>
      <td>[*:1]c1ccc(OC)cc1&gt;&gt;[*:1]c1ccc([N+](=O)[O-])o1</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
      <td>-2.80</td>
      <td>-6.37</td>
      <td>-3.57</td>
      <td>[*:1]c1ccc(OC)cc1</td>
      <td>[*:1]c1ccc([N+](=O)[O-])o1</td>
    </tr>
    <tr>
      <th>13602</th>
      <td>COc1cc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc(...</td>
      <td>Cc1ccc(N2C(=O)NC(=O)/C(=C\c3ccc([N+](=O)[O-])o...</td>
      <td>31112</td>
      <td>31119</td>
      <td>[*:1]c1cc(I)c(O)c(OC)c1&gt;&gt;[*:1]c1ccc([N+](=O)[O...</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
      <td>8.95</td>
      <td>-6.37</td>
      <td>-15.32</td>
      <td>[*:1]c1cc(I)c(O)c(OC)c1</td>
      <td>[*:1]c1ccc([N+](=O)[O-])o1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1404506</th>
      <td>Br.CCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>27987</td>
      <td>28236</td>
      <td>[*:1]CC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-6.60</td>
      <td>-5.56</td>
      <td>1.04</td>
      <td>[*:1]CC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404512</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28118</td>
      <td>28236</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>1.42</td>
      <td>-5.56</td>
      <td>-6.98</td>
      <td>[*:1]CCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404517</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404521</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1405306</th>
      <td>O=C([O-])Cn1cnnn1.[K+]</td>
      <td>O=[N+]([O-])c1nonc1NCn1cnnn1</td>
      <td>73296</td>
      <td>64759</td>
      <td>[*:1]CC(=O)[O-]&gt;&gt;[*:1]CNc1nonc1[N+](=O)[O-]</td>
      <td>[*:1]n1cnnn1</td>
      <td>-13.82</td>
      <td>-2.55</td>
      <td>11.27</td>
      <td>[*:1]CC(=O)[O-]</td>
      <td>[*:1]CNc1nonc1[N+](=O)[O-]</td>
    </tr>
  </tbody>
</table>
<p>437 rows × 11 columns</p>
</div>




```python
master_functions.clean_mmpa_pairs_len(substrate_transforms)
```

    Initial number of transofrms: 14668 
    Number fo transforms disqualified based on length discrepancy: 4150 
    Remaining number of transforms: 10518





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B</th>
      <th>idsmiles_A</th>
      <th>idsmiles_B</th>
      <th>smirks</th>
      <th>common_core</th>
      <th>measurement_A</th>
      <th>measurement_B</th>
      <th>measurement_delta</th>
      <th>LHS</th>
      <th>RHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>329</th>
      <td>C/C(=N\NC(=O)c1nnn(-c2nonc2N)c1-c1ccccc1)c1ccc...</td>
      <td>Nc1nonc1-n1nnc(C(=O)N/N=C/c2ccccc2O)c1-c1ccccc1</td>
      <td>69387</td>
      <td>69423</td>
      <td>[*:1]C&gt;&gt;[*:1][H]</td>
      <td>[*:1]/C(=N\NC(=O)c1nnn(-c2nonc2N)c1-c1ccccc1)c...</td>
      <td>8.85</td>
      <td>55.86</td>
      <td>47.01</td>
      <td>[*:1]C</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>2191</th>
      <td>C/C(=N\Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1)c1cc...</td>
      <td>C/C(=N\Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1)c1cc...</td>
      <td>54132</td>
      <td>54133</td>
      <td>[*:1]c1ccc(Cl)cc1&gt;&gt;[*:1]c1ccc(N)cc1</td>
      <td>[*:1]/C(C)=N/Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1</td>
      <td>-0.44</td>
      <td>63.09</td>
      <td>63.53</td>
      <td>[*:1]c1ccc(Cl)cc1</td>
      <td>[*:1]c1ccc(N)cc1</td>
    </tr>
    <tr>
      <th>2258</th>
      <td>C/C(=N/Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1)c1cc...</td>
      <td>C/C(=N/Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1)c1cc...</td>
      <td>54138</td>
      <td>54140</td>
      <td>[*:1]c1ccc(Br)cc1&gt;&gt;[*:1]c1ccccc1</td>
      <td>[*:1]/C(C)=N\Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1</td>
      <td>14.87</td>
      <td>66.49</td>
      <td>51.62</td>
      <td>[*:1]c1ccc(Br)cc1</td>
      <td>[*:1]c1ccccc1</td>
    </tr>
    <tr>
      <th>2259</th>
      <td>C/C(=N/Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1)c1cc...</td>
      <td>C/C(=N/Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1)c1cc...</td>
      <td>54139</td>
      <td>54140</td>
      <td>[*:1]c1ccc(F)cc1&gt;&gt;[*:1]c1ccccc1</td>
      <td>[*:1]/C(C)=N\Nc1nc(Nc2cccc(Br)c2)nc(N2CCOCC2)n1</td>
      <td>47.29</td>
      <td>66.49</td>
      <td>19.20</td>
      <td>[*:1]c1ccc(F)cc1</td>
      <td>[*:1]c1ccccc1</td>
    </tr>
    <tr>
      <th>2723</th>
      <td>Cc1ccc2[nH]c(/C(C#N)=C/c3c(Cl)cccc3Cl)nc2c1</td>
      <td>Cc1ccc2[nH]c(/C(C#N)=C/c3cc(Br)c(O)c(Br)c3)nc2c1</td>
      <td>58348</td>
      <td>58331</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(Br)c(O)c(Br)c1</td>
      <td>[*:1]/C=C(\C#N)c1nc2cc(C)ccc2[nH]1</td>
      <td>3.82</td>
      <td>62.41</td>
      <td>58.59</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc(Br)c(O)c(Br)c1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1405589</th>
      <td>Cc1ccccc1NCn1nc(-c2ccc(Cl)cc2Cl)oc1=S</td>
      <td>OCn1nc(-c2ccc(Cl)cc2Cl)oc1=S</td>
      <td>28917</td>
      <td>13687</td>
      <td>[*:1]CNc1ccccc1C&gt;&gt;[*:1]CO</td>
      <td>[*:1]n1nc(-c2ccc(Cl)cc2Cl)oc1=S</td>
      <td>46.66</td>
      <td>61.85</td>
      <td>15.19</td>
      <td>[*:1]CNc1ccccc1C</td>
      <td>[*:1]CO</td>
    </tr>
    <tr>
      <th>1406560</th>
      <td>O=[N+]([O-])c1ccc(-n2ncc3c([N+](=O)[O-])cc([N+...</td>
      <td>O=[N+]([O-])c1cc([N+](=O)[O-])c2cn[nH]c2c1</td>
      <td>63724</td>
      <td>62159</td>
      <td>[*:1]c1ccc([N+](=O)[O-])cc1&gt;&gt;[*:1][H]</td>
      <td>[*:1]n1ncc2c([N+](=O)[O-])cc([N+](=O)[O-])cc21</td>
      <td>-19.26</td>
      <td>80.56</td>
      <td>99.82</td>
      <td>[*:1]c1ccc([N+](=O)[O-])cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1406563</th>
      <td>Cc1ccc(-n2ncc3c([N+](=O)[O-])cc([N+](=O)[O-])c...</td>
      <td>O=[N+]([O-])c1cc([N+](=O)[O-])c2cn[nH]c2c1</td>
      <td>63726</td>
      <td>62159</td>
      <td>[*:1]c1ccc(C)cc1&gt;&gt;[*:1][H]</td>
      <td>[*:1]n1ncc2c([N+](=O)[O-])cc([N+](=O)[O-])cc21</td>
      <td>-14.22</td>
      <td>80.56</td>
      <td>94.78</td>
      <td>[*:1]c1ccc(C)cc1</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1406565</th>
      <td>O=C(O)c1ccccc1-n1ncc2c([N+](=O)[O-])cc([N+](=O...</td>
      <td>O=[N+]([O-])c1cc([N+](=O)[O-])c2cn[nH]c2c1</td>
      <td>70401</td>
      <td>62159</td>
      <td>[*:1]c1ccccc1C(=O)O&gt;&gt;[*:1][H]</td>
      <td>[*:1]n1ncc2c([N+](=O)[O-])cc([N+](=O)[O-])cc21</td>
      <td>-8.49</td>
      <td>80.56</td>
      <td>89.05</td>
      <td>[*:1]c1ccccc1C(=O)O</td>
      <td>[*:1][H]</td>
    </tr>
    <tr>
      <th>1406566</th>
      <td>Cc1ccccc1-n1ncc2c([N+](=O)[O-])cc([N+](=O)[O-]...</td>
      <td>O=[N+]([O-])c1cc([N+](=O)[O-])c2cn[nH]c2c1</td>
      <td>70407</td>
      <td>62159</td>
      <td>[*:1]c1ccccc1C&gt;&gt;[*:1][H]</td>
      <td>[*:1]n1ncc2c([N+](=O)[O-])cc([N+](=O)[O-])cc21</td>
      <td>12.90</td>
      <td>80.56</td>
      <td>67.66</td>
      <td>[*:1]c1ccccc1C</td>
      <td>[*:1][H]</td>
    </tr>
  </tbody>
</table>
<p>10518 rows × 11 columns</p>
</div>




```python
a, b, c = master_functions.calculate_fractions_mk4(comp_a_evader_overlap)

```

      0%|          | 8/1609 [00:00<00:20, 79.63it/s]

    Generating molecular objects from pre-defined substructures
    Calcualting LHS+RHS matches


    100%|██████████| 1609/1609 [00:09<00:00, 173.84it/s]



```python
features_all_evade, l_feats_evade, r_feats_evade = master_functions.calculate_fractions_mk4(evader_transforms)

```

      0%|          | 1/2385 [00:00<06:05,  6.53it/s]

    Generating molecular objects from pre-defined substructures
    Calcualting LHS+RHS matches


    100%|██████████| 2385/2385 [00:12<00:00, 183.98it/s]



```python
features_all_neg, l_feats_neg, r_feats_neg = master_functions.calculate_fractions_mk4(mmpa_zero_neg)

features_all_pos, l_feats_pos, r_feats_pos = master_functions.calculate_fractions_mk4(mmpa_zero_pos)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B</th>
      <th>idsmiles_A</th>
      <th>idsmiles_B</th>
      <th>smirks</th>
      <th>common_core</th>
      <th>measurement_A</th>
      <th>measurement_B</th>
      <th>measurement_delta</th>
      <th>LHS</th>
      <th>RHS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5578</th>
      <td>O/C(=C\C=N\c1ccc(Br)cc1)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>752</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>15.84</td>
      <td>30.93</td>
      <td>15.09</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5581</th>
      <td>Cc1ccc(/N=C/C=C(\O)c2ccc(Br)cc2)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>753</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccc(C)cc1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>-1.73</td>
      <td>30.93</td>
      <td>32.66</td>
      <td>[*:1]/C=N\c1ccc(C)cc1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5583</th>
      <td>O/C(=C\C=N\c1cccc(Cl)c1)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>755</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>-21.10</td>
      <td>30.93</td>
      <td>52.03</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5584</th>
      <td>COc1ccccc1/N=C/C=C(\O)c1ccc(Br)cc1</td>
      <td>O=C(/C=C(\O)c1ccc(Br)cc1)C(F)(F)F</td>
      <td>756</td>
      <td>1153</td>
      <td>[*:1]/C=N\c1ccccc1OC&gt;&gt;[*:1]C(=O)C(F)(F)F</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
      <td>6.04</td>
      <td>30.93</td>
      <td>24.89</td>
      <td>[*:1]/C=N\c1ccccc1OC</td>
      <td>[*:1]C(=O)C(F)(F)F</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>CN(/C=C/[N+](=O)[O-])c1ccccc1</td>
      <td>O=[N+]([O-])/C=C/c1ccc(Cl)cc1Cl</td>
      <td>43517</td>
      <td>43527</td>
      <td>[*:1]N(C)c1ccccc1&gt;&gt;[*:1]c1ccc(Cl)cc1Cl</td>
      <td>[*:1]/C=C/[N+](=O)[O-]</td>
      <td>0.93</td>
      <td>-0.05</td>
      <td>-0.98</td>
      <td>[*:1]N(C)c1ccccc1</td>
      <td>[*:1]c1ccc(Cl)cc1Cl</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1404506</th>
      <td>Br.CCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>27987</td>
      <td>28236</td>
      <td>[*:1]CC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-6.60</td>
      <td>-5.56</td>
      <td>1.04</td>
      <td>[*:1]CC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404512</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28118</td>
      <td>28236</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>1.42</td>
      <td>-5.56</td>
      <td>-6.98</td>
      <td>[*:1]CCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404517</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1404521</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>1405306</th>
      <td>O=C([O-])Cn1cnnn1.[K+]</td>
      <td>O=[N+]([O-])c1nonc1NCn1cnnn1</td>
      <td>73296</td>
      <td>64759</td>
      <td>[*:1]CC(=O)[O-]&gt;&gt;[*:1]CNc1nonc1[N+](=O)[O-]</td>
      <td>[*:1]n1cnnn1</td>
      <td>-13.82</td>
      <td>-2.55</td>
      <td>11.27</td>
      <td>[*:1]CC(=O)[O-]</td>
      <td>[*:1]CNc1nonc1[N+](=O)[O-]</td>
    </tr>
  </tbody>
</table>
<p>2385 rows × 11 columns</p>
</div>




```python
sub_and_evade_features = calc_feats(sub_and_evade['Mol'])
sub_and_evade_features['label'] = sub_and_evade['label']
```

# DEV


```python
evader_transforms.compound_structure_A.iloc[0] # compound from evader transform that should be INACTIVE in wild
```




    'O/C(=C\\C=N\\c1ccc(Br)cc1)c1ccc(Br)cc1'




```python
e_coli_wild_efflux['SMILES'].apply(Chem.CanonSmiles)
```




    0                                     C1B2CC3CC1CC(C2)C3.N
    1                   C1B2CC3CC1CC(C2)C3.c1ccc(Cc2ccncc2)cc1
    2                        Brc1cncc(Br)c1.C1B2CC3CC1CC(C2)C3
    3                   C1B2CC3CC1CC(C2)C3.c1ccc(Cc2cccnc2)cc1
    4                         C1B2CC3CC1CC(C2)C3.CN(C)c1ccncc1
                                   ...                        
    73732                               Cl.c1csc(CNCc2cccs2)c1
    73733                                Clc1ccc(NCc2cccs2)cc1
    73734                            O=[As](O)(c1cccs1)c1cccs1
    73735    COc1cc([C@@H]2c3cc4c(cc3[C@@H](OC3OC5CO[C@@H](...
    73736                               O=C(O)C=CC(=O)Nc1nccs1
    Name: SMILES, Length: 73737, dtype: object




```python
e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == evader_transforms.compound_structure_A.iloc[0] ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>INHIB_AVE_wild</th>
      <th>INHIB_AVE_efflux</th>
      <th>Mol</th>
      <th>fps</th>
      <th>abs_diff</th>
      <th>sub_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>751</th>
      <td>O/C(=C\C=N\c1ccc(Br)cc1)c1ccc(Br)cc1</td>
      <td>-16.91</td>
      <td>-1.07</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
      <td>15.84</td>
      <td>increase</td>
    </tr>
  </tbody>
</table>
</div>




```python
e_coli_wild_efflux[e_coli_wild_efflux['SMILES'].isin(evader_transforms.compound_structure_B)].INHIB_AVE_wild.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b3cd3955890>




    
![png](co_add_chaser_files/co_add_chaser_178_1.png)
    



```python
matches_for_evaders = []
for smile in range(len(efflux_evader)):
    matches_for_evaders.append(len(ecoli_wild_index[ecoli_wild_index['compound_structure_B'] == efflux_evader.SMILES.iloc[smile]]))
```


```python
comp_a_evader_overlap.compound_structure_A.iloc[0]
```




    'Cc1ccc2nc(-c3ccc(/N=C/c4cc(Br)cc(Cl)c4O)cc3)sc2c1'




```python
comp  = comp_a_evader_overlap.compound_structure_A.iloc[0]

Chem.MolFromSmiles(comp_a_evader_overlap[comp_a_evader_overlap['compound_structure_A'] == comp].iloc[0].compound_structure_A)
```




    
![png](co_add_chaser_files/co_add_chaser_181_0.png)
    




```python
# evader: 

Chem.MolFromSmiles(comp_a_evader_overlap[comp_a_evader_overlap['compound_structure_A'] == comp].iloc[0].compound_structure_B)
```




    
![png](co_add_chaser_files/co_add_chaser_182_0.png)
    




```python
Chem.MolFromSmiles(comp_a_substrate_overlap[comp_a_substrate_overlap['compound_structure_A'] == comp].iloc[0].compound_structure_B)
```




    
![png](co_add_chaser_files/co_add_chaser_183_0.png)
    




```python
comp_a_substrate_overlap
```




    array(['O=C(CNc1ccc(Cl)cc1)N/N=C/c1ccc(Cl)cc1',
           'O=C(CSCc1ccc(Cl)cc1)N/N=C/c1ccc(Cl)cc1',
           'O=[N+]([O-])c1ccc(/N=C/c2cc(Cl)cc(Cl)c2O)cc1',
           'O=[N+]([O-])c1cc(Cl)cc(/C=N/c2ccc(F)cc2)c1O',
           'CCOc1ccc(/N=C/c2ccc([N+](=O)[O-])s2)cc1',
           'Cc1ccc(/N=C/c2ccc([N+](=O)[O-])s2)cc1I',
           'CC(=O)c1cccc(/N=C/c2ccc([N+](=O)[O-])s2)c1',
           'COc1cc(Br)cc(Br)c1OCC(=O)N/N=C/c1ccc([N+](=O)[O-])o1',
           'CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)[O-].[K+]',
           'O=C(CCl)NCCc1ccccc1', 'CC(Oc1ccccc1)C(=O)Nc1nc(-c2ccccn2)cs1',
           'COc1cc(C(=O)Nc2nccs2)cc(OC)c1OC',
           'C/C(=N/Nc1ccccc1[N+](=O)[O-])c1ccccc1',
           'CN(C)c1c([N+](=O)[O-])ncn1C', 'O=C(CSc1ccccn1)N/N=C/C=C/c1ccccc1',
           'CCn1cc(C(=O)O)c(=O)c2cnc(N3CCN(C(=S)Nc4cc(OC)ccc4OC)CC3)nc21',
           'Cc1cccc(O)c1/N=C/c1ccc([N+](=O)[O-])cc1',
           'O=[N+]([O-])c1ccc(/C=N/c2cc(Cl)ccc2O)cc1',
           'O=C(N/N=C/c1ccc([N+](=O)[O-])o1)c1cccc([N+](=O)[O-])c1',
           'Cc1n[n+]2c(C(=O)N3CCN(C)CC3)c(C)[n-]n2c1[N+](=O)[O-]',
           'CN(C)C(=O)c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
           'O=C(c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-])N1CCOCC1',
           'O=[N+]([O-])c1cc([N+](=O)[O-])c(/C=N/O)c([N+](=O)[O-])c1',
           'CCN(CC)c1c([N+](=O)[O-])ncn1C', 'Cn1cnc([N+](=O)[O-])c1NCC(=O)O',
           'Cn1cnc([N+](=O)[O-])c1Nc1ccc(O)cc1',
           'Cn1cnc([N+](=O)[O-])c1Nc1cccc(O)c1',
           'Cn1cnc([N+](=O)[O-])c1NC(=N)N',
           'CC(C)Nc1c([N+](=O)[O-])nn(C)[n+]1[O-]',
           'O=C(NC1CC1)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'O=C(NCC1CC1)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'CC(C)=NNC(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'CCCC/C=N/NC(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'O=C(Nc1ccncc1)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'Cc1cccc(NC(=O)c2cc([N+](=O)[O-])cc([N+](=O)[O-])c2)c1',
           'Cc1cccc(NC(=O)c2cc([N+](=O)[O-])cc([N+](=O)[O-])c2)n1',
           'O=C(Nc1nccs1)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'O=C(OCC1CCCCC1)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'O=C(OCC1CCCO1)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'NNC(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1.O=C(O)C(F)(F)F.O=C(O)C(F)(F)F',
           'C#CCNC(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
           'Cc1ccc2nc(-c3ccc(/N=C/c4cc(Br)cc(Cl)c4O)cc3)sc2c1',
           'N/C(=N\\O)c1ccc(Br)cc1', '[O-][n+]1onc2cc(Br)ccc21',
           'Br.C#CCNCc1ccccc1', 'O/N=C(\\CBr)c1ccccc1', 'N#CNC(=N)c1ccccc1',
           'NNCCc1ccccc1.O=S(=O)(O)O', 'N#CCC(=O)c1ccccc1',
           'N#CC(c1ccccc1)C(C/C(=N\\O)c1ccccc1)c1ccccc1',
           'O=C(/C=C(\\O)c1ccccc1)C(c1ccccc1)c1ccccc1',
           'CCC/C(=N\\NC(=O)C(O)(c1ccccc1)c1ccccc1)c1ccccc1',
           'C/C(=N/NC(=O)C(O)(c1ccccc1)c1ccccc1)c1ccccc1', 'Cn1ccc(=N)cc1.I',
           'CCOC(=O)Cn1ccc(=N)cc1.Cl', 'Br.CCn1ccc(=N)cc1',
           'CCCn1ccc(=N)cc1.I', 'CCCCn1ccc(=N)cc1.I', 'Br.CCCCCCn1ccc(=N)cc1'],
          dtype=object)



# Compound_A and LHS overlap in evader and substrate


```python
l = [['DEV1', 'READ1'], ['DEV1', 'READ2'], ['DEV2','READ1']]
l = [tuple(i) for i in l]
df[df[['DEVICE', 'READING']].apply(tuple, axis = 1).isin(l)]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-1a202ebebc1b> in <module>
          1 l = [['DEV1', 'READ1'], ['DEV1', 'READ2'], ['DEV2','READ1']]
          2 l = [tuple(i) for i in l]
    ----> 3 df[df[['DEVICE', 'READING']].apply(tuple, axis = 1).isin(l)]
    

    NameError: name 'df' is not defined



```python
evader_transforms[['compound_structure_A', 'LHS', 'common_core']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>LHS</th>
      <th>common_core</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5578</th>
      <td>O/C(=C\C=N\c1ccc(Br)cc1)c1ccc(Br)cc1</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
    </tr>
    <tr>
      <th>5581</th>
      <td>Cc1ccc(/N=C/C=C(\O)c2ccc(Br)cc2)cc1</td>
      <td>[*:1]/C=N\c1ccc(C)cc1</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
    </tr>
    <tr>
      <th>5583</th>
      <td>O/C(=C\C=N\c1cccc(Cl)c1)c1ccc(Br)cc1</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
    </tr>
    <tr>
      <th>13581</th>
      <td>COc1ccc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc1</td>
      <td>[*:1]c1ccc(OC)cc1</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
    </tr>
    <tr>
      <th>13602</th>
      <td>COc1cc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc(...</td>
      <td>[*:1]c1cc(I)c(O)c(OC)c1</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1404516</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1404517</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1404520</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1404521</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1405306</th>
      <td>O=C([O-])Cn1cnnn1.[K+]</td>
      <td>[*:1]CC(=O)[O-]</td>
      <td>[*:1]n1cnnn1</td>
    </tr>
  </tbody>
</table>
<p>612 rows × 3 columns</p>
</div>




```python
evader_transforms[['compound_structure_A', 'LHS', 'common_core']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>LHS</th>
      <th>common_core</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5578</th>
      <td>O/C(=C\C=N\c1ccc(Br)cc1)c1ccc(Br)cc1</td>
      <td>[*:1]/C=N\c1ccc(Br)cc1</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
    </tr>
    <tr>
      <th>5581</th>
      <td>Cc1ccc(/N=C/C=C(\O)c2ccc(Br)cc2)cc1</td>
      <td>[*:1]/C=N\c1ccc(C)cc1</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
    </tr>
    <tr>
      <th>5583</th>
      <td>O/C(=C\C=N\c1cccc(Cl)c1)c1ccc(Br)cc1</td>
      <td>[*:1]/C=N\c1cccc(Cl)c1</td>
      <td>[*:1]/C=C(\O)c1ccc(Br)cc1</td>
    </tr>
    <tr>
      <th>13581</th>
      <td>COc1ccc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc1</td>
      <td>[*:1]c1ccc(OC)cc1</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
    </tr>
    <tr>
      <th>13602</th>
      <td>COc1cc(/C=C2\C(=O)NC(=O)N(c3ccc(C)cc3)C2=O)cc(...</td>
      <td>[*:1]c1cc(I)c(O)c(OC)c1</td>
      <td>[*:1]/C=C1\C(=O)NC(=O)N(c2ccc(C)cc2)C1=O</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1404505</th>
      <td>Br.CCn1ccc(=N)cc1</td>
      <td>[*:1]CC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1404511</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>[*:1]CCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1404516</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1404520</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
    </tr>
    <tr>
      <th>1405306</th>
      <td>O=C([O-])Cn1cnnn1.[K+]</td>
      <td>[*:1]CC(=O)[O-]</td>
      <td>[*:1]n1cnnn1</td>
    </tr>
  </tbody>
</table>
<p>527 rows × 3 columns</p>
</div>




```python
evader_tuples = evader_transforms[['compound_structure_A', 'LHS', 'common_core']].apply(tuple, axis = 1).drop_duplicates()

substrate_tuples = substrate_transforms[['compound_structure_A', 'LHS', 'common_core']].apply(tuple, axis = 1).drop_duplicates()
```


```python
sub_overlap = substrate_transforms[substrate_transforms[['compound_structure_A', 'LHS', 'common_core']].apply(tuple, axis = 1).isin(evader_tuples)]

evade_overlap = evader_transforms[evader_transforms[['compound_structure_A', 'LHS', 'common_core']].apply(tuple, axis = 1).isin(substrate_tuples)]
```


```python
substrate_transforms.merge(evader_transforms, on=['compound_structure_A', 'LHS', 'common_core'], suffixes=['_substrate','_evader'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B_substrate</th>
      <th>idsmiles_A_substrate</th>
      <th>idsmiles_B_substrate</th>
      <th>smirks_substrate</th>
      <th>common_core</th>
      <th>measurement_A_substrate</th>
      <th>measurement_B_substrate</th>
      <th>measurement_delta_substrate</th>
      <th>LHS</th>
      <th>RHS_substrate</th>
      <th>compound_structure_B_evader</th>
      <th>idsmiles_A_evader</th>
      <th>idsmiles_B_evader</th>
      <th>smirks_evader</th>
      <th>measurement_A_evader</th>
      <th>measurement_B_evader</th>
      <th>measurement_delta_evader</th>
      <th>RHS_evader</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>O=C(N/N=C/c1cc([N+](=O)[O-])ccc1O)C(F)(F)C(F)(...</td>
      <td>12181</td>
      <td>12182</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>63.12</td>
      <td>43.03</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>O=C(N/N=C/c1ccc(O)cc1)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>12181</td>
      <td>12189</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1ccc(O)cc1</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>62.07</td>
      <td>41.98</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1ccc(O)cc1</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>O=C(N/N=C/c1ccc(O)cc1O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12201</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1ccc(O)cc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>49.35</td>
      <td>29.26</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1ccc(O)cc1O</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>O=C(N/N=C/c1cc([N+](=O)[O-])ccc1O)C(F)(F)C(F)(...</td>
      <td>12198</td>
      <td>12182</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>63.12</td>
      <td>64.24</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>O=C(N/N=C/c1ccc(O)cc1)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>12198</td>
      <td>12189</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1ccc(O)cc1</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>62.07</td>
      <td>63.19</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1ccc(O)cc1</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28118</td>
      <td>28233</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>1.42</td>
      <td>56.99</td>
      <td>55.57</td>
      <td>[*:1]CCC</td>
      <td>[*:1]CCCCCCC</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28118</td>
      <td>28236</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>1.42</td>
      <td>-5.56</td>
      <td>-6.98</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>197</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28145</td>
      <td>28233</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>56.99</td>
      <td>71.51</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCC</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28235</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>-14.52</td>
      <td>32.75</td>
      <td>47.27</td>
      <td>[*:1]CCCCCCCC</td>
    </tr>
    <tr>
      <th>198</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28145</td>
      <td>28233</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>56.99</td>
      <td>71.51</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCC</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28228</td>
      <td>28233</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>56.99</td>
      <td>43.27</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCC</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28235</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>13.72</td>
      <td>32.75</td>
      <td>19.03</td>
      <td>[*:1]CCCCCCCC</td>
    </tr>
    <tr>
      <th>200</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28228</td>
      <td>28233</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>56.99</td>
      <td>43.27</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCC</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCCCCCC</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 19 columns</p>
</div>




```python
comp_a_lhs_overlap = evader_transforms.merge(substrate_transforms, on=['compound_structure_A', 'LHS', 'common_core'], suffixes=['_evader','_substrate'])
```


```python
comp_a_lhs_overlap.to_pickle('data_curated/comp_a_lhs_overlap.pkl')
```


```python
len(comp_a_lhs_overlap.compound_structure_A.unique())
```




    85




```python
len(comp_a_lhs_overlap)
```




    201




```python
len(comp_a_lhs_overlap.compound_structure_B_substrate.unique())
```




    39




```python
len(comp_a_lhs_overlap.compound_structure_B_evader.unique())
```




    21




```python
lhs_and_comp_a_evader = comp_a_evader_overlap[(comp_a_evader_overlap['LHS'].isin(comp_a_substrate_overlap['LHS'].values)) & (comp_a_evader_overlap['compound_structure_A'].isin(comp_a_substrate_overlap['compound_structure_A'].values))]
```


```python
lhs_and_comp_a_substrate = comp_a_substrate_overlap[comp_a_substrate_overlap['LHS'].isin(comp_a_evader_overlap['LHS'].values) & (comp_a_substrate_overlap['compound_structure_A'].isin(comp_a_evader_overlap['compound_structure_A'].values))]
```


```python
len(lhs_and_comp_a_evader), len(lhs_and_comp_a_substrate)
```




    (120, 211)




```python
len(lhs_and_comp_a_evader.LHS.unique())
```




    98



# Plot some overlapping compounds 2.0

181 transfroms in substrates that equate to 85 compound_A into 39 compound B

111 transfroms in evader that equate to 85 compound_A into 21 compound B


```python
len(sub_overlap.LHS.unique())
```




    93




```python
len(evade_overlap.LHS.unique())
```




    93




```python
len(sub_overlap), len(sub_overlap.compound_structure_A.unique()), len(sub_overlap.compound_structure_B.unique())
```




    (181, 85, 39)




```python
len(evade_overlap), len(evade_overlap.compound_structure_A.unique()), len(evade_overlap.compound_structure_B.unique())
```




    (111, 85, 21)




```python
evade_overlap.drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B</th>
      <th>idsmiles_A</th>
      <th>idsmiles_B</th>
      <th>smirks</th>
      <th>common_core</th>
      <th>measurement_A</th>
      <th>measurement_B</th>
      <th>measurement_delta</th>
      <th>LHS</th>
      <th>RHS</th>
      <th>core_mol</th>
      <th>core_fps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24121</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>24216</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>24304</th>
      <td>COc1cc(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12194</td>
      <td>12199</td>
      <td>[*:1]c1ccc([*:2])c(OC)c1&gt;&gt;[*:2]c1ccc(OC)cc1[*:1]</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>23.46</td>
      <td>45.24</td>
      <td>21.78</td>
      <td>[*:1]c1ccc([*:2])c(OC)c1</td>
      <td>[*:2]c1ccc(OC)cc1[*:1]</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>58739</th>
      <td>Cc1cccc(O)c1/N=C/c1cc(Br)cc(Br)c1O</td>
      <td>Cc1cccc(O)c1/N=C/c1cc(I)cc(I)c1O</td>
      <td>24071</td>
      <td>24075</td>
      <td>[*:1]c1cc(Br)cc(Br)c1O&gt;&gt;[*:1]c1cc(I)cc(I)c1O</td>
      <td>[*:1]/C=N/c1c(C)cccc1O</td>
      <td>11.55</td>
      <td>38.39</td>
      <td>26.84</td>
      <td>[*:1]c1cc(Br)cc(Br)c1O</td>
      <td>[*:1]c1cc(I)cc(I)c1O</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>58746</th>
      <td>Cc1cccc(O)c1/N=C/c1cc(Br)cc(Cl)c1O</td>
      <td>Cc1cccc(O)c1/N=C/c1cc(I)cc(I)c1O</td>
      <td>24072</td>
      <td>24075</td>
      <td>[*:1]c1cc(Br)cc(Cl)c1O&gt;&gt;[*:1]c1cc(I)cc(I)c1O</td>
      <td>[*:1]/C=N/c1c(C)cccc1O</td>
      <td>6.35</td>
      <td>38.39</td>
      <td>32.04</td>
      <td>[*:1]c1cc(Br)cc(Cl)c1O</td>
      <td>[*:1]c1cc(I)cc(I)c1O</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1404512</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28118</td>
      <td>28236</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>1.42</td>
      <td>-5.56</td>
      <td>-6.98</td>
      <td>[*:1]CCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>1404516</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28235</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>32.75</td>
      <td>47.27</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCC</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>1404517</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>1404520</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28235</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>32.75</td>
      <td>19.03</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCC</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
    <tr>
      <th>1404521</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>&lt;img data-content="rdkit/molecule" src="data:i...</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...</td>
    </tr>
  </tbody>
</table>
<p>111 rows × 13 columns</p>
</div>




```python
for i in range(0):
    comp_a = Chem.MolFromSmiles(evade_overlap.compound_structure_A.unique().iloc[nr])
    
    comp_b_evader = Chem.MolFromSmiles(evade_overlap.compound_structure_B.iloc[nr])

```


```python
plt.figure(figsize=(10,10))


venn2([ set(evader_transforms[['compound_structure_A', 'LHS']].apply(tuple, axis = 1)), 
        set(substrate_transforms[['compound_structure_A', 'LHS']].apply(tuple, axis = 1))],
        set_labels=('Evader Comp_A+ LHS', 'Substrate Comp_A+LHS')
     )
```




    <matplotlib_venn._common.VennDiagram at 0x2b5839a39bd0>




    
![png](co_add_chaser_files/co_add_chaser_211_1.png)
    



```python
mol = evade_overlap.core_mol.iloc[0]
```


```python
a = evade_overlap['core_fps'].iloc[0]
```

# Plot some overlapping compounds


```python
comp_a_lhs_overlap[comp_a_lhs_overlap.compound_structure_B_evader.isin(comp_a_lhs_overlap.compound_structure_B_evader.unique())]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B_evader</th>
      <th>idsmiles_A_evader</th>
      <th>idsmiles_B_evader</th>
      <th>smirks_evader</th>
      <th>common_core</th>
      <th>measurement_A_evader</th>
      <th>measurement_B_evader</th>
      <th>measurement_delta_evader</th>
      <th>LHS</th>
      <th>RHS_evader</th>
      <th>compound_structure_B_substrate</th>
      <th>idsmiles_A_substrate</th>
      <th>idsmiles_B_substrate</th>
      <th>smirks_substrate</th>
      <th>measurement_A_substrate</th>
      <th>measurement_B_substrate</th>
      <th>measurement_delta_substrate</th>
      <th>RHS_substrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1cc([N+](=O)[O-])ccc1O)C(F)(F)C(F)(...</td>
      <td>12181</td>
      <td>12182</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>20.09</td>
      <td>63.12</td>
      <td>43.03</td>
      <td>[*:1]c1cc([N+](=O)[O-])ccc1O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1ccc(O)cc1)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>12181</td>
      <td>12189</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1ccc(O)cc1</td>
      <td>20.09</td>
      <td>62.07</td>
      <td>41.98</td>
      <td>[*:1]c1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1ccc(O)cc1O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12201</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1ccc(O)cc1O</td>
      <td>20.09</td>
      <td>49.35</td>
      <td>29.26</td>
      <td>[*:1]c1ccc(O)cc1O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1cc([N+](=O)[O-])ccc1O)C(F)(F)C(F)(...</td>
      <td>12198</td>
      <td>12182</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>-1.12</td>
      <td>63.12</td>
      <td>64.24</td>
      <td>[*:1]c1cc([N+](=O)[O-])ccc1O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1ccc(O)cc1)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>12198</td>
      <td>12189</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1ccc(O)cc1</td>
      <td>-1.12</td>
      <td>62.07</td>
      <td>63.19</td>
      <td>[*:1]c1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28118</td>
      <td>28236</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>1.42</td>
      <td>-5.56</td>
      <td>-6.98</td>
      <td>[*:1]CCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28118</td>
      <td>28233</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>1.42</td>
      <td>56.99</td>
      <td>55.57</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>197</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28235</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>32.75</td>
      <td>47.27</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28145</td>
      <td>28233</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>-14.52</td>
      <td>56.99</td>
      <td>71.51</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>198</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28145</td>
      <td>28233</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>-14.52</td>
      <td>56.99</td>
      <td>71.51</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28235</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>32.75</td>
      <td>19.03</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28228</td>
      <td>28233</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>13.72</td>
      <td>56.99</td>
      <td>43.27</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>200</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28228</td>
      <td>28233</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>13.72</td>
      <td>56.99</td>
      <td>43.27</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 19 columns</p>
</div>




```python
len(comp_a_lhs_overlap.compound_structure_B_substrate.unique())
```




    39




```python
len(comp_a_lhs_overlap.compound_structure_A.unique())
```




    85




```python
comp_a_lhs_overlap
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound_structure_A</th>
      <th>compound_structure_B_evader</th>
      <th>idsmiles_A_evader</th>
      <th>idsmiles_B_evader</th>
      <th>smirks_evader</th>
      <th>common_core</th>
      <th>measurement_A_evader</th>
      <th>measurement_B_evader</th>
      <th>measurement_delta_evader</th>
      <th>LHS</th>
      <th>RHS_evader</th>
      <th>compound_structure_B_substrate</th>
      <th>idsmiles_A_substrate</th>
      <th>idsmiles_B_substrate</th>
      <th>smirks_substrate</th>
      <th>measurement_A_substrate</th>
      <th>measurement_B_substrate</th>
      <th>measurement_delta_substrate</th>
      <th>RHS_substrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1cc([N+](=O)[O-])ccc1O)C(F)(F)C(F)(...</td>
      <td>12181</td>
      <td>12182</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>20.09</td>
      <td>63.12</td>
      <td>43.03</td>
      <td>[*:1]c1cc([N+](=O)[O-])ccc1O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1ccc(O)cc1)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>12181</td>
      <td>12189</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1ccc(O)cc1</td>
      <td>20.09</td>
      <td>62.07</td>
      <td>41.98</td>
      <td>[*:1]c1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C/C=C/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12199</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>20.09</td>
      <td>45.24</td>
      <td>25.15</td>
      <td>[*:1]/C=C\C</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1ccc(O)cc1O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12181</td>
      <td>12201</td>
      <td>[*:1]/C=C\C&gt;&gt;[*:1]c1ccc(O)cc1O</td>
      <td>20.09</td>
      <td>49.35</td>
      <td>29.26</td>
      <td>[*:1]c1ccc(O)cc1O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1cc([N+](=O)[O-])ccc1O)C(F)(F)C(F)(...</td>
      <td>12198</td>
      <td>12182</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc([N+](=O)[O-])ccc1O</td>
      <td>-1.12</td>
      <td>63.12</td>
      <td>64.24</td>
      <td>[*:1]c1cc([N+](=O)[O-])ccc1O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O=C(N/N=C/c1c(Cl)cccc1Cl)C(F)(F)C(F)(F)C(F)(F)...</td>
      <td>COc1ccc(O)c(/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>12198</td>
      <td>12199</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1cc(OC)ccc1O</td>
      <td>[*:1]/C=N/NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(...</td>
      <td>-1.12</td>
      <td>45.24</td>
      <td>46.36</td>
      <td>[*:1]c1c(Cl)cccc1Cl</td>
      <td>[*:1]c1cc(OC)ccc1O</td>
      <td>O=C(N/N=C/c1ccc(O)cc1)C(F)(F)C(F)(F)C(F)(F)C(F...</td>
      <td>12198</td>
      <td>12189</td>
      <td>[*:1]c1c(Cl)cccc1Cl&gt;&gt;[*:1]c1ccc(O)cc1</td>
      <td>-1.12</td>
      <td>62.07</td>
      <td>63.19</td>
      <td>[*:1]c1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>CCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28118</td>
      <td>28236</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>1.42</td>
      <td>-5.56</td>
      <td>-6.98</td>
      <td>[*:1]CCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28118</td>
      <td>28233</td>
      <td>[*:1]CCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>1.42</td>
      <td>56.99</td>
      <td>55.57</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>197</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28235</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>32.75</td>
      <td>47.27</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28145</td>
      <td>28233</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>-14.52</td>
      <td>56.99</td>
      <td>71.51</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>198</th>
      <td>CCCCn1ccc(=N)cc1.I</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28145</td>
      <td>28236</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>-14.52</td>
      <td>-5.56</td>
      <td>8.96</td>
      <td>[*:1]CCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28145</td>
      <td>28233</td>
      <td>[*:1]CCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>-14.52</td>
      <td>56.99</td>
      <td>71.51</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28235</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>32.75</td>
      <td>19.03</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28228</td>
      <td>28233</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>13.72</td>
      <td>56.99</td>
      <td>43.27</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
    <tr>
      <th>200</th>
      <td>Br.CCCCCCn1ccc(=N)cc1</td>
      <td>Br.CCCCCCCCCCn1ccc(=N)cc1</td>
      <td>28228</td>
      <td>28236</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCCCCC</td>
      <td>[*:1]n1ccc(=N)cc1</td>
      <td>13.72</td>
      <td>-5.56</td>
      <td>-19.28</td>
      <td>[*:1]CCCCCC</td>
      <td>[*:1]CCCCCCCCCC</td>
      <td>CCCCCCCn1ccc(=N)cc1.I</td>
      <td>28228</td>
      <td>28233</td>
      <td>[*:1]CCCCCC&gt;&gt;[*:1]CCCCCCC</td>
      <td>13.72</td>
      <td>56.99</td>
      <td>43.27</td>
      <td>[*:1]CCCCCCC</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 19 columns</p>
</div>




```python
def draw_full_mols(nr, name, save):

    # compound_A
    comp_a = Chem.MolFromSmiles(comp_a_lhs_overlap.compound_structure_A.iloc[nr])

    core = Chem.MolFromSmiles(comp_a_lhs_overlap.compound_structure_A.iloc[nr])

    # compound_B_evader
    comp_b_evader = Chem.MolFromSmiles(comp_a_lhs_overlap.compound_structure_B_evader.iloc[nr])

    # compound_B_substrate
    comp_b_substrate = Chem.MolFromSmiles(comp_a_lhs_overlap.compound_structure_B_substrate.iloc[nr])

    # labels
    inactive_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_A.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    evader_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_evader.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    substrate_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_substrate.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    lab = ['Inactive no_{}; '+ '\n' + 'WT: {}; tolC: {}'.format(nr, inactive_label[0][0], inactive_label[0][1]), 'Substrate no_{}; WT: {}; tolC: {}'.format(nr, substrate_label[0][0], substrate_label[0][1]), 'Evader no_{}; WT: {}; tolC: {}'.format(nr, evader_label[0][0], evader_label[0][1]),]

    mols=[comp_a, comp_b_substrate, comp_b_evader]
    font = {'fontsize': 10}

    img = Chem.Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(400,400), legends=lab, useSVG=True), 
    
    if save==True:

        with open(name + '.svg', 'w') as f:
            f.write(img[0].data)

        return print('saved {}.svg'.format(name))
    else:
        
        return img[0]
```


```python
draw_full_mols(150, 'figure_mols/compound_80_full', save = False)

```




    
![svg](co_add_chaser_files/co_add_chaser_220_0.svg)
    




```python
def draw_partial_mols(nr, name, save):

    # compound_A
    core = Chem.MolFromSmiles(comp_a_lhs_overlap.common_core.iloc[nr])

    # LHS
    lhs = Chem.MolFromSmiles(comp_a_lhs_overlap.LHS.iloc[nr])

    # compound_B_evader
    RHS_evader = Chem.MolFromSmiles(comp_a_lhs_overlap.RHS_evader.iloc[nr])

    # compound_B_substrate
    RHS_substrate = Chem.MolFromSmiles(comp_a_lhs_overlap.RHS_substrate.iloc[nr])

    mols=[core, lhs , RHS_substrate, RHS_evader]
    
    # labels
    inactive_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_A.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    evader_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_evader.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    substrate_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_substrate.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    lab = ['Common core no_{}'.format(nr), 'Inactive; WT: {:.1f}%; tolC: {:.1f}%'.format(inactive_label[0][0], inactive_label[0][1]), 'Substrate; WT: {:.1f}%; tolC: {:.1f}%'.format(substrate_label[0][0], substrate_label[0][1]), 'Evader; WT: {:.1f}%; tolC: {:.1f}%'.format(evader_label[0][0], evader_label[0][1]),]

    img = Chem.Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250,250), legends=lab, useSVG=True)
    
    if save == True:
        with open(name + '.svg', 'w') as f:
            f.write(img.data)

        return print('saved {}.svg'.format(name))
    else:
        return Chem.Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250,250), legends=lab, useSVG=True)

```


```python
# nr = 80
draw_partial_mols(80, 'figure_mols/compound_80_partial', save=False)

```




    
![svg](co_add_chaser_files/co_add_chaser_222_0.svg)
    




```python
no=137

draw_partial_mols(no, 'figure_mols/compound_'+str(no)+'_partial', save=False)

```




    
![svg](co_add_chaser_files/co_add_chaser_223_0.svg)
    




```python
nr=137
core = comp_a_lhs_overlap.common_core.iloc[nr]
print('core:', core)
# LHS
lhs = comp_a_lhs_overlap.LHS.iloc[nr]
print('inactive:',lhs)
# compound_B_substrate
RHS_substrate = comp_a_lhs_overlap.RHS_substrate.iloc[nr]
print('RHS_substrate:',RHS_substrate)
# compound_B_evader
RHS_evader = comp_a_lhs_overlap.RHS_evader.iloc[nr]
print('RHS_evader:',RHS_evader)


```

    core: [*:1]c1cc(I)cc(I)c1O
    inactive: [*:1]/C=N\c1ccccc1
    RHS_substrate: [*:1]/C=N\c1ccccc1I
    RHS_evader: [*:1]/C=N\c1ncccc1O



```python
import sympy.printing as printing
delta__y_l = sp.symbols('Delta__y_l')
print(printing.latex(delta__y_l))
```

    \Delta^{y}_{l}



```python
nr=137


# labels

inactive_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_A.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

evader_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_evader.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

substrate_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_substrate.iloc[nr]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

lab = ['Inactive; WT: {:.1f}%; tolC: {:.1f}%'.format(inactive_label[0][0], inactive_label[0][1]), 'Substrate; WT: {:.1f}%; tolC: {:.1f}%'.format(substrate_label[0][0], substrate_label[0][1]), 'Evader; WT: {:.1f}%; tolC: {:.1f}%'.format(evader_label[0][0], evader_label[0][1]),]

print(lab)


```

    ['Inactive; WT: 7.0%; tolC: -2.5%', 'Substrate; WT: 28.6%; tolC: 91.0%', 'Evader; WT: 60.7%; tolC: 97.1%']



```python
mols=[]

labels=[]


for i in range(len(comp_a_lhs_overlap)):

    # compound_A
    core = Chem.MolFromSmiles(comp_a_lhs_overlap.common_core.iloc[i])
    # LHS
    lhs = Chem.MolFromSmiles(comp_a_lhs_overlap.LHS.iloc[i])
    # compound_B_evader
    RHS_evader = Chem.MolFromSmiles(comp_a_lhs_overlap.RHS_evader.iloc[i])
    # compound_B_substrate
    RHS_substrate = Chem.MolFromSmiles(comp_a_lhs_overlap.RHS_substrate.iloc[i])

#     mols=[core, lhs , RHS_substrate, RHS_evader]

    # labels
    inactive_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_A.iloc[i]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    evader_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_evader.iloc[i]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    substrate_label = e_coli_wild_efflux[e_coli_wild_efflux['SMILES'] == comp_a_lhs_overlap.compound_structure_B_substrate.iloc[i]][['INHIB_AVE_wild', 'INHIB_AVE_efflux']].values

    lab = ['Common core no_{}'.format(i), 'Inactive; WT: {:.1f}%; tolC: {:.1f}%'.format(inactive_label[0][0], inactive_label[0][1]), 'Substrate; WT: {:.1f}%; tolC: {:.1f}%'.format(substrate_label[0][0], substrate_label[0][1]), 'Evader; WT: {:.1f}%; tolC: {:.1f}%'.format(evader_label[0][0], evader_label[0][1]),]

# img = Chem.Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250,250), legends=lab, useSVG=True)
    mols.append(core)
    mols.append(lhs)
    mols.append(RHS_substrate)
    mols.append(RHS_evader)
    
    labels.append(lab[0])
    labels.append(lab[1])
    labels.append(lab[2])
    labels.append(lab[3])


img = Chem.Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200,200), legends=labels, useSVG=False, maxMols= 600)

# with open('master_transform' + '.svg', 'w') as f:
#     f.write(img.data)

# print('saved {}.svg'.format(name))
```


```python
img.save('master_transform.png')
```

# TSNE of overlap


```python
evader_overlap = efflux_evader[efflux_evader['SMILES'].isin(comp_a_lhs_overlap.compound_structure_B_evader)]
```


```python
inactive_overlap = inactive[inactive['SMILES'].isin(comp_a_lhs_overlap.compound_structure_A)]
```


```python
substrate_overlap = efflux_substrate[efflux_substrate['SMILES'].isin(comp_a_lhs_overlap.compound_structure_B_substrate)]
```


```python
one = evader_overlap.append(inactive_overlap)
```


```python
two = one.append(substrate_overlap).reset_index(drop=True)
```


```python
two_tsne =  master_functions.tsne_no_plot(two['fps'], 20)
```

    [t-SNE] Computing 61 nearest neighbors...
    [t-SNE] Indexed 145 samples in 0.001s...
    [t-SNE] Computed neighbors for 145 samples in 0.008s...
    [t-SNE] Computed conditional probabilities for sample 145 / 145
    [t-SNE] Mean sigma: 0.928431
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 55.186184
    [t-SNE] KL divergence after 800 iterations: 0.193251



```python
two.Class.value_counts()
```




    Inactive            85
    Efflux Substrate    39
    Efflux Evader       21
    Name: Class, dtype: int64




```python
fig, ax = plt.subplots(figsize=(12,12))

sns.scatterplot(x='TC1',y='TC2',data=two_tsne, s=20 ,alpha=0.7, hue=two['Class']) 

fig, ax = plt.subplots(figsize=(12,12))

sns.kdeplot(x='TC1',y='TC2',data=two_tsne,alpha=0.7, hue=two['Class'], levels = 3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2ad8a772e3d0>




    
![png](co_add_chaser_files/co_add_chaser_237_1.png)
    



    
![png](co_add_chaser_files/co_add_chaser_237_2.png)
    



```python

```
