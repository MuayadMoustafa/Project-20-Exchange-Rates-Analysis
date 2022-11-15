```python
import numpy as np                # linear algebra
import pandas as pd               # data frames
import seaborn as sns             # visualizations
import matplotlib.pyplot as plt   # visualizations
import scipy.stats                # statistics
from sklearn import preprocessing


```


```python
df = pd.read_csv("D:Courses/Projects/exchange_rate_to_usd.csv",index_col=0)

# Print the head of df
print(df.head())

# Print the info of df
print(df.info())

# Print the shape of df
print(df.shape)
```

                chinese_yuan_to_usd  euro_to_usd  japanese_yen_to_usd  \
    date                                                                
    2004-01-02                  NaN       1.2592                  NaN   
    2004-01-05             0.120817       1.2657             0.009355   
    2004-01-06             0.120815       1.2756             0.009412   
    2004-01-07             0.120814       1.2679             0.009413   
    2004-01-08             0.120814       1.2634             0.009421   
    
                uk_pound_to_usd  us_dollar_to_usd  algerian_dinar_to_usd  \
    date                                                                   
    2004-01-02           1.7858               1.0                    NaN   
    2004-01-05           1.7996               1.0                    NaN   
    2004-01-06           1.8209               1.0                    NaN   
    2004-01-07           1.8140               1.0                    NaN   
    2004-01-08           1.8122               1.0                    NaN   
    
                australian_dollar_to_usd  bahrain_dinar_to_usd  \
    date                                                         
    2004-01-02                    0.7527              2.659574   
    2004-01-05                    0.7630              2.659574   
    2004-01-06                    0.7668              2.659574   
    2004-01-07                    0.7677              2.659574   
    2004-01-08                    0.7679              2.659574   
    
                botswana_pula_to_usd  brazilian_real_to_usd  ...  \
    date                                                     ...   
    2004-01-02                   NaN               0.346212  ...   
    2004-01-05               0.23115               0.346572  ...   
    2004-01-06               0.22965               0.349418  ...   
    2004-01-07               0.22630               0.350877  ...   
    2004-01-08               0.22400               0.348250  ...   
    
                south_african_rand_to_usd  sri_lankan_rupee_to_usd  \
    date                                                             
    2004-01-02                   0.149813                 0.010313   
    2004-01-05                   0.157233                 0.010277   
    2004-01-06                   0.154440                 0.010245   
    2004-01-07                        NaN                      NaN   
    2004-01-08                   0.150602                 0.010222   
    
                swedish_krona_to_usd  swiss_franc_to_usd  thai_baht_to_usd  \
    date                                                                     
    2004-01-02              0.138889                 NaN               NaN   
    2004-01-05              0.139665            0.812876          0.025393   
    2004-01-06                   NaN            0.813008          0.025508   
    2004-01-07              0.139519            0.807363          0.025602   
    2004-01-08              0.138600            0.802053          0.025630   
    
                trinidadian_dollar_to_usd  tunisian_dinar_to_usd  \
    date                                                           
    2004-01-02                        NaN                    NaN   
    2004-01-05                        NaN                    NaN   
    2004-01-06                   0.159727                    NaN   
    2004-01-07                   0.159393                    NaN   
    2004-01-08                        NaN                    NaN   
    
                uae_dirham_to_usd  uruguayan_peso_to_usd  bolivar_fuerte_to_usd  
    date                                                                         
    2004-01-02           0.272294                    NaN                    NaN  
    2004-01-05           0.272294                    NaN                    NaN  
    2004-01-06           0.272294                    NaN                    NaN  
    2004-01-07           0.272294                    NaN                    NaN  
    2004-01-08           0.272294                    NaN                    NaN  
    
    [5 rows x 51 columns]
    <class 'pandas.core.frame.DataFrame'>
    Index: 4762 entries, 2004-01-02 to 2022-11-14
    Data columns (total 51 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   chinese_yuan_to_usd         4419 non-null   float64
     1   euro_to_usd                 4701 non-null   float64
     2   japanese_yen_to_usd         4503 non-null   float64
     3   uk_pound_to_usd             4678 non-null   float64
     4   us_dollar_to_usd            4762 non-null   float64
     5   algerian_dinar_to_usd       2889 non-null   float64
     6   australian_dollar_to_usd    4551 non-null   float64
     7   bahrain_dinar_to_usd        3762 non-null   float64
     8   botswana_pula_to_usd        4453 non-null   float64
     9   brazilian_real_to_usd       4308 non-null   float64
     10  brunei_dollar_to_usd        4468 non-null   float64
     11  canadian_dollar_to_usd      4407 non-null   float64
     12  chilean_peso_to_usd         4532 non-null   float64
     13  colombian_peso_to_usd       4058 non-null   float64
     14  czech_koruna_to_usd         4567 non-null   float64
     15  danish_krone_to_usd         4547 non-null   float64
     16  hungarian_forint_to_usd     3641 non-null   float64
     17  icelandic_krona_to_usd      3669 non-null   float64
     18  indian_rupee_to_usd         4368 non-null   float64
     19  indonesian_rupiah_to_usd    3540 non-null   float64
     20  iranian_rial_to_usd         2762 non-null   float64
     21  israeli_new_shekel_to_usd   4411 non-null   float64
     22  kazakhstani_tenge_to_usd    3099 non-null   float64
     23  korean_won_to_usd           4437 non-null   float64
     24  kuwaiti_dinar_to_usd        4068 non-null   float64
     25  libyan_dinar_to_usd         3760 non-null   float64
     26  malaysian_ringgit_to_usd    4447 non-null   float64
     27  mauritian_rupee_to_usd      4298 non-null   float64
     28  mexican_peso_to_usd         4348 non-null   float64
     29  nepalese_rupee_to_usd       3512 non-null   float64
     30  new_zealand_dollar_to_usd   4550 non-null   float64
     31  norwegian_krone_to_usd      4530 non-null   float64
     32  omani_rial_to_usd           4515 non-null   float64
     33  pakistani_rupee_to_usd      3420 non-null   float64
     34  peruvian_sol_to_usd         2657 non-null   float64
     35  philippine_peso_to_usd      2769 non-null   float64
     36  polish_zloty_to_usd         4562 non-null   float64
     37  qatari_riyal_to_usd         4538 non-null   float64
     38  russian_ruble_to_usd        4416 non-null   float64
     39  saudi_arabian_riyal_to_usd  4487 non-null   float64
     40  singapore_dollar_to_usd     4550 non-null   float64
     41  south_african_rand_to_usd   4311 non-null   float64
     42  sri_lankan_rupee_to_usd     3501 non-null   float64
     43  swedish_krona_to_usd        4502 non-null   float64
     44  swiss_franc_to_usd          4579 non-null   float64
     45  thai_baht_to_usd            4313 non-null   float64
     46  trinidadian_dollar_to_usd   4223 non-null   float64
     47  tunisian_dinar_to_usd       1880 non-null   float64
     48  uae_dirham_to_usd           4509 non-null   float64
     49  uruguayan_peso_to_usd       2706 non-null   float64
     50  bolivar_fuerte_to_usd       2382 non-null   float64
    dtypes: float64(51)
    memory usage: 1.9+ MB
    None
    (4762, 51)
    

As we can see in the following chunk of code, the dataset contains 51 exchange rates from all across the world. There is data since january of 2004, but there are some NaN values for some exchange rates. The dataframe contains 4588 rows (each one representing one date) and 52 columns (The first one represents the Date and each one the remaining columns represents an exhange rate to the US dollar.

# Plots of the time series, to see their dynamics

It is possible to see from the estimated mean that since 2004 some currencies have had losses and some other gains in their value. We should be careful because somo exhnage rates mean: How many units of the currency are necessary to buy one US dollar (as the euro_usd exchange rate) and some others mean: How many US dollars are necessary to buy one unit of the currency(as it is the case with the Colombian Peso). Particularly the EURUSD exhange rate has a positive daily variation mean, then it has lost value against the dollar, and the colombian peso has lost value against the US dollar because the daily percentage variation change is also positive. Now let´s see the daily percentage variation plots.


```python
sns.set(font_scale=1.2)
for i in range(len(df.columns)):
    ex_rate=df.iloc[:,i]
    ex_rate.plot(title=df.columns[i])
    #Rotate xticks (Dates) 45 degrees
    plt.xticks(rotation=45)
    plt.show()
```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    



    
![png](output_5_3.png)
    



    
![png](output_5_4.png)
    



    
![png](output_5_5.png)
    



    
![png](output_5_6.png)
    



    
![png](output_5_7.png)
    



    
![png](output_5_8.png)
    



    
![png](output_5_9.png)
    



    
![png](output_5_10.png)
    



    
![png](output_5_11.png)
    



    
![png](output_5_12.png)
    



    
![png](output_5_13.png)
    



    
![png](output_5_14.png)
    



    
![png](output_5_15.png)
    



    
![png](output_5_16.png)
    



    
![png](output_5_17.png)
    



    
![png](output_5_18.png)
    



    
![png](output_5_19.png)
    



    
![png](output_5_20.png)
    



    
![png](output_5_21.png)
    



    
![png](output_5_22.png)
    



    
![png](output_5_23.png)
    



    
![png](output_5_24.png)
    



    
![png](output_5_25.png)
    



    
![png](output_5_26.png)
    



    
![png](output_5_27.png)
    



    
![png](output_5_28.png)
    



    
![png](output_5_29.png)
    



    
![png](output_5_30.png)
    



    
![png](output_5_31.png)
    



    
![png](output_5_32.png)
    



    
![png](output_5_33.png)
    



    
![png](output_5_34.png)
    



    
![png](output_5_35.png)
    



    
![png](output_5_36.png)
    



    
![png](output_5_37.png)
    



    
![png](output_5_38.png)
    



    
![png](output_5_39.png)
    



    
![png](output_5_40.png)
    



    
![png](output_5_41.png)
    



    
![png](output_5_42.png)
    



    
![png](output_5_43.png)
    



    
![png](output_5_44.png)
    



    
![png](output_5_45.png)
    



    
![png](output_5_46.png)
    



    
![png](output_5_47.png)
    



    
![png](output_5_48.png)
    



    
![png](output_5_49.png)
    



    
![png](output_5_50.png)
    


Now, let´s see the descriptive statistics for the daily percentage change of each exchange rate. We will see these statistics for the daily percentage variation, because it could be also seen as a devaluation indicator, and because usually the exchange rates are not stationary, therefore the mean particularly is not constant over time. It is important to note that the variance (and therefore the standard deviation) is not constant over time neither, therefore the standar deviation calculated by the describe() method is probably not a good estimator of the actual standard deviation of the series


```python
df.pct_change(1).describe()
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
      <th>chinese_yuan_to_usd</th>
      <th>euro_to_usd</th>
      <th>japanese_yen_to_usd</th>
      <th>uk_pound_to_usd</th>
      <th>us_dollar_to_usd</th>
      <th>algerian_dinar_to_usd</th>
      <th>australian_dollar_to_usd</th>
      <th>bahrain_dinar_to_usd</th>
      <th>botswana_pula_to_usd</th>
      <th>brazilian_real_to_usd</th>
      <th>...</th>
      <th>south_african_rand_to_usd</th>
      <th>sri_lankan_rupee_to_usd</th>
      <th>swedish_krona_to_usd</th>
      <th>swiss_franc_to_usd</th>
      <th>thai_baht_to_usd</th>
      <th>trinidadian_dollar_to_usd</th>
      <th>tunisian_dinar_to_usd</th>
      <th>uae_dirham_to_usd</th>
      <th>uruguayan_peso_to_usd</th>
      <th>bolivar_fuerte_to_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4760.000000</td>
      <td>4761.000000</td>
      <td>4760.000000</td>
      <td>4761.000000</td>
      <td>4761.0</td>
      <td>3110.000000</td>
      <td>4761.000000</td>
      <td>4761.0</td>
      <td>4760.000000</td>
      <td>4761.000000</td>
      <td>...</td>
      <td>4761.000000</td>
      <td>4761.000000</td>
      <td>4761.000000</td>
      <td>4760.000000</td>
      <td>4760.000000</td>
      <td>4759.000000</td>
      <td>3090.000000</td>
      <td>4.761000e+03</td>
      <td>3114.000000</td>
      <td>3701.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000036</td>
      <td>-0.000025</td>
      <td>-0.000037</td>
      <td>-0.000068</td>
      <td>0.0</td>
      <td>-0.000196</td>
      <td>0.000007</td>
      <td>0.0</td>
      <td>-0.000201</td>
      <td>-0.000082</td>
      <td>...</td>
      <td>-0.000131</td>
      <td>-0.000128</td>
      <td>-0.000047</td>
      <td>0.000078</td>
      <td>0.000025</td>
      <td>-0.000014</td>
      <td>-0.000218</td>
      <td>9.726637e-11</td>
      <td>-0.000223</td>
      <td>-0.001386</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.001737</td>
      <td>0.005789</td>
      <td>0.006220</td>
      <td>0.006210</td>
      <td>0.0</td>
      <td>0.002595</td>
      <td>0.007983</td>
      <td>0.0</td>
      <td>0.007670</td>
      <td>0.009559</td>
      <td>...</td>
      <td>0.011795</td>
      <td>0.001774</td>
      <td>0.007706</td>
      <td>0.006780</td>
      <td>0.003114</td>
      <td>0.002202</td>
      <td>0.003515</td>
      <td>1.394897e-05</td>
      <td>0.004725</td>
      <td>0.026922</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.018235</td>
      <td>-0.046251</td>
      <td>-0.042172</td>
      <td>-0.079842</td>
      <td>0.0</td>
      <td>-0.035401</td>
      <td>-0.096489</td>
      <td>0.0</td>
      <td>-0.132697</td>
      <td>-0.085340</td>
      <td>...</td>
      <td>-0.142533</td>
      <td>-0.030710</td>
      <td>-0.049669</td>
      <td>-0.080476</td>
      <td>-0.024930</td>
      <td>-0.011374</td>
      <td>-0.033383</td>
      <td>-6.802721e-04</td>
      <td>-0.042424</td>
      <td>-0.997010</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.000382</td>
      <td>-0.003100</td>
      <td>-0.003257</td>
      <td>-0.003345</td>
      <td>0.0</td>
      <td>-0.001184</td>
      <td>-0.003979</td>
      <td>0.0</td>
      <td>-0.003481</td>
      <td>-0.004098</td>
      <td>...</td>
      <td>-0.005477</td>
      <td>-0.000229</td>
      <td>-0.003985</td>
      <td>-0.003112</td>
      <td>-0.001410</td>
      <td>-0.001320</td>
      <td>-0.000961</td>
      <td>0.000000e+00</td>
      <td>-0.001815</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000486</td>
      <td>0.003103</td>
      <td>0.003003</td>
      <td>0.003354</td>
      <td>0.0</td>
      <td>0.000929</td>
      <td>0.004358</td>
      <td>0.0</td>
      <td>0.003411</td>
      <td>0.004268</td>
      <td>...</td>
      <td>0.005722</td>
      <td>0.000055</td>
      <td>0.004022</td>
      <td>0.003258</td>
      <td>0.001467</td>
      <td>0.001090</td>
      <td>0.000384</td>
      <td>0.000000e+00</td>
      <td>0.001684</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.020568</td>
      <td>0.041203</td>
      <td>0.050188</td>
      <td>0.042012</td>
      <td>0.0</td>
      <td>0.032816</td>
      <td>0.074184</td>
      <td>0.0</td>
      <td>0.153141</td>
      <td>0.096561</td>
      <td>...</td>
      <td>0.185975</td>
      <td>0.020556</td>
      <td>0.047779</td>
      <td>0.167314</td>
      <td>0.033489</td>
      <td>0.010166</td>
      <td>0.044717</td>
      <td>6.807352e-04</td>
      <td>0.026944</td>
      <td>0.209316</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 51 columns</p>
</div>



It is possible to see from the estimated mean that since 2004 some currencies have had losses and some other gains in their value. We should be careful because somo exhnage rates mean: How many units of the currency are necessary to buy one US dollar (as the euro_usd exchange rate) and some others mean: How many US dollars are necessary to buy one unit of the currency(as it is the case with the Colombian Peso). Particularly the EURUSD exhange rate has a positive daily variation mean, then it has lost value against the dollar, and the colombian peso has lost value against the US dollar because the daily percentage variation change is also positive. Now let´s see the daily percentage variation plots.


```python
df_returns=df.pct_change(1)
for i in range(1,len(df_returns.columns)):
    ex_rate=df_returns.iloc[:,i]
    ex_rate.plot(title=df_returns.columns[i])
    #Rotate xticks (Dates) 45 degrees
    plt.xticks(rotation=45)
    plt.show()
```


    
![png](output_9_0.png)
    



    
![png](output_9_1.png)
    



    
![png](output_9_2.png)
    



    
![png](output_9_3.png)
    



    
![png](output_9_4.png)
    



    
![png](output_9_5.png)
    



    
![png](output_9_6.png)
    



    
![png](output_9_7.png)
    



    
![png](output_9_8.png)
    



    
![png](output_9_9.png)
    



    
![png](output_9_10.png)
    



    
![png](output_9_11.png)
    



    
![png](output_9_12.png)
    



    
![png](output_9_13.png)
    



    
![png](output_9_14.png)
    



    
![png](output_9_15.png)
    



    
![png](output_9_16.png)
    



    
![png](output_9_17.png)
    



    
![png](output_9_18.png)
    



    
![png](output_9_19.png)
    



    
![png](output_9_20.png)
    



    
![png](output_9_21.png)
    



    
![png](output_9_22.png)
    



    
![png](output_9_23.png)
    



    
![png](output_9_24.png)
    



    
![png](output_9_25.png)
    



    
![png](output_9_26.png)
    



    
![png](output_9_27.png)
    



    
![png](output_9_28.png)
    



    
![png](output_9_29.png)
    



    
![png](output_9_30.png)
    



    
![png](output_9_31.png)
    



    
![png](output_9_32.png)
    



    
![png](output_9_33.png)
    



    
![png](output_9_34.png)
    



    
![png](output_9_35.png)
    



    
![png](output_9_36.png)
    



    
![png](output_9_37.png)
    



    
![png](output_9_38.png)
    



    
![png](output_9_39.png)
    



    
![png](output_9_40.png)
    



    
![png](output_9_41.png)
    



    
![png](output_9_42.png)
    



    
![png](output_9_43.png)
    



    
![png](output_9_44.png)
    



    
![png](output_9_45.png)
    



    
![png](output_9_46.png)
    



    
![png](output_9_47.png)
    



    
![png](output_9_48.png)
    



    
![png](output_9_49.png)
    


Now, let´s get the correlation matrix for the daily returns of the currencies exchange rates


```python
# Compute the correlation matrix
corr=df_returns.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.set(font_scale=0.5)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```

    C:\Users\Mua\AppData\Local\Temp/ipykernel_15468/4248955765.py:5: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      mask = np.zeros_like(corr, dtype=np.bool)
    




    <AxesSubplot:>




    
![png](output_11_2.png)
    


The Danish Krone and EuroUSD seem to have a really strong negative correlation.Let´s take them and create a scatter plot with its regression line


```python
sns.lmplot(x='euro_to_usd',y='danish_krone_to_usd',data=df_returns)
```




    <seaborn.axisgrid.FacetGrid at 0x22fe044d910>




    
![png](output_13_1.png)
    


Effectively, the show a really strong correlation, and therefore we could think in an investment estrategy based on this correlation. We could try to predict the returns of one of the exchange rates based on the returns of the other one

Let's analize just the colombian peso, because we are not sure about what do all of the exchange rates mean. Then, let´s plot the Colombian Peso exchange rate and then transform it to how many colombian peso unit are necessary to buy one US Dollar (We are going to continue the analysis with this interpretation of the exchange rate), and then plot it again, with its daily returns (daily percentage variation).


```python
Colombia=df["colombian_peso_to_usd"]
Colombia.plot(title='COPUSD Exchange Rate')
#Rotate xticks (Dates) 45 degrees
plt.xticks(rotation=45)
plt.show()
Colombia=1/Colombia
Colombia.plot(title='USDCOP Exchange Rate')
#Rotate xticks (Dates) 45 degrees
plt.xticks(rotation=45)
plt.show()
#Now plot the daily returns
Colombia_returns=Colombia.pct_change(1)
Colombia_returns.plot(title='USDCOP Exchange Rate Daily Returns')
#Rotate xticks (Dates) 45 degrees
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_16_0.png)
    



    
![png](output_16_1.png)
    



    
![png](output_16_2.png)
    


As you can see the USDCOP Exchange Rate Daily Returns series seem to has a constan mean along the time (Probably not constant, but at least around 0, it could be modeled with an ARMA model), but the variance along the time series seems to have some specific clusters of higher volatility, for example, it can be seen from the daily returns plot that in 2007 the Colombian Peso gained value against the US Dollar because of the subrpime crisis in United States.


```python
sns.distplot( Colombia_returns)
np.mean(Colombia_returns)
```

    C:\Users\Mua\anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    9.838138782897928e-05




    
![png](output_18_2.png)
    


We can see that the mean seems to be 0, but it is actually higher and therefore in the long run the colombian currency have devalued
