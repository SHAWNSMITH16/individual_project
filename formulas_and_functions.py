import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt
from scipy.stats import pearsonr, spearmanr
from scipy import stats

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from env import get_connection
import prepare


# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

seed = 23


def new_stat_cols(df):
    
    df['ppds'] = df.Hrry + df.QBKD + (df.Sk * 2.5) + df.QBHits

    df['TDs'] = df.TD + df['TD.1']
    
    return df



def player_ratings_cols(df):
    
    df['DBR'] = (((df.Int * 5) + (df.PD * 3.75) + (df.FF * 3) + 
                     (df.FR * 4) + (df.TDs * 7)) * 1.5)

#    df['DBR'] = np.where(df['Pos'] == 'DL', (((df.Int * 5) + (df.PD * 3.5) + (df.FF * 3) + 
#                     (df.FR * 4) + (df.TDs * 7)) * 1.5), ((((df.Int * 5) + (df.PD * 3.5) + (df.FF * 3) + 
#                     (df.FR * 4) + (df.TDs * 7)) * 1.5) - df['Cmp%']))
    
    df['DLR'] = ((df.ppds * 1.5) + (df.Comb * .25) + (df.TFL * 1) + 
                   (df.FF * 1.5) + (df.FR * 2.5) + (df.TDs * 7))
    
    df['LBR'] = ((df.Comb * .25) + (df.ppds * .5) + (df.Int * 2.5) + 
                   (df.FF * 1.5) + (df.PD * 2) + (df.TDs * 7))
    
    df['ODR'] = round((df.DBR + df.DLR + df.LBR) / 3, 1)
     
    return df


def drumroll_please(df, rtg):
    results = df[df['ODR'] >= rtg]
    results.loc[len(results.index)] = ['League Average', 'N/A', 'N/A', df.Int.mean(), df['Cmp%'].mean(),
                                         df.Sk.mean(), df.Comb.mean(), df.PD.mean(), df.FF.mean(),
                                         df.FR.mean(), df.TFL.mean(), df.ppds.mean(), df.TDs.mean(),
                                         df.DBR.mean(), df.DLR.mean(), df.LBR.mean(), df.ODR.mean()] 
    return results

###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------

### Visual functions

def DB_stat_correlation(df):
    
    sns.countplot(df['Int'], hue = df.Pos, palette = 'Paired')
    plt.title('Correlation of Interceptions\nto Overall Defensive Rating', fontdict = { 'fontsize': 20})
    plt.ylabel('Players per Position', fontdict = { 'fontsize': 15})
    plt.xlabel('Amount of Interceptions', fontdict = { 'fontsize': 15})
    plt.text(7, 135, 'Takeaway: This correlation shows that\nInterceptions favor defensive backs\nmore than Linemen or Linebackers', fontsize = 15, bbox = dict(facecolor = 'cyan', alpha = .5))
    plt.show()


def DL_stat_correlation(df):
    
    sns.relplot(y = 'ppds', x = 'ODR', hue = 'Pos', palette = 'gist_ncar', data = df)
    plt.title('Pass Play Disruption Score Correlation\nto Overall Rating', fontdict = { 'fontsize': 15})
    plt.ylabel('Pass Play Disruption Score', fontdict = { 'fontsize': 15})
    plt.xlabel('Overall Defensive Rating', fontdict = { 'fontsize': 15})
    plt.show() 

def ppds_top_5(df):

    fig, ax = plt.subplots()
    plt.title('Top 5 Pass Play\nDisruption Leaders')
    plt.ylabel('Pass PLay Disruption Score')
    plt.xlabel('Top 5 Players vs. League Average')
    ax.bar(df['Player'], df['ppds'], 
           color = ['Red', 'lightcoral', 'limegreen', 
                    'tomato', 'orangered', 'black'], edgecolor = 'black')
    ax.set_ylim(0, 140)
    plt.xticks(rotation=45)
    plt.text(6, 80, '* - Pro-Bowler', fontsize = 10, bbox = dict(facecolor = 'cyan', alpha = .5))
    plt.text(6, 65, '+ - 1st Team All-Pro', fontsize = 10, bbox = dict(facecolor = 'cyan', alpha = .5))

    plt.show()
    
def LB_tackle_correlation(df):
    positions = ('LineBacker', 'Defensive Back\n ', 'Defensive Lineman')
    index = np.arange(3)
    sns.barplot(y = df.Comb, x = df.Pos, data = df, palette = 'husl', edgecolor = 'black')
    plt.xticks(ticks = (0, 1, 2), labels = positions)
    plt.title('Average Tackles by Position', fontdict = { 'fontsize': 15})
    plt.ylabel('Average Tackles', fontdict = { 'fontsize': 15})
    plt.xlabel('Positions', fontdict = { 'fontsize': 15})
    plt.text(2.75, 35, 'Takeaway: Tackles are much more\nsignificant for Linebackers than\nthe other two positions', 
             fontsize = 15, bbox = dict(facecolor = 'cyan', alpha = .5))

    plt.show()
    
def model_report():
    data = {'Model': ['Linear Regression', 'Lasso + Lars', 'Polynomial Regression'],
            'Train Predictions': [221977.25, 221977.25, 220570.62],
            'Validate Predictions': [222972.52, 222972.52, 220216.19]}
    return pd.DataFrame(data)



###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------

### Stats functions

def confirm_statistical_correlation(df, var1, var2):
    r, p_value = spearmanr(var1, var2)
    if r > .50:
        print(f'There is a strong positive correlation with an R-value of: {r}\nP-value of: {p_value}')
    else:
        print('Garbage')
        
