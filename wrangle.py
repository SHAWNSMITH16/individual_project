import pandas as pd
import os
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt
from scipy.stats import pearsonr, spearmanr

from env import get_connection
import prepare

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

def any_given_sunday():
    
    df_adv18 = pd.read_csv('2018_adv.csv')
    df_adv19 = pd.read_csv('2019_adv.csv')
    df_adv20 = pd.read_csv('2020_adv.csv')
    df_adv21 = pd.read_csv('2021_adv.csv')
    df_adv22 = pd.read_csv('2022_adv.csv')

    df_std18 = pd.read_csv('2018_std.csv')
    df_std19 = pd.read_csv('2019_std.csv')
    df_std20 = pd.read_csv('2020_std.csv')
    df_std21 = pd.read_csv('2021_std.csv')
    df_std22 = pd.read_csv('2022_std.csv')
    
    return df_adv18, df_adv19, df_adv20, df_adv21, df_adv22, df_std18, df_std19, df_std20, df_std21, df_std22
    
    
def drop(df1, df2):
    
    adv_stat_cols = ['Rk', 'Age', 'G', 'GS', 'Tgt', 'Cmp', 'Yds', 'Yds/Cmp', 'Yds/Tgt', 
                'Rat', 'DADOT', 'Air', 'YAC', 'Bltz', 'MTkl', 'MTkl%', '-9999', 'TD']
    
    std_stat_cols = ['Rk', 'Age', 'G', 'GS', 'Yds', 'Lng', 'Fmb', 'Yds.1', 'Sfty', 
                 '-9999', 'Comb', 'Int', 'Sk', 'Tm', 'Pos', 'Solo', 'Ast']
    
    df1 = df1.drop(columns = adv_stat_cols)
    
    df2 = df2.drop(columns = std_stat_cols)
    
    df18 = df1.merge(df2[['TD', 'PD', 'FF', 'FR', 'TD.1', 'TFL', 'QBHits', 
                                'Player']], on = 'Player', how = 'left')
    
    return df18


def obsolete_col_drop(df):
    
    obsolete_cols = ['Hrry', 'QBKD', 'QBHits', 'Prss', 'TD', 'TD.1']
    
    df = df.drop(columns = obsolete_cols)
    
    return df


def cleanup(df):
    
    df['Cmp%'] = df['Cmp%'].str.rstrip('%').astype('float')
    
    df = df.fillna(0)

    df = df.replace(to_replace = ['LLB', 'RLB', 'RILB', 'LILB', 'LLB/MLB', 'MLB', 'ROLB', 
                                  'LOLB', 'LB/RLB', 'LB/RILB', 'LB/ROLB', 'MLB/RLB', 
                                  'LILB/RIL', 'ROLB/RIL', 'LB/LDE', 'ROLB/LOL', 'RILB/LIL', 
                                  'RLB/MLB', 'RLB/LLB', 'LB/LILB', 'OLB', 'MLB/RILB', 
                                  'LOLB/ROL', 'ROLB/LIL', 'LOLB/RDE', 'LB/OLB'], value = 'LB')
    
    df = df.replace(to_replace = ['LCB', 'RCB', 'SS', 'FS', 'DB/RCB', 'RCB/SS', 'RCB/DB', 
                                  'SS/LCB', 'LCB/FS', 'RCB/LCB', 'LCB/RCB', 'FS/SS', 'DB/LCB', 
                                  'SS/FS', 'RCB/FS', 'DB/FS', 'CB', 'S', 'SS/RLB', 
                                  'CB/RCB', 'CB/DB', 'DB/S'], value = 'DB')
    
    df = df.replace(to_replace = ['DE', 'DT', 'LDE', 'RDE', 'LDT', 'RDT', 'NT', 'RDE/NT', 
                                  'LDT/RDT', 'LDE/RDE', 'NT/RDT', 'DE/ROLB', 'RDE/LDT', 
                                  'DE/RDE', 'RDE/LDE', 'DT/FB', 'RDT/LDT', 'DT/DE', 
                                  'DT/NT', 'DE/LOLB', 'DE/DT', 'LDT/LDE', 'RDT/RDE', 
                                  'RDT/LDE', 'DE/LDE', 'DE/DL', 'DE/OLB'], value = 'DL')
    
    offensive_player = df[(df['Pos'] == 'WR') | (df['Pos'] == 'TE') | (df['Pos'] == 'RB') | 
                          (df['Pos'] == 'FB') | (df['Pos'] == 'C') | (df['Pos'] == 'RG') | 
                          (df['Pos'] == 'QB') | (df['Pos'] == 'LG') | (df['Pos'] == 'T') | 
                          (df['Pos'] == 'LS') | (df['Pos'] == 'RG/C') | (df['Pos'] == 'P') |
                          (df['Pos'] == 'OL') | (df['Pos'] == 'G') | (df['Pos'] == 'K')].index
    
    df = df.drop(offensive_player)
    
    df = df[df['Pos'] != 0]
    
    df = df.drop_duplicates(subset="Player", keep = 'first')
    
    return df



###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------


