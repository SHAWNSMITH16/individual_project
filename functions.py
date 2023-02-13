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

def distributions(wine):
    num_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 
                'rs', 'chlorides', 'free_s02', 'total_s02', 'density', 
                'pH', 'sulphates', 'alcohol', 'quality'] 

    for col in num_cols:
    
        plt.hist(wine[col], color = 'purple')
        plt.title(f'Distribution of {col}')
        plt.grid()
        plt.show()


def show_outliers(df):
    sns.boxplot(data = df)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    labels = df.columns
    plt.xticks(x, labels, rotation = 65)
    plt.show()        
        
        
def density_quality(train):
    p = sns.stripplot(y = train.density, x = train.quality, data = train, size = 2, jitter = .4, palette = 'magma')
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="quality",
                y="density",
                data=train,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.ylabel('Density Level', fontdict = { 'fontsize': 15})
    plt.xlabel('Wine Quality', fontdict = { 'fontsize': 15})
    plt.title('Does Density Affect Wine Quality', fontdict = { 'fontsize': 20})
    plt.show()

    
def t_test(a, b):
    '''
    This function will take in two arguments in the form of a continuous and discrete variable and runs
    an independent t-test and prints whether or not to print whether or not to rejec the Null Hypothesis
    based on those results
    '''
    alpha = 0.05
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print("T-Score is: ", t)
    print("")
    print("P-Value is: ", p/2)
    print("")
    if p / 2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject the null hypothesis as there is a\nsignificant relationship between density and quality of wine.")

        
def chi_sq(a, b):
    '''
    This function will take in two arguments in the for of two discrete variables and runs a chi^2 test
    to determine if the the two variables are independent of each other and prints the results based on the findings
    '''
    alpha = 0.05
    
    result = pd.crosstab(a, b)

    chi2, p, degf, expected = stats.chi2_contingency(result)

    print(f'chi^2  = {chi2:.4f}') 
    print("")
    print(f'p-value = {p:.4f}')
    print("")
    if p / 2 > alpha:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject the null hypothesis as there is a dependence \nbetween the selected feature and quality of wine.")
        


def cluster_sugar_acid(sugar_acid_df):
    df = sugar_acid_df[['rs', 'citric_acid']]
    
    kmeans = KMeans(n_clusters= 3, random_state = seed)

    kmeans.fit(df)

    kmeans.predict(df)
    
    sugar_acid_df['sugar_acid'] = kmeans.predict(df)
    
    sns.relplot(y = 'rs', x = 'citric_acid', hue = 'sugar_acid', palette = 'Accent', data = sugar_acid_df)
    
    plt.show()   
    
    
    
    
# function to add a 'sugar_acid' column to a scaled df
# df = train_scaled, val_scaled or test_scaled

def sugar_acid_col(df):
    
    '''
    this function will create a column on the scaled dataset 
    to allow for regression modelling
    '''
    
    df2 = df[['rs', 'citric_acid']]
    
    kmeans = KMeans(n_clusters = 3, random_state = seed)

    kmeans.fit(df2)

    kmeans.predict(df2)
    
    df['sugar_acid'] = kmeans.predict(df2)
    
    return df

def add_cols(z, a, b, c, d):
    z = sugar_acid_col(z)
    
    a = sugar_acid_col(a)

    b = sugar_acid_col(b)

    c = sugar_acid_col(c)

    d = sugar_acid_col(d)
    
    return z, a, b, c, d
        
def sugar_acid_compare(train_scaled):        
    sns.countplot(train_scaled['sugar_acid'], hue = train_scaled.quality, palette = 'Accent')
    plt.ylabel('Number of Wines')
    plt.xlabel('Sugar and Citric Acid Group')
    plt.title('Are Residual Sugar and\n Citric Acid Related to Quality')
    labels = ['High Acid\nLow Sugar', 'Medium Acid\nHigh Sugar', 'Low Acid\nLow Sugar']
    plt.xticks(ticks = (0, 1, 2), labels = labels)
    plt.show()
        
        
        
        
def cluster_sulphites(sulphites_df):
    df = sulphites_df[['free_s02', 'total_s02']]
    
    kmeans = KMeans(n_clusters= 3, random_state = seed)

    kmeans.fit(df)

    kmeans.predict(df)
    
    sulphites_df['sulphites'] = kmeans.predict(df)
    
    sns.relplot(y = 'free_s02', x = 'total_s02', hue = 'sulphites', palette='Accent', data=sulphites_df)
    
    plt.show()   

    
def sulphites_compare(train_scaled):
    c = sns.diverging_palette(300, 10, s = 90)
    sns.countplot(train_scaled['sulphites'], hue = train_scaled.quality, palette = c, orient = "h")
    plt.ylabel('Number of Wines')
    plt.xlabel('Free and Total S02 Groupings')
    plt.title('Are Free and Total S02 Levels\nGood Indicators of Quality?')
    labels = ['Low Free\nLow Total S02', 'High Free\nHigh Total S02', 'Low Free\nHigh Total S02']
    plt.xticks(ticks = (0, 1, 2), labels = labels)
    plt.show()
     

        
def cluster_sug_dens(sug_dens_df):
    
    df = sug_dens_df[['rs', 'density']]

    kmeans = KMeans(n_clusters= 3, random_state = seed)

    kmeans.fit(df)

    kmeans.predict(df)

    sug_dens_df['sugar_dens'] = kmeans.predict(df)
    
    sns.relplot(y = 'rs', x = 'density', hue='sugar_dens', palette = 'Set1', data = sug_dens_df)
    
    plt.show()
        
    
def sugar_dens_compare(train_scaled):

    sns.countplot(train_scaled['sugar_dens'], hue = train_scaled.quality, palette = "Set1")
    plt.ylabel('Number of Wines')
    plt.xlabel('Sugar and Density Group')
    plt.title('Are Residual Sugar and\n Density Related to Quality')
    labels = ['Low Density\nLow Sugar', 'Med - High Density\nLow Sugar', 'High Density\nHigh Sugar']
    plt.xticks(ticks = (0, 1, 2), labels = labels)
    plt.show()    
    
    
def model_report():
    data = {'Model': ['Linear Regression', 'Lasso + Lars', 'Polynomial Regression'],
            'Train Predictions': [221977.25, 221977.25, 220570.62],
            'Validate Predictions': [222972.52, 222972.52, 220216.19]}
    return pd.DataFrame(data)




def quality_ols(df, col):
    
    '''
    this function runs the OLS Linear Regression model for residual the sugar & citric acid
    cluster on an entered dataframe with the feature 'column_name', comparing it to 
    wine quality rating. It returns the RMSE baseline and the OLS RMSE.
    '''
    
    # getting mean of target variable
    df['quality'].mean()

    # rounding and setting target variable name
    baseline_preds = round(df['quality'].mean(), 3)

    # create a dataframe
    predictions_df = df[[col, 'quality']]

    # MAKE NEW COLUMN ON DF FOR BASELINE PREDICTIONS
    predictions_df['baseline_preds'] = baseline_preds

    # our linear regression model
    ols_model = LinearRegression()
    ols_model.fit(df[[col]], df[['quality']])

    # predicting on density after it's been fit
    ols_model.predict(df[[col]])

    # model predictions from above line of codes with 'yhat' as variable name and append it on to df
    predictions_df['yhat'] = ols_model.predict(df[[col]])

    # computing residual of baseline predictions
    predictions_df['baseline_residual'] = predictions_df['quality'] - predictions_df['baseline_preds']

    # looking at difference between yhat predictions and actual preds ['quality']
    predictions_df['yhat_res'] = predictions_df['yhat'] - predictions_df['quality']

    # finding the RMSE in one step (x = original, y = prediction)
    dens_qual_rmse = sqrt(mean_squared_error(predictions_df['quality'], predictions_df['baseline_preds']))
    print(f'The RMSE on the baseline against wine quality is {round(dens_qual_rmse,4)}.')

    # RMSE of linear regression model
    OLS_rmse = mean_squared_error(predictions_df['yhat'], predictions_df['quality'], squared = False)
    print(f'The RMSE for the OLS Linear Regression model was {round(OLS_rmse, 4)}.')

    
    
    
   
    
# function for tweedie regresor    
 
# tweedie regressor function on train + residual sugar & citric acid cluster

def tweedie_sugar_acid(df, X_df, y_df):
        
    '''
    This function intakes a scaled dataframe, and its X_ and y_ dataframes. 
    It compares against 'quality'.
    It returns the Tweedie Regressor RMSE and the baseline RMSE.
    '''

    # baseline on mean
    baseline_pred_sca = round(df['quality'].mean(), 3)
    
    tweedie_sca = TweedieRegressor()

    # fit the created object to training dataset
    tweedie_sca.fit(X_df[['sugar_acid']], y_df)
    predictions_sca_df = df[['sugar_acid', 'quality']]

    # then predict on X_train
    predictions_sca_df['tweedie_sca'] = tweedie_sca.predict(X_df[['sugar_acid']])
    predictions_sca_df['baseline_pred_sca'] = baseline_pred_sca


    # check the error against the baseline
    tweedie_sca_rmse = sqrt(mean_squared_error(predictions_sca_df['quality'], predictions_sca_df['tweedie_sca']))
    print(f'The RMSE for the Tweedie Regressor model was {round(tweedie_sca_rmse, 4)}.')

    # finding the error cf the baseline
    sca_qual_rmse = sqrt(mean_squared_error(predictions_sca_df['quality'], predictions_sca_df['baseline_pred_sca']))
    print(f'The RMSE on the baseline of sugar & citric acid against wine quality is {round(sca_qual_rmse,4)}.')




    
def tweedie_density(df, X_df, y_df):
           
    '''
    This function intakes a scaled dataframe, and its X_ and y_ dataframes. 
    It compares against 'quality'.
    It returns the Tweedie Regressor RMSE and the baseline RMSE.
    '''

    # baseline on mean
    baseline_pred_d = round(df['quality'].mean(), 3)
    
    # tweedie regresor
    tweedie_d = TweedieRegressor()

    # fit the created object to training dataset
    tweedie_d.fit(X_df[['density']], y_df)
    
    # predictions dataframe
    predictions_d_df = df[['density', 'quality']]

    # then predict on X_train
    predictions_d_df['tweedie_d'] = tweedie_d.predict(X_df[['density']])
    predictions_d_df['baseline_pred_d'] = baseline_pred_d

    # check the error against the baseline
    tweedie_d_rmse = sqrt(mean_squared_error(predictions_d_df['quality'], predictions_d_df['tweedie_d']))
    print(f'The RMSE for the Tweedie Regressor model was {round(tweedie_d_rmse, 4)}.')

    # finding the error cf the baseline
    d_qual_rmse = sqrt(mean_squared_error(predictions_d_df['quality'], predictions_d_df['baseline_pred_d']))
    print(f'The RMSE on the baseline of density against wine quality is {round(d_qual_rmse,4)}.')


    
    
def tts_xy(train, val, test, target):
    
    '''
    This function splits train, val, test into X_train, X_val, X_test
    (the dataframe of features, exludes the target variable) 
    and y-train (target variable), etc
    '''

    X_train = train.drop(columns = [target])
    y_train = train[target]


    X_val = val.drop(columns = [target])
    y_val = val[target]


    X_test = test.drop(columns = [target])
    y_test = test[target]
    
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

    
    
def decision_tree(train_scaled, X_train, y_train):
    
    '''
    this function intakes a scaled_df, X_train & y_train and 
    runs a decision tree on them with a max depth of 11.
    It outputs the accuracy.
    '''
    # setting the baseline for quality to 6
    quality_baseline = (train_scaled['quality'] == 6).mean()
    
    print(f'The baseline of about {round(quality_baseline, 4)} indicates ' + 
          'the likelihood that a wine will score a 6 for its quality rating.')

    # initialise the Decision Tree Classifier = clf
    seed = 23
    clf5 = DecisionTreeClassifier(max_depth = 11, random_state = seed)

    ### fitting the model : 
    clf5 = clf5.fit(X_train, y_train)
    
    # Examining accuracy of Decision Tree Classifier model
    clf5 = MultiOutputClassifier(clf5, n_jobs = -1)
    clf5.fit(X_train, y_train)

    # accurcy of the decision tree model
    print(f'Decision Tree Accuracy, max depth of 11 : {round(clf5.score(X_train, y_train), 4)}')    

def random_forest(X_train, y_train):
    
    '''
    this function intakes X_train & y_train and 
    runs a random forest on them with a max depth of 11.
    It outputs the accuracy.
    '''
    
    # setting random forest classifier to 11 branches
    random = RandomForestClassifier(max_depth = 11, random_state = 23,
                           max_samples = 0.5)        
                            # 50pc of all observations will be placed into each random sample
        
    # training the random forest on the data
    random.fit(X_train, y_train)    

    # scoring the accuracy of the training set
    random.score(X_train, y_train)

    # accurcy of the decision tree model
    print(f'Random Forest Accuracy, max depth of 11 : {round(random.score(X_train, y_train), 4)}')



    
# col = column name ; needs to be entered in 'col_name' format.
# df = train_scaled

def elbow(df, col):
    
    # inertia loop

    inertia = []
    seed = 23

    for i in range (1, 7):

        # clustering increments
        kmeans = KMeans(n_clusters = i, random_state = seed)

        kmeans.fit(df[[col]])

        # append the inertia
        inertia.append(kmeans.inertia_)
        
        
    # creating a df for the sugar-citric acid features
    inertia_sca_df = pd.DataFrame({'n_clusters' : list(range(1,7)),
                                   'inertia' : inertia})

    # elbow of the number for k
    sns.relplot(data = inertia_sca_df, x = 'n_clusters', y = 'inertia', kind = 'line')
    plt.grid()
    plt.show()