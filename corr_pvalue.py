import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def pearson_with_pvalues(df, numeric_columns):
    '''Calcula dos matrices, una con los valores de la correlación 
    de Pearson y la segunda con sus p-valores correspondientes.
    df es el dataframe del que se quiere calcular y numeric_columns
    son las columnas numéricas de la que se quiere calcular.
    
    ¡IMPORTANTE!: Las función depende de la función perasonr
    de la librería scipy.stats. Si salta el error ejecutar antes de 
    la función 
                from scipy.stats import pearsonr, spearmanr'''
    corr_matrix = pd.DataFrame(index=numeric_columns, columns=numeric_columns, dtype=float)
    p_values = pd.DataFrame(index=numeric_columns, columns=numeric_columns, dtype=float)
    
    for i in numeric_columns:
        for j in numeric_columns:
            corr, p_val = pearsonr(df[i], df[j])
            corr_matrix.loc[i, j] = corr
            p_values.loc[i, j] = p_val
    
    return corr_matrix, p_values

def spearman_with_pvalues(df, numeric_columns):
    '''Calcula dos matrices, una con los valores de la correlación 
    de Spearman y la segunda con sus p-valores correspondientes.
    df es el dataframe del que se quiere calcular y numeric_columns
    son las columnas numéricas de la que se quiere calcular.
    
    ¡IMPORTANTE!: Las función depende de la función spearmanr
    de la librería scipy.stats. Si salta el error ejecutar antes de 
    la función 
                from scipy.stats import pearsonr, spearmanr'''
    corr_matrix = pd.DataFrame(index=numeric_columns, columns=numeric_columns, dtype=float)
    p_values = pd.DataFrame(index=numeric_columns, columns=numeric_columns, dtype=float)
    
    for i in numeric_columns:
        for j in numeric_columns:
            corr, p_val = spearmanr(df[i], df[j])
            corr_matrix.loc[i, j] = corr
            p_values.loc[i, j] = p_val
    
    return corr_matrix, p_values