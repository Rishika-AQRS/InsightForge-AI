import pandas as pd
import numpy as numpy

def get_dataset_summary(df):
    summary={
     'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'column_types': {col: str(df[col].dtype) for col in df.columns},
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicate_rows': df.duplicated().sum(),   
    }


    numeric_cols=df.select_dtypes(include=[numpy.number]).columns
    if len(numeric_cols)>0:
        summary['numeric_stats']={
            col: {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
            }
            for col in numeric_cols
        }

    return summary


def get_missing_value_analysis(df):
    missing=df.isnull().sum()
    missing_pct=(missing/len(df)*100).round(2)

    columns_with_missing=missing[missing>0]
    missing_pct_with_missing=missing_pct[missing_pct>0]

    return {
        'total_missing':missing.sum(),
        'columns_with_missing': len(columns_with_missing),
        'details':{
            col:{
                'missing_count':int(missing[col]),
                'missing_percentage':float(missing_pct[col])
            }
            for col in columns_with_missing.index
        }
    }


def get_correlation_matrix(df):
    numeric_df=df.select_dtypes(include=[numpy.number])
    if len(numeric_df.columns)<2:
        return None
    
    return numeric_df.corr()


def get_outliers(df, column, method='iqr'):
    if column not in df.columns:
        return []
    
    if method=='iqr':
        q1=df[column].quantile(0.25)
        q3=df[column].quantile(0.75)
        iqr=q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr

        outliers=df[(df[column]<lower_bound) | (df[column]>upper_bound)]
    else:
        mean=df[column].mean()
        std=df[column].std()
        z_scores=(df[column]-mean)/std
        outliers=df[z_scores.abs()>3]

    return outliers.index.tolist()

