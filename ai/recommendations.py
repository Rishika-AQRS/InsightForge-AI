"""
AI Smart Recommendations Engine
"""
import pandas as pd
import numpy as numpy
from typing import Dict, List
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


def analyze_data_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze data patterns for recommendations
    
    Returns:
        Dictionary with analyzed patterns
    """
    patterns = {
        'skewed_columns': [],
        'high_correlations': [],
        'outlier_columns': [],
        'data_type_issues': [],
        'missing_patterns': {}
    }
    
    # 1. Skewness Analysis
    numeric_cols = df.select_dtypes(include=[numpy.number]).columns
    
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:  # Highly skewed
            patterns['skewed_columns'].append({
                'column': col,
                'skewness': skewness,
                'direction': 'positive' if skewness > 0 else 'negative'
            })
    
    # 2. Correlation Analysis
    if len(numeric_cols) >= 2:
        correlation = df[numeric_cols].corr()
        
        for i in range(len(correlation.columns)):
            for j in range(i + 1, len(correlation.columns)):
                corr_value = correlation.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation
                    patterns['high_correlations'].append({
                        'column1': correlation.columns[i],
                        'column2': correlation.columns[j],
                        'correlation': corr_value
                    })
    
    # 3. Outlier Analysis
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            outlier_pct = (outliers / len(df) * 100)
            patterns['outlier_columns'].append({
                'column': col,
                'outlier_count': outliers,
                'outlier_pct': outlier_pct
            })
    
    # 4. Data Type Issues
    for col in df.columns:
        # Check for mixed types
        if df[col].dtype == 'object':
            # Check if column could be numeric
            try:
                numeric_test = pd.to_numeric(df[col], errors='raise')
                patterns['data_type_issues'].append({
                    'column': col,
                    'issue': 'Could be numeric but stored as string',
                    'suggestion': 'Convert to numeric type'
                })
            except:
                pass
        
        # Check for boolean stored as object
        unique_vals = df[col].unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset([True, False, 'True', 'False', 0, 1]):
            patterns['data_type_issues'].append({
                'column': col,
                'issue': 'Boolean data stored as object',
                'suggestion': 'Convert to boolean type'
            })
    
    # 5. Missing Patterns
    missing_per_col = df.isnull().sum()
    missing_pct = (missing_per_col / len(df) * 100)
    
    for col in missing_pct[missing_pct > 0].index:
        patterns['missing_patterns'][col] = {
            'missing_count': int(missing_per_col[col]),
            'missing_pct': float(missing_pct[col])
        }
    
    return patterns


def generate_ai_recommendations(df: pd.DataFrame, patterns: Dict) -> List[str]:
    """
    Generate AI-powered recommendations based on data patterns
    
    Args:
        df: DataFrame
        patterns: Analyzed patterns
    
    Returns:
        List of recommendation strings
    """
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return generate_simple_recommendations(patterns)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Build pattern summary
    pattern_summary = f"""
Data Patterns Analysis:

Skewed Columns:
{json.dumps(patterns['skewed_columns'], indent=2)}

High Correlations:
{json.dumps(patterns['high_correlations'], indent=2)}

Outlier Columns:
{json.dumps(patterns['outlier_columns'], indent=2)}

Data Type Issues:
{json.dumps(patterns['data_type_issues'], indent=2)}

Missing Patterns:
{json.dumps(patterns['missing_patterns'], indent=2)}
"""
    
    prompt = f"""
You are a data scientist. Based on the following data patterns analysis, provide 5-7 actionable recommendations for improving the dataset and extracting better insights.

{pattern_summary}

Provide recommendations for:
1. Data transformations (skewness, outliers)
2. Feature engineering (correlations)
3. Data cleaning (missing values, type issues)
4. Modeling suggestions
5. Business insights

Format each recommendation as:
**Recommendation #**: Clear actionable advice
- Why: Brief explanation
- How: Specific implementation

Be specific and actionable.
"""
    
    try:
        response = model.generate_content(prompt)
        recommendations = response.text.split('\n\n')
        return [rec.strip() for rec in recommendations if rec.strip()]
    except Exception as e:
        return generate_simple_recommendations(patterns)


def generate_simple_recommendations(patterns: Dict) -> List[str]:
    """
    Generate simple recommendations without AI (fallback)
    
    Args:
        patterns: Analyzed patterns
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Skewness recommendations
    for col in patterns['skewed_columns']:
        if col['direction'] == 'positive':
            recommendations.append(
                f"**Log Transformation for {col['column']}**\n"
                f"- Why: Column is positively skewed (skewness: {col['skewness']:.2f})\n"
                f"- How: Apply log transformation: `df['{col['column']}'] = np.log(df['{col['column']}'] + 1)`"
            )
        else:
            recommendations.append(
                f"**Transform {col['column']}**\n"
                f"- Why: Column is negatively skewed (skewness: {col['skewness']:.2f})\n"
                f"- How: Apply transformation: `df['{col['column']}'] = -np.log(-df['{col['column']}'] + 1)`"
            )
    
    # Correlation recommendations
    for corr in patterns['high_correlations']:
        recommendations.append(
            f"**Feature Combination: {corr['column1']} & {corr['column2']}**\n"
            f"- Why: Strong correlation ({corr['correlation']:.2f}) indicates relationship\n"
            f"- How: Create combined feature or use for predictive modeling"
        )
    
    # Outlier recommendations
    for col in patterns['outlier_columns']:
        if col['outlier_pct'] > 10:
            recommendations.append(
                f"**Handle Outliers in {col['column']}**\n"
                f"- Why: {col['outlier_pct']:.1f}% outliers detected\n"
                f"- How: Use IQR method to cap outliers or remove extreme values"
            )
    
    # Missing value recommendations
    for col, stats in patterns['missing_patterns'].items():
        if stats['missing_pct'] > 20:
            recommendations.append(
                f"**Address Missing Values in {col}**\n"
                f"- Why: {stats['missing_pct']:.1f}% missing (high)\n"
                f"- How: Consider removing column or using advanced imputation (KNN, MICE)"
            )
        else:
            recommendations.append(
                f"**Impute Missing Values in {col}**\n"
                f"- Why: {stats['missing_pct']:.1f}% missing\n"
                f"- How: Use mean/median imputation: `df['{col}'].fillna(df['{col}'].median())`"
            )
    
    # Data type recommendations
    for issue in patterns['data_type_issues']:
        recommendations.append(
            f"**Fix Data Type for {issue['column']}**\n"
            f"- Why: {issue['issue']}\n"
            f"- How: {issue['suggestion']}"
        )
    
    return recommendations[:7]  # Return max 7 recommendations


def display_recommendations(recommendations: List[str]):
    """
    Display recommendations in professional card format
    
    Args:
        recommendations: List of recommendation strings
    """
    import streamlit as st
    
    st.header("💡 AI Smart Recommendations")
    st.markdown("Based on data pattern analysis, here are actionable recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        with st.card():
            st.markdown(rec)