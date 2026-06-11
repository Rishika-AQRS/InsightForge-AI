"""
Dataset Quality Score Calculator
"""
import pandas as pd
import numpy as numpy
from typing import Dict, List, Tuple
import streamlit as st


def calculate_dataset_health_score(df: pd.DataFrame) -> Dict:
    """
    Calculate overall dataset health score (0-100)
    
    Factors:
    - Missing values (25 points)
    - Duplicate rows (20 points)
    - Outliers (15 points)
    - Data consistency (20 points)
    - Column completeness (20 points)
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with score, strengths, weaknesses, recommendations
    """
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 1. Missing Values Score (25 points)
    missing_per_col = df.isnull().sum()
    total_missing = missing_per_col.sum()
    missing_pct = (total_missing / (total_rows * total_cols) * 100)
    
    if missing_pct == 0:
        missing_score = 25
    elif missing_pct < 5:
        missing_score = 20
    elif missing_pct < 10:
        missing_score = 15
    elif missing_pct < 20:
        missing_score = 10
    else:
        missing_score = 5
    
    # 2. Duplicate Rows Score (20 points)
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / total_rows * 100)
    
    if duplicate_pct == 0:
        duplicate_score = 20
    elif duplicate_pct < 1:
        duplicate_score = 18
    elif duplicate_pct < 5:
        duplicate_score = 15
    elif duplicate_pct < 10:
        duplicate_score = 10
    else:
        duplicate_score = 5
    
    # 3. Outliers Score (15 points)
    numeric_cols = df.select_dtypes(include=[numpy.number]).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = int(outliers)
    
    total_outliers = sum(outlier_counts.values())
    outlier_pct = (total_outliers / total_rows * 100)
    
    if outlier_pct == 0:
        outlier_score = 15
    elif outlier_pct < 5:
        outlier_score = 13
    elif outlier_pct < 10:
        outlier_score = 10
    elif outlier_pct < 20:
        outlier_score = 7
    else:
        outlier_score = 3
    
    # 4. Data Consistency Score (20 points)
    # Check for inconsistent data types, empty strings, etc.
    consistency_issues = 0
    
    for col in df.columns:
        # Empty strings
        if df[col].dtype == 'object':
            empty_strings = df[col].strip().eq('').sum()
            consistency_issues += empty_strings
        
        # Infinite values
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            inf_values = numpy.isinf(df[col]).sum()
            consistency_issues += inf_values
    
    if consistency_issues == 0:
        consistency_score = 20
    elif consistency_issues < 5:
        consistency_score = 18
    elif consistency_issues < 10:
        consistency_score = 15
    elif consistency_issues < 20:
        consistency_score = 10
    else:
        consistency_score = 5
    
    # 5. Column Completeness Score (20 points)
    # Check if all columns have meaningful data
    empty_cols = df.columns[df.isnull().all()].tolist()
    single_value_cols = []
    
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
    
    completeness_issues = len(empty_cols) + len(single_value_cols)
    
    if completeness_issues == 0:
        completeness_score = 20
    elif completeness_issues == 1:
        completeness_score = 18
    elif completeness_issues < 3:
        completeness_score = 15
    elif completeness_issues < 5:
        completeness_score = 10
    else:
        completeness_score = 5
    
    # Calculate total score
    total_score = missing_score + duplicate_score + outlier_score + consistency_score + completeness_score
    
    # Strengths
    strengths = []
    if missing_score == 25:
        strengths.append("✅ No missing values")
    if duplicate_score == 20:
        strengths.append("✅ No duplicate rows")
    if outlier_score == 15:
        strengths.append("✅ No outliers detected")
    if consistency_score == 20:
        strengths.append("✅ Data is consistent")
    if completeness_score == 20:
        strengths.append("✅ All columns have meaningful data")
    
    # Weaknesses
    weaknesses = []
    if missing_score < 20:
        weaknesses.append(f"⚠️ {missing_pct:.1f}% missing values")
    if duplicate_score < 18:
        weaknesses.append(f"⚠️ {duplicate_pct:.1f}% duplicate rows")
    if outlier_score < 13:
        weaknesses.append(f"⚠️ {outlier_pct:.1f}% outliers detected")
    if consistency_score < 18:
        weaknesses.append(f"⚠️ {consistency_issues} consistency issues")
    if completeness_score < 18:
        weaknesses.append(f"⚠️ {completeness_issues} columns with issues")
    
    # Recommendations
    recommendations = []
    if missing_score < 20:
        recommendations.append("🔹 Consider imputing missing values (mean/median/mode)")
    if duplicate_score < 18:
        recommendations.append("🔹 Remove duplicate rows")
    if outlier_score < 13:
        recommendations.append("🔹 Investigate and handle outliers (IQR/Z-score)")
    if consistency_score < 18:
        recommendations.append("🔹 Clean inconsistent data (empty strings, infinite values)")
    if completeness_score < 18:
        recommendations.append("🔹 Review empty or single-value columns")
    
    return {
        'score': total_score,
        'max_score': 100,
        'missing_score': missing_score,
        'duplicate_score': duplicate_score,
        'outlier_score': outlier_score,
        'consistency_score': consistency_score,
        'completeness_score': completeness_score,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations,
        'details': {
            'total_rows': total_rows,
            'total_cols': total_cols,
            'total_missing': total_missing,
            'missing_pct': missing_pct,
            'duplicate_rows': duplicate_rows,
            'duplicate_pct': duplicate_pct,
            'total_outliers': total_outliers,
            'outlier_pct': outlier_pct,
            'consistency_issues': consistency_issues,
            'completeness_issues': completeness_issues
        }
    }


def display_quality_score(quality_result: Dict):
    """
    Display quality score with gauges and progress bars
    
    Args:
        quality_result: Result from calculate_dataset_health_score
    """
    import streamlit as st
    
    st.header("📊 Dataset Quality Score")
    
    # Main score gauge
    score = quality_result['score']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Overall Score", f"{score}/100")
    
    with st.empty():
        # Progress bar
        st.progress(score / 100)
    
    # Score categorization
    if score >= 90:
        score_label = "Excellent ✨"
        score_color = "#4CAF50"  # Green
    elif score >= 80:
        score_label = "Good ✅"
        score_color = "#8BC34A"  # Light Green
    elif score >= 70:
        score_label = "Fair ⚠️"
        score_color = "#FFC107"  # Orange
    elif score >= 60:
        score_label = "Poor ❌"
        score_color = "#FF9800"  # Dark Orange
    else:
        score_label = "Critical 🔴"
        score_color = "#F44336"  # Red
    
    st.markdown(f"**Quality Level:** <span style='color:{score_color}; font-size:1.2rem'>{score_label}</span>", unsafe_allow_html=True)
    
    st.divider()
    
    # Component scores
    st.subheader("📈 Component Scores")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("**Missing Values** (25 pts)")
        st.progress(quality_result['missing_score'] / 25)
        st.text(f"Score: {quality_result['missing_score']}/25")
    
    with comp_col2:
        st.markdown("**Duplicate Rows** (20 pts)")
        st.progress(quality_result['duplicate_score'] / 20)
        st.text(f"Score: {quality_result['duplicate_score']}/20")
    
    with comp_col1:
        st.markdown("**Outliers** (15 pts)")
        st.progress(quality_result['outlier_score'] / 15)
        st.text(f"Score: {quality_result['outlier_score']}/15")
    
    with comp_col2:
        st.markdown("**Consistency** (20 pts)")
        st.progress(quality_result['consistency_score'] / 20)
        st.text(f"Score: {quality_result['consistency_score']}/20")
    
    with comp_col1:
        st.markdown("**Completeness** (20 pts)")
        st.progress(quality_result['completeness_score'] / 20)
        st.text(f"Score: {quality_result['completeness_score']}/20")
    
    st.divider()
    
    # Strengths
    if quality_result['strengths']:
        st.subheader("✅ Strengths")
        for strength in quality_result['strengths']:
            st.success(strength)
    
    # Weaknesses
    if quality_result['weaknesses']:
        st.subheader("⚠️ Weaknesses")
        for weakness in quality_result['weaknesses']:
            st.warning(weakness)
    
    # Recommendations
    if quality_result['recommendations']:
        st.subheader("🔹 Recommendations")
        for rec in quality_result['recommendations']:
            st.info(rec)
    
    # Details
    st.divider()
    st.subheader("📋 Detailed Metrics")
    
    details = quality_result['details']
    details_df = pd.DataFrame({
        'Metric': [
            'Total Rows',
            'Total Columns',
            'Total Missing Values',
            'Missing Percentage',
            'Duplicate Rows',
            'Duplicate Percentage',
            'Total Outliers',
            'Outlier Percentage',
            'Consistency Issues',
            'Completeness Issues'
        ],
        'Value': [
            details['total_rows'],
            details['total_cols'],
            details['total_missing'],
            f"{details['missing_pct']:.2f}%",
            details['duplicate_rows'],
            f"{details['duplicate_pct']:.2f}%",
            details['total_outliers'],
            f"{details['outlier_pct']:.2f}%",
            details['consistency_issues'],
            details['completeness_issues']
        ]
    })
    
    st.dataframe(details_df, use_container_width=True, hide_index=True)