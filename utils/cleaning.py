"""
Automated Data Cleaning Studio
"""
import pandas as pd
import numpy as numpy
from typing import Dict, List, Optional, Tuple
import streamlit as st


class DataCleaner:
    """Data cleaning engine with multiple strategies"""
    
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.current_df = df.copy()
        self.cleaning_log: List[str] = []
    
    def log_action(self, action: str):
        """Log cleaning action"""
        self.cleaning_log.append(action)
    
    # ============ MISSING VALUES ============
    
    def remove_rows_with_missing(self, columns: Optional[List[str]] = None):
        """Remove rows with missing values"""
        if columns:
            self.current_df = self.current_df.dropna(subset=columns)
            self.log_action(f"Removed rows with missing values in columns: {columns}")
        else:
            self.current_df = self.current_df.dropna()
            self.log_action("Removed rows with any missing values")
        return self.current_df
    
    def remove_columns_with_missing(self, threshold: float = 0.5):
        """Remove columns with missing values above threshold"""
        missing_pct = self.current_df.isnull().sum() / len(self.current_df)
        cols_to_remove = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_remove:
            self.current_df = self.current_df.drop(columns=cols_to_remove)
            self.log_action(f"Removed columns with >{threshold*100}% missing: {cols_to_remove}")
        
        return self.current_df
    
    def impute_mean(self, columns: Optional[List[str]] = None):
        """Impute missing values with mean"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=[numpy.number]).columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns and self.current_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                mean_val = self.current_df[col].mean()
                self.current_df[col] = self.current_df[col].fillna(mean_val)
        
        self.log_action(f"Imputed missing values with mean for columns: {columns}")
        return self.current_df
    
    def impute_median(self, columns: Optional[List[str]] = None):
        """Impute missing values with median"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=[numpy.number]).columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns and self.current_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                median_val = self.current_df[col].median()
                self.current_df[col] = self.current_df[col].fillna(median_val)
        
        self.log_action(f"Imputed missing values with median for columns: {columns}")
        return self.current_df
    
    def impute_mode(self, columns: Optional[List[str]] = None):
        """Impute missing values with mode (most frequent)"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns and self.current_df[col].dtype == 'object':
                mode_val = self.current_df[col].mode()[0]
                self.current_df[col] = self.current_df[col].fillna(mode_val)
        
        self.log_action(f"Imputed missing values with mode for columns: {columns}")
        return self.current_df
    
    def impute_constant(self, columns: Optional[List[str]] = None, value: any = 0):
        """Impute missing values with constant value"""
        if columns is None:
            columns = self.current_df.columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns:
                self.current_df[col] = self.current_df[col].fillna(value)
        
        self.log_action(f"Imputed missing values with constant {value} for columns: {columns}")
        return self.current_df
    
    # ============ DUPLICATES ============
    
    def detect_duplicates(self, columns: Optional[List[str]] = None):
        """Detect duplicate rows"""
        if columns:
            duplicates = self.current_df.duplicated(subset=columns, keep=False)
        else:
            duplicates = self.current_df.duplicated(keep=False)
        
        duplicate_count = duplicates.sum()
        duplicate_rows = self.current_df[duplicates]
        
        return {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(self.current_df) * 100),
            'rows': duplicate_rows
        }
    
    def remove_duplicates(self, columns: Optional[List[str]] = None):
        """Remove duplicate rows"""
        if columns:
            self.current_df = self.current_df.drop_duplicates(subset=columns, keep='first')
            self.log_action(f"Removed duplicates based on columns: {columns}")
        else:
            self.current_df = self.current_df.drop_duplicates(keep='first')
            self.log_action("Removed all duplicate rows")
        return self.current_df
    
    # ============ OUTLIERS ============
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5):
        """Detect outliers using IQR method"""
        q1 = self.current_df[column].quantile(0.25)
        q3 = self.current_df[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = ((self.current_df[column] < lower_bound) | (self.current_df[column] > upper_bound))
        
        return {
            'count': outliers.sum(),
            'percentage': (outliers.sum() / len(self.current_df) * 100),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': self.current_df[outliers].index.tolist()
        }
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0):
        """Detect outliers using Z-score method"""
        mean = self.current_df[column].mean()
        std = self.current_df[column].std()
        
        z_scores = (self.current_df[column] - mean) / std
        outliers = z_scores.abs() > threshold
        
        return {
            'count': outliers.sum(),
            'percentage': (outliers.sum() / len(self.current_df) * 100),
            'threshold': threshold,
            'outlier_indices': self.current_df[outliers].index.tolist()
        }
    
    def remove_outliers_iqr(self, column: str, multiplier: float = 1.5):
        """Remove outliers using IQR method"""
        q1 = self.current_df[column].quantile(0.25)
        q3 = self.current_df[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        self.current_df = self.current_df[
            (self.current_df[column] >= lower_bound) & 
            (self.current_df[column] <= upper_bound)
        ]
        
        self.log_action(f"Removed outliers in {column} using IQR (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        return self.current_df
    
    def cap_outliers_iqr(self, column: str, multiplier: float = 1.5):
        """Cap outliers to IQR bounds instead of removing"""
        q1 = self.current_df[column].quantile(0.25)
        q3 = self.current_df[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        self.current_df[column] = self.current_df[column].clip(lower=lower_bound, upper=upper_bound)
        
        self.log_action(f"Capped outliers in {column} to IQR bounds: {lower_bound:.2f} - {upper_bound:.2f}")
        return self.current_df
    
    # ============ ENCODING ============
    
    def label_encode(self, columns: Optional[List[str]] = None):
        """Label encode categorical columns"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns and self.current_df[col].dtype == 'object':
                unique_vals = self.current_df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.current_df[col] = self.current_df[col].map(mapping)
        
        self.log_action(f"Label encoded columns: {columns}")
        return self.current_df
    
    def one_hot_encode(self, columns: Optional[List[str]] = None):
        """One-hot encode categorical columns"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=['object']).columns.tolist()
        
        self.current_df = pd.get_dummies(self.current_df, columns=columns, drop_first=False)
        
        self.log_action(f"One-hot encoded columns: {columns}")
        return self.current_df
    
    # ============ SCALING ============
    
    def standard_scale(self, columns: Optional[List[str]] = None):
        """Standardize columns (mean=0, std=1)"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=[numpy.number]).columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns:
                mean = self.current_df[col].mean()
                std = self.current_df[col].std()
                if std > 0:
                    self.current_df[col] = (self.current_df[col] - mean) / std
        
        self.log_action(f"Standard scaled columns: {columns}")
        return self.current_df
    
    def minmax_scale(self, columns: Optional[List[str]] = None):
        """Min-max scale columns (0-1)"""
        if columns is None:
            columns = self.current_df.select_dtypes(include=[numpy.number]).columns.tolist()
        
        for col in columns:
            if col in self.current_df.columns:
                min_val = self.current_df[col].min()
                max_val = self.current_df[col].max()
                if max_val > min_val:
                    self.current_df[col] = (self.current_df[col] - min_val) / (max_val - min_val)
        
        self.log_action(f"Min-max scaled columns: {columns}")
        return self.current_df
    
    # ============ RESET ============
    
    def reset_to_original(self):
        """Reset to original dataframe"""
        self.current_df = self.original_df.copy()
        self.cleaning_log = []
        self.log_action("Reset to original dataset")
        return self.current_df
    
    def get_cleaning_log(self) -> List[str]:
        """Get cleaning action log"""
        return self.cleaning_log


def create_cleaning_interface(df: pd.DataFrame):
    """
    Create Streamlit interface for data cleaning
    
    Args:
        df: Original DataFrame
    """
    import streamlit as st
    
    st.header("🧹 Automated Data Cleaning Center")
    st.markdown("Clean, transform, and prepare your dataset for analysis")
    
    # Initialize cleaner
    if 'data_cleaner' not in st.session_state:
        st.session_state.data_cleaner = DataCleaner(df)
    
    cleaner = st.session_state.data_cleaner
    
    # Before/After comparison
    st.subheader("📊 Before/After Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Rows", len(cleaner.original_df))
    
    with col2:
        st.metric("Cleaned Rows", len(cleaner.current_df))
    
    with col3:
        rows_removed = len(cleaner.original_df) - len(cleaner.current_df)
        st.metric("Rows Removed", rows_removed if rows_removed > 0 else 0)
    
    # Cleaning actions
    st.divider()
    st.subheader("⚙️ Cleaning Actions")
    
    # Missing Values
    with st.expander("❌ Missing Values", expanded=False):
        st.markdown("**Handle Missing Values**")
        
        missing_col = st.selectbox(
            "Select column(s) to clean:",
            cleaner.current_df.columns.tolist(),
            help="Select column with missing values"
        )
        
        action = st.radio(
            "Cleaning Action:",
            ["Remove rows", "Remove column", "Impute mean", "Impute median", "Impute mode", "Impute constant"],
            horizontal=True
        )
        
        if action == "Remove rows":
            if st.button("Remove rows with missing values"):
                cleaner.remove_rows_with_missing([missing_col])
                st.success(f"Removed rows with missing {missing_col}")
                st.rerun()
        
        elif action == "Remove column":
            if st.button("Remove column"):
                cleaner.current_df = cleaner.current_df.drop(columns=[missing_col])
                cleaner.log_action(f"Removed column: {missing_col}")
                st.success(f"Removed column {missing_col}")
                st.rerun()
        
        elif action == "Impute mean":
            if st.button("Impute with mean"):
                cleaner.impute_mean([missing_col])
                st.success(f"Imputed {missing_col} with mean")
                st.rerun()
        
        elif action == "Impute median":
            if st.button("Impute with median"):
                cleaner.impute_median([missing_col])
                st.success(f"Imputed {missing_col} with median")
                st.rerun()
        
        elif action == "Impute mode":
            if st.button("Impute with mode"):
                cleaner.impute_mode([missing_col])
                st.success(f"Imputed {missing_col} with mode")
                st.rerun()
        
        elif action == "Impute constant":
            value = st.number_input("Constant value:", value=0)
            if st.button("Impute with constant"):
                cleaner.impute_constant([missing_col], value)
                st.success(f"Imputed {missing_col} with {value}")
                st.rerun()
    
    # Duplicates
    with st.expander("🔄 Duplicates", expanded=False):
        st.markdown("**Detect and Remove Duplicates**")
        
        if st.button("Detect duplicates"):
            duplicates = cleaner.detect_duplicates()
            st.info(f"Found {duplicates['count']} duplicate rows ({duplicates['percentage']:.2f}%)")
            
            if duplicates['count'] > 0:
                st.dataframe(duplicates['rows'])
        
        if st.button("Remove all duplicates"):
            cleaner.remove_duplicates()
            st.success("Removed all duplicate rows")
            st.rerun()
    
    # Outliers
    with st.expander("🎯 Outliers", expanded=False):
        st.markdown("**Detect and Handle Outliers**")
        
        outlier_col = st.selectbox(
            "Select numeric column:",
            cleaner.current_df.select_dtypes(include=[numpy.number]).columns.tolist()
        )
        
        method = st.radio(
            "Method:",
            ["IQR Method", "Z-Score Method"],
            horizontal=True
        )
        
        if method == "IQR Method":
            multiplier = st.number_input("IQR multiplier:", value=1.5, step=0.5)
            
            if st.button("Detect outliers"):
                outliers = cleaner.detect_outliers_iqr(outlier_col, multiplier)
                st.info(f"Found {outliers['count']} outliers ({outliers['percentage']:.2f}%)")
                st.text(f"Bounds: {outliers['lower_bound']:.2f} - {outliers['upper_bound']:.2f}")
            
            if st.button("Remove outliers"):
                cleaner.remove_outliers_iqr(outlier_col, multiplier)
                st.success(f"Removed outliers in {outlier_col}")
                st.rerun()
            
            if st.button("Cap outliers"):
                cleaner.cap_outliers_iqr(outlier_col, multiplier)
                st.success(f"Capped outliers in {outlier_col}")
                st.rerun()
        
        else:  # Z-Score
            threshold = st.number_input("Z-score threshold:", value=3.0, step=0.5)
            
            if st.button("Detect outliers"):
                outliers = cleaner.detect_outliers_zscore(outlier_col, threshold)
                st.info(f"Found {outliers['count']} outliers ({outliers['percentage']:.2f}%)")
            
            if st.button("Remove outliers"):
                # Z-score outlier removal (simplified)
                mean = cleaner.current_df[outlier_col].mean()
                std = cleaner.current_df[outlier_col].std()
                z_scores = (cleaner.current_df[outlier_col] - mean) / std
                cleaner.current_df = cleaner.current_df[z_scores.abs() <= threshold]
                cleaner.log_action(f"Removed outliers in {outlier_col} using Z-score (threshold: {threshold})")
                st.success(f"Removed outliers in {outlier_col}")
                st.rerun()
    
    # Encoding
    with st.expander("🔤 Encoding", expanded=False):
        st.markdown("**Encode Categorical Variables**")
        
        encoding_method = st.radio(
            "Encoding Method:",
            ["Label Encoding", "One-Hot Encoding"],
            horizontal=True
        )
        
        if st.button(encoding_method):
            if encoding_method == "Label Encoding":
                cleaner.label_encode()
                st.success("Label encoded all categorical columns")
            else:
                cleaner.one_hot_encode()
                st.success("One-hot encoded all categorical columns")
            st.rerun()
    
    # Scaling
    with st.expander("📏 Scaling", expanded=False):
        st.markdown("**Scale Numeric Variables**")
        
        scaling_method = st.radio(
            "Scaling Method:",
            ["Standard Scale (Z-score)", "Min-Max Scale (0-1)"],
            horizontal=True
        )
        
        if st.button(scaling_method):
            if scaling_method == "Standard Scale (Z-score)":
                cleaner.standard_scale()
                st.success("Standard scaled all numeric columns")
            else:
                cleaner.minmax_scale()
                st.success("Min-max scaled all numeric columns")
            st.rerun()
    
    # Reset
    st.divider()
    if st.button("🔄 Reset to Original", type="secondary"):
        cleaner.reset_to_original()
        st.success("Reset to original dataset")
        st.rerun()
    
    # Cleaning Log
    st.divider()
    st.subheader("📋 Cleaning Log")
    
    if cleaner.get_cleaning_log():
        for action in cleaner.get_cleaning_log():
            st.info(action)
    else:
        st.info("No cleaning actions performed yet")
    
    # Download cleaned data
    st.divider()
    st.subheader("💾 Download Cleaned Dataset")
    
    # Convert to CSV
    cleaned_csv = cleaner.current_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download as CSV",
        data=cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Convert to Excel
    import io
    buffer = io.BytesIO()
    cleaner.current_df.to_excel(buffer, index=False)
    cleaned_excel = buffer.getvalue()
    st.download_button(
        label="📥 Download as Excel",
        data=cleaned_excel,
        file_name="cleaned_data.xlsx",
        mime="application/vnd.ms-excel",
        use_container_width=True
    )
    
    # Show cleaned data preview
    st.divider()
    st.subheader("🔍 Cleaned Dataset Preview")
    st.dataframe(
        cleaner.current_df.head(20),
        use_container_width=True,
        hide_index=True
    )