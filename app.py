"""
InsightForge AI - AI-Powered Exploratory Data Analysis Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as numpy
import plotly.express as px
from ai.chat_engine import ChatEngine
from analytics.quality_score import calculate_dataset_health_score
from ai.recommendations import analyze_data_patterns, generate_simple_recommendations
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Import utilities
from utils.data_utils import (
    get_dataset_summary,
    get_missing_value_analysis,
    get_correlation_matrix,
    get_outliers
)
from utils.visualization import (
    create_histogram,
    create_boxplot,
    create_correlation_heatmap,
    create_scatter_plot,
    create_bar_chart
)
from ai.gemini_client import generate_business_insights, generate_executive_summary

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="InsightForge AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============ HELPER FUNCTIONS ============

@st.cache_data
def load_file(file):
    """Load uploaded file with caching"""
    file_ext = file.name.rsplit(".", 1)[-1].lower()
    
    try:
        if file_ext == "csv":
            df = pd.read_csv(file)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(file)
        elif file_ext == "parquet":
            df = pd.read_parquet(file)
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


# ============ MAIN APP ============

def main():
    # Header
    st.markdown('<div class="main-header">🚀 InsightForge AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Exploratory Data Analysis Platform</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # App mode
        mode = st.radio(
            "Analysis Mode:",
            ["Quick EDA", "Deep Analysis", "AI Insights", "AI Data Chat", "Data Cleaning"],
            help="Choose analysis depth"
        )
        
        st.divider()
        
        # About
        st.markdown("**About InsightForge AI**")
        st.info("""
        📊 Upload datasets and get instant insights
        
        🤖 AI-powered business recommendations
        
        📈 Interactive visualizations
        
        ⚡ Fast exploratory data analysis
        """)
    
    # Main content
    if mode == "Quick EDA":
        quick_eda_mode()
    elif mode == "Deep Analysis":
        deep_analysis_mode()
    elif mode=="AI Insights":
        ai_insights_mode()
    elif mode=="AI Data Chat":
        ai_data_chat_mode()
    elif mode=="Data Cleaning":
        data_cleaning_mode()



def quick_eda_mode():
    """Quick EDA mode - Fast dataset overview"""
    
    # Section 1: File Upload
    st.header("📤 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Parquet file",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supports CSV, Excel (XLS/XLSX), and Parquet files"
    )
    
    if uploaded_file is not None:
        # Load file
        df = load_file(uploaded_file)
        
        if df is None:
            return
        
        # Display success message
        st.success(f"✅ Successfully loaded: {uploaded_file.name}")
        
        # Section 2: Dataset Summary
        st.header("📊 Dataset Summary")
        
        summary = get_dataset_summary(df)
        
        # Display summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Rows", summary['rows'])
        
        with col2:
            st.metric("📋 Total Columns", summary['columns'])
        
        with col3:
            total_missing = sum(summary['missing_values'].values())
            st.metric("❌ Missing Values", total_missing)
        
        with col4:
            st.metric("🔄 Duplicate Rows", summary['duplicate_rows'])
        
        st.divider()
        
        # Column details
        st.subheader("📑 Column Information")
        
        col_info_df = pd.DataFrame({
            'Column': summary['column_names'],
            'Type': [summary['column_types'][col] for col in summary['column_names']],
            'Missing': [summary['missing_values'][col] for col in summary['column_names']],
            'Missing %': [summary['missing_percentage'][col] for col in summary['column_names']]
        })
        
        st.dataframe(
            col_info_df,
            use_container_width=True,
            hide_index=True,
        )
        
        st.divider()
        
        # Section 3: Missing Value Analysis
        st.header("❌ Missing Value Analysis")
        
        missing_analysis = get_missing_value_analysis(df)
        
        if missing_analysis['columns_with_missing'] > 0:
            st.warning(f"⚠️ {missing_analysis['columns_with_missing']} columns have missing values")
            
            # Missing values bar chart
            missing_df = pd.DataFrame({
                'Column': list(missing_analysis['details'].keys()),
                'Missing Count': [v['missing_count'] for v in missing_analysis['details'].values()],
                'Missing %': [v['missing_percentage'] for v in missing_analysis['details'].values()]
            })
            
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing %',
                title='Missing Values by Column (%)',
                color='Missing %',
                color_continuous_scale='Reds',
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed missing values
            for col, details in missing_analysis['details'].items():
                st.text(f"• **{col}**: {details['missing_count']} missing ({details['missing_percentage']}%)")
        else:
            st.info("✅ No missing values found! Great data quality.")
        
        st.divider()
        
        # Section 4: Data Preview
        st.header("🔍 Data Preview")
        
        st.subheader("First 10 Rows")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )
        
        st.subheader("Last 10 Rows")
        st.dataframe(
            df.tail(10),
            use_container_width=True,
            hide_index=True,
        )
        
    else:
        # No file uploaded - show demo
        st.info("👆 Upload a dataset to get started with Quick EDA")
        
        # Show sample data
        st.subheader("📋 Sample Dataset (for preview)")
        
        sample_df = pd.DataFrame({
            'Product': ['A', 'B', 'C', 'D', 'E'],
            'Category': ['Electronics', 'Accessories', 'Electronics', 'Services', 'Accessories'],
            'Sales': [150, 75, 200, 300, 90],
            'Quantity': [5, 3, 4, 10, 2],
            'Price': [30, 25, 50, 30, 45]
        })
        
        st.dataframe(sample_df, use_container_width=True)


def deep_analysis_mode():
    """Deep Analysis mode - Full EDA with visualizations"""
    
    st.header("📤 Upload Dataset for Deep Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Parquet file",
        type=["csv", "xlsx", "xls", "parquet"],
    )
    
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        
        if df is None:
            return
        
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Dataset summary
        summary = get_dataset_summary(df)
        
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", summary['rows'])
        
        with col2:
            st.metric("Columns", summary['columns'])
        
        with col3:
            st.metric("Missing Values", sum(summary['missing_values'].values()))
        
        with col4:
            st.metric("Duplicates", summary['duplicate_rows'])
        
        st.divider()
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[numpy.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.header("📈 Numeric Columns Analysis")
            
            # Column selector
            selected_col = st.selectbox(
                "Select column to analyze:",
                numeric_cols,
                help="Choose a numeric column for detailed analysis"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                st.subheader("Histogram")
                fig_hist = create_histogram(df, selected_col)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Boxplot
                st.subheader("Boxplot")
                fig_box = create_boxplot(df, selected_col)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistics
            st.subheader("📊 Statistical Summary")
            
            stats = summary['numeric_stats'][selected_col]
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
                'Value': [
                    stats['mean'],
                    stats['median'],
                    stats['std'],
                    stats['min'],
                    stats['max'],
                    stats['q25'],
                    stats['q75']
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Outliers
            outliers = get_outliers(df, selected_col)
            if len(outliers) > 0:
                st.warning(f"⚠️ Found {len(outliers)} outliers in {selected_col}")
        
        st.divider()
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            st.header("📊 Categorical Columns Analysis")
            
            selected_cat_col = st.selectbox(
                "Select categorical column:",
                categorical_cols,
            )
            
            st.subheader(f"Distribution of {selected_cat_col}")
            fig_bar = create_bar_chart(df, selected_cat_col)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            st.header("🔗 Correlation Analysis")
            
            correlation = get_correlation_matrix(df)
            
            if correlation is not None:
                st.subheader("Correlation Heatmap")
                fig_corr = create_correlation_heatmap(correlation)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Top correlations
                st.subheader("Top 5 Strongest Correlations")
                
                corr_pairs = []
                for i in range(len(correlation.columns)):
                    for j in range(i + 1, len(correlation.columns)):
                        corr_pairs.append({
                            'Column 1': correlation.columns[i],
                            'Column 2': correlation.columns[j],
                            'Correlation': correlation.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values('Correlation', key=lambda x: x.abs(), ascending=False)
                
                st.dataframe(
                    corr_df.head(5),
                    use_container_width=True,
                    hide_index=True,
                )
        
        st.divider()
        
        # Scatter plots
        if len(numeric_cols) >= 2:
            st.header("🔗 Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X Axis", numeric_cols, index=0)
            
            with col2:
                y_col = st.selectbox("Y Axis", numeric_cols, index=1)
            
            fig_scatter = create_scatter_plot(df, x_col, y_col)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.info("👆 Upload a dataset to start Deep Analysis")


def ai_insights_mode():
    """AI Insights mode - Gemini-powered business insights"""
    
    st.header("📤 Upload Dataset for AI Insights")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Parquet file",
        type=["csv", "xlsx", "xls", "parquet"],
    )
    
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        
        if df is None:
            return
        
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Get dataset summary
        summary = get_dataset_summary(df)
        
        st.header("🤖 AI-Generated Insights")
        
        # Generate insights button
        if st.button("🚀 Generate Business Insights", type="primary"):
            with st.spinner("🤖 Analyzing dataset with Gemini AI..."):
                try:
                    insights = generate_business_insights(df, summary)
                    from reports.pdf_generator import create_pdf_report
                    from analytics.quality_score import calculate_dataset_health_score
                    from utils.data_utils import get_correlation_matrix
                    from ai.recommendations import analyze_data_patterns, generate_ai_recommendations
                    
                    # Calculate quality score
                    quality_result = calculate_dataset_health_score(df)
                    
                    # Get recommendations
                    patterns = analyze_data_patterns(df)
                    recommendations = generate_ai_recommendations(df, patterns)
                    
                    # Get correlation matrix
                    correlation = get_correlation_matrix(df)
                    
                    # Generate PDF
                    pdf_filename = create_pdf_report(
                        df=df,
                        summary=summary,
                        quality_result=quality_result,
                        recommendations=recommendations,
                        ai_insights=insights,
                        executive_summary=summary_text,
                        correlation_matrix=correlation,
                        filename="EDA_Report.pdf"
                    )
                    
                    # Provide download
                    with open(pdf_filename, "rb") as f:
                        pdf_data = f.read()
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name="EDA_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully!")
                    
                    st.subheader("💡 Key Business Insights")
                    st.markdown(insights)
                    
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
        
        st.divider()
        
        # Executive summary
        if st.button("📝 Generate Executive Summary", type="primary"):
            with st.spinner("🤖 Creating executive summary..."):
                try:
                    summary_text = generate_executive_summary(df, summary)
                    
                    st.subheader("📋 Executive Summary")
                    st.markdown(summary_text)
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        
        st.divider()
        
        # Dataset preview
        st.subheader("🔍 Dataset Preview (for reference)")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )
        
    else:
        st.info("👆 Upload a dataset to get AI-powered insights")


def ai_data_chat_mode():
    """AI Data Chat mode - Natural language dataset querying"""
    import streamlit as st
    from ai.chat_engine import create_chat_interface, SAMPLE_QUESTIONS
    
    st.header("📤 Upload Dataset for Chat")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Parquet file",
        type=["csv", "xlsx", "xls", "parquet"],
    )
    
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        
        if df is None:
            return
        
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Show dataset preview
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Sample questions
        st.subheader("💡 Try These Sample Questions:")
        sample_cols = st.columns(2)
        
        for i, question in enumerate(SAMPLE_QUESTIONS):
            col = sample_cols[i % 2]
            if col.button(question, use_container_width=True):
                # Trigger chat with this question
                st.session_state.sample_question = question
        
        # Handle sample question click
        if 'sample_question' in st.session_state:
            question = st.session_state.sample_question
            from ai.chat_engine import ChatEngine
            
            if 'chat_engine' not in st.session_state:
                st.session_state.chat_engine = ChatEngine()
            
            chat_engine = st.session_state.chat_engine
            
            with st.spinner("🤖 Analyzing dataset..."):
                response = chat_engine.generate_chat_response(question, df)
                st.session_state.latest_response = response
            
            del st.session_state.sample_question
        
        st.divider()
        
        # Chat interface
        create_chat_interface(df)
        
    else:
        st.info("👆 Upload a dataset to start AI chat")

def data_cleaning_mode():
    """Data Cleaning Studio mode"""
    from utils.cleaning import create_cleaning_interface
    
    st.header("📤 Upload Dataset for Cleaning")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Parquet file",
        type=["csv", "xlsx", "xls", "parquet"],
    )
    
    if uploaded_file is not None:
        df = load_file(uploaded_file)
        
        if df is None:
            return
        
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Show original preview
        st.subheader("📊 Original Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Cleaning interface
        create_cleaning_interface(df)
        
    else:
        st.info("👆 Upload a dataset to start cleaning")


if __name__ == "__main__":
    main()