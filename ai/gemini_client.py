import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

def get_gemini_client():
    api_key=os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-flash-latest')


def generate_business_insights(df, summary):
    client=get_gemini_client()
    data_description=f"""
Dataset Summary:
- Rows: {summary['rows']}
- Columns: {summary['columns']}
- Column Names: {summary['column_names']}
- Column Types: {summary['column_types']}
- Missing Values: {summary['missing_values']}
- Numeric Statistics: {summary.get('numeric_stats', {})}
"""
    

    prompt=f"""
You are a data analyst. Analyze this dataset and provide 5-7 key business insights.
{data_description}
Please provide:
1. Key trends and patterns
2. Potential business opportunities
3. Risk factors or concerns
4. Recommendations for action
5. Any surprising findings

Format your response as a bulleted list with clear, actionable insights.
"""
    
    response=client.generate_content(prompt)
    return response.text

def generate_executive_summary(df, summary):
    client=get_gemini_client()
    data_description=f"""
Dataset Overview:
- Size: {summary['rows']} rows x {summary['columns']} columns
- Columns: {summary['column_names']}
- Data Quality: {summary['missing_values']} missing values total
- Duplicate Rows: {summary['duplicate_rows']}
"""
    
    prompt=f"""
You are a business analyst. Create a concise executive summary (150-200 words) for this dataset.
{data_description}

Include:
- What this dataset represents
- Key characteristics and quality
- Main insights at a high level
- Business implications

Write in professional, executive-friendly language.
"""
    
    response=client.generate_content(prompt)
    return response.text



