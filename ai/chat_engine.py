import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
from typing import List, Dict, Optional
import json
import numpy as numpy

load_dotenv()


class ChatEngine:
    def __init__(self):
        api_key=os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=api_key)
        self.model=genai.GenerativeModel('gemini-flash-latest')
        self.chat_history: List[Dict]=[]

    def _prepare_dataframe_context(self, df: pd.DataFrame, sample_rows: int=5)->str:
        columns_info=[]
        for col in df.columns:
            col_type=str(df[col].dtype)
            missing_count=df[col].isnull().sum()
            missing_pct=(missing_count/len(df)*100).round(2)
            columns_info.append({
                'name': col,
                'type': col_type,
                'missing_count': int(missing_count),
                'missing_pct':float(missing_pct)
            })


        numeric_cols=df.select_dtypes(include=['number']).columns
        numeric_stats={}
        for col in numeric_cols:
            numeric_stats[col]={
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }


        categorical_cols=df.select_dtypes(include=['object']).columns
        categorical_stats={}
        for col in categorical_cols:
            categorical_stats[col]= df[col].value_counts().head(5).to_dict()


        def convert_to_native(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (numpy.integer, numpy.int64, numpy.int32)):
                return int(obj)
            elif isinstance(obj, (numpy.floating, numpy.float64, numpy.float32)):
                return float(obj)
            elif isinstance(obj, numpy.ndarray):
                return obj.tolist()
            else:
                return obj
        context = f"""
Dataset: {len(df)} rows × {len(df.columns)} columns

Columns:
{json.dumps(convert_to_native(columns_info), indent=2)}

Numeric Statistics:
{json.dumps(convert_to_native(numeric_stats), indent=2)}

Categorical Value Counts (top 5):
{json.dumps(convert_to_native(categorical_stats), indent=2)}

Sample Data (first {sample_rows} rows):
{df.head(sample_rows).to_json()}
"""
        return context
    

    def generate_chat_response(
            self,
            question: str,
            df: pd.DataFrame,
            previous_context: Optional[str]=None
    )->str:
        """
        Generate AI response to natural language question about dataset

        Args:
            question: User's natural language question
            df: DataFrame to query
            previous_context: Previous conversation context

        Returns:
            AI-generated response string
        """



        current_context=self._prepare_dataframe_context(df)

        if previous_context:
            full_prompt=f"""
Your are a data analyst chat assistant. The user is asking questions about a dataset.

DATASET CONTEXT:
{current_context}

PREVIOUS CONVERSATION:
{previous_context}

USER QUESTION: {question}

Please provide a clear, concise, and actionable answer. If the question requires data analysis:
1. Identify the relevant columns
2. Perform the analysis mentally
3. Provide specific numbers and insights
4. Give business recommendations if applicable

Format your response with:
- Clear answer header
- Supporting data/evidence
- Business implications
- Recommendations (if applicable)
"""
            
        else:
            full_prompt=f"""
You are a data analyst chat assistant. The user is asking questions about a dataset.

DATASET CONTEXT:
{current_context}

USER QUESTION: {question}

Please provide a clear, concise, and actionable answer. If the question requires data analysis:
1. Identify the relevant columns
2. Perform the analysis mentally
3. Provide specific numbers and insights
4. Give business recommendations if applicable

Format your response with:
- Clear answer header
- Supporting data/evidence
- Business implications
- Recommendations (if applicable)
"""


        try:
            response=self.model.generate_content(full_prompt)
            answer=response.text
            self.chat_history.append({
                'user': question,
                'assistant': answer
            })

            return answer
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def get_chat_history(self)->List[Dict]:
        return self.chat_history
    
    def clear_chat_history(self):
        self.chat_history=[]

def create_chat_interface(df: pd.DataFrame):
    """
    Create Streamlit chat interface for dataset
    
    Args:
        df: DataFrame to chat with
    """
    import streamlit as st
    
    st.header("🤖 AI Data Chat")
    st.markdown("Ask natural language questions about your dataset")
    
    # Initialize chat engine
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = ChatEngine()
    
    chat_engine = st.session_state.chat_engine
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        chat_engine.clear_chat_history()
        st.rerun()
    
    # Display chat history
    for message in chat_engine.get_chat_history():
        with st.chat_message("user"):
            st.write(message['user'])
        
        with st.chat_message("assistant"):
            st.markdown(message['assistant'])
    
    # Chat input
    user_question = st.chat_input("Ask a question about your dataset...")
    
    if user_question:
        with st.spinner("🤖 Analyzing dataset..."):
            # Get previous context (last 2 messages)
            previous_context = None
            if len(chat_engine.get_chat_history()) >= 2:
                prev_messages = chat_engine.get_chat_history()[-2:]
                previous_context = "\n\n".join([
                    f"{m['user']}: {m['assistant']}" for m in prev_messages
                ])
            
            # Generate response
            response = chat_engine.generate_chat_response(
                user_question, 
                df,
                previous_context
            )
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(response)


# Example usage questions
SAMPLE_QUESTIONS = [
    "Which category generated the most revenue?",
    "Show columns with highest missing values.",
    "What trends do you see in the data?",
    "What business recommendations can you provide?",
    "What is the correlation between numeric columns?",
    "Are there any outliers in the data?",
    "Which region has the highest sales?",
    "What is the average value by category?"
]







