import pandas as pd
import numpy as numpy
import plotly.express as px
import plotly.graph_objects as go

def create_histogram(df, column, color=None):
    fig=px.histogram(
        df,
        x=column,
        color=color,
        title=f'Distribution of {column}',
        labels={column:column},
        
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title=column,
        yaxis_title='Count',
        height=500,
    )
    return fig


def create_boxplot(df, column):
    fig=px.box(
       df,
       x=column,
       title=f'Boxplot of {column}',
       labels={column: column}, 
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title=column,
        yaxis_title='Value',
        height=500,
    )
    return fig

def create_correlation_heatmap(correlation_matrix):
    fig=px.imshow(
        correlation_matrix,
        text_auto='.2f',
        title='Correlation Heatmap',
        color_continuous_scale='RdBu',
    )
    fig.update_layout(
        template='plotly_white',
        height=600,
    )
    return fig

def create_scatter_plot(df, x_column, y_column, color=None):
    fig=px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color,
        title=f'{y_column} vs {x_column}',
        labels={x_column: x_column, y_column: y_column},
        size_max=10,
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_title=x_column,
        yaxis_title=y_column,
        height=500,
    )
    return fig

def create_bar_chart(df, column):
    counts=df[column].value_counts().reset_index()
    counts.columns=[column, 'count']

    fig=px.bar(
        counts,
        x=column,
        y='count',
        title=f'Distribution of {column}',
        labels={column: column, 'count': 'Count'},
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_title=column,
        yaxis_title='Count',
        height=500,
    )
    return fig

