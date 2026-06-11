"""
Professional PDF Report Generator for EDA
"""
import pandas as pd
import numpy as numpy
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import (
    Document, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import Color, black, white
from reportlab.pdfgen import canvas
import plotly
from io import BytesIO
from datetime import datetime
from typing import Optional


def create_pdf_report(
    df: pd.DataFrame,
    summary: dict,
    quality_result: dict,
    recommendations: list,
    ai_insights: str,
    executive_summary: str,
    correlation_matrix: Optional[numpy.matrix] = None,
    histograms: Optional[list] = None,
    filename: str = "EDA_Report.pdf"
):
    """
    Generate professional PDF report
    
    Args:
        df: DataFrame
        summary: Dataset summary
        quality_result: Quality score result
        recommendations: List of recommendations
        ai_insights: AI-generated business insights
        executive_summary: Executive summary
        correlation_matrix: Correlation matrix (optional)
        histograms: List of histogram Plotly figures (optional)
        filename: Output filename
    """
    
    # Create temporary directory for plot images
    import tempfile
    import os
    
    pdf = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='Title',
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=Color(0.1, 0.4, 0.8),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        name='Heading',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=Color(0.1, 0.4, 0.8),
        spaceAfter=12,
        spaceBefore=12
    )
    
    content_style = ParagraphStyle(
        name='Content',
        fontName='Helvetica',
        fontSize=11,
        spaceAfter=6
    )
    
    story = []
    
    # ============ COVER PAGE ============
    cover_width, cover_height = letter
    
    c = canvas.Canvas(filename)
    c.setSceneBox(cover_width, cover_height)
    
    # Background
    c.setFillColor(Color(0.95, 0.95, 1))
    c.rect(0, 0, cover_width, cover_height, fill=1)
    
    # Title
    c.setFont("Helvetica-Bold", 36)
    c.setFillColor(Color(0.1, 0.4, 0.8))
    c.drawCentredString(cover_width/2, cover_height - 3*inch, "InsightForge AI")
    
    c.setFont("Helvetica", 24)
    c.setFillColor(black)
    c.drawCentredString(cover_width/2, cover_height - 3.5*inch, "Exploratory Data Analysis Report")
    
    c.setFont("Helvetica", 14)
    c.setFillColor(Color(0.3, 0.3, 0.3))
    c.drawCentredString(cover_width/2, cover_height - 5*inch, f"Dataset: {filename.replace('.pdf', '').replace('_EDI_Report', '')}")
    c.drawCentredString(cover_width/2, cover_height - 5.5*inch, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    
    c.drawCentredString(cover_width/2, cover_height - 7*inch, "📊")
    
    c.save()
    
    # Add cover to story
    story.append(Paragraph("InsightForge AI", title_style))
    story.append(Paragraph("Exploratory Data Analysis Report", heading_style))
    story.append(Spacer(1, 2*inch))
    
    # ============ EXECUTIVE SUMMARY ============
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(executive_summary, content_style))
    story.append(Spacer(1, 0.5*inch))
    
    # ============ DATASET OVERVIEW ============
    story.append(Paragraph("Dataset Overview", heading_style))
    
    # Metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Rows', str(summary['rows'])],
        ['Total Columns', str(summary['columns'])],
        ['Missing Values', str(sum(summary['missing_values'].values()))],
        ['Duplicate Rows', str(summary['duplicate_rows'])],
    ]
    
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Color(0.1, 0.4, 0.8)),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), Color(0.95, 0.95, 1)),
        ('GRID', (0, 0), (-1, -1), 1, black),
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.5*inch))
    
    # ============ DATASET QUALITY SCORE ============
    story.append(Paragraph("Dataset Quality Score", heading_style))
    
    score = quality_result['score']
    if score >= 80:
        score_color = Color(0, 1, 0)
    elif score >= 60:
        score_color = Color(1, 0.5, 0)
    else:
        score_color = Color(1, 0, 0)
    
    
    story.append(Paragraph(f"Quality Score: {score}/100", content_style))
    story.append(Spacer(1, 0.3*inch))
    
    if quality_result['strengths']:
        story.append(Paragraph("Strengths:", content_style))
        for strength in quality_result['strengths']:
            story.append(Paragraph(f"• {strength}", content_style))
    
    if quality_result['weaknesses']:
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Weaknesses:", content_style))
        for weakness in quality_result['weaknesses']:
            story.append(Paragraph(f"• {weakness}", content_style))
    
    story.append(Spacer(1, 0.5*inch))
    
    # ============ AI INSIGHTS ============
    story.append(Paragraph("AI-Generated Business Insights", heading_style))
    story.append(Paragraph(ai_insights, content_style))
    story.append(Spacer(1, 0.5*inch))
    
    # ============ RECOMMENDATIONS ============
    story.append(Paragraph("Smart Recommendations", heading_style))
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", content_style))
    
    story.append(Spacer(1, 0.5*inch))
    
    # ============ COLUMN DETAILS ============
    story.append(Paragraph("Column Information", heading_style))
    
    col_data = [['Column', 'Type', 'Missing', 'Missing %']]
    
    for col in summary['column_names']:
        col_data.append([
            col,
            summary['column_types'][col],
            str(summary['missing_values'][col]),
            f"{summary['missing_percentage'][col]}%"
        ])
    
    col_table = Table(col_data)
    col_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Color(0.1, 0.4, 0.8)),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), Color(0.95, 0.95, 1)),
        ('GRID', (0, 0), (-1, -1), 1, black),
    ]))
    
    story.append(col_table)
    story.append(Spacer(1, 0.5*inch))
    
    # ============ BUILD DOCUMENT ============
    pdf.build(story)
    
    return filename


def convert_plotly_to_image(fig, width=8, height=6):
    """Convert Plotly figure to image for PDF"""
    img_bytes = BytesIO()
    plotly.io.write_image(fig, img_bytes, format='png', width=width*100, height=height*100)
    img_bytes.seek(0)
    return img_bytes