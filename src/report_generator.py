from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import base64
from io import BytesIO


def generate_statistics_pdf(stats: dict, video_filename: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("⚽ Soccer Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report info
    info_style = styles['Normal']
    story.append(Paragraph(f"<b>Video File:</b> {video_filename}", info_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Statistics section
    story.append(Paragraph("Video Statistics", heading_style))
    
    # Create statistics table
    stats_data = [
        ['Metric', 'Value'],
        ['Total Frames', str(stats.get('total_frames', 'N/A'))],
        ['Frames with Detection', str(stats.get('frames_with_detection', 'N/A'))],
        ['Detection Rate', f"{stats.get('detection_percentage', 0):.2f}%"],
        ['Average Ball Speed', f"{stats.get('average_ball_speed', 0):.2f} px/frame"],
        ['Maximum Ball Speed', f"{stats.get('max_ball_speed', 0):.2f} px/frame"],
        ['Average Confidence', f"{stats.get('average_confidence', 0):.2f}%"],
        ['Total Detections', str(stats.get('total_detections', 'N/A'))],
        ['Processing Time', f"{stats.get('processing_time', 0):.2f} seconds"],
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4ff')]),
    ]))
    
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Heatmap section
    if stats.get('heatmap_image'):
        story.append(Paragraph("Ball Position Heatmap", heading_style))
        
        try:
            # Extract base64 image data
            image_data = stats['heatmap_image']
            if image_data.startswith('data:image/png;base64,'):
                image_base64 = image_data.replace('data:image/png;base64,', '')
                image_bytes = base64.b64decode(image_base64)
                
                # Create temporary image file
                img_buffer = BytesIO(image_bytes)
                img = Image(img_buffer, width=5.5*inch, height=5.5*inch)
                story.append(img)
        except Exception as e:
            story.append(Paragraph(f"<i>Unable to include heatmap image: {str(e)}</i>", info_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
