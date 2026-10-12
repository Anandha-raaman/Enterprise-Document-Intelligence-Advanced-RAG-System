import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_DIR = Path(__file__).resolve().parent

def create_employee_handbook():
    pdf_path = OUTPUT_DIR / "TechCorp_Employee_Handbook_2026.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        spaceAfter=4
    )

    story = []
    
    # Page 1
    story.append(Paragraph("TechCorp Global Employee Handbook (2026 Edition)", title_style))
    story.append(Paragraph("Operational Guidelines, Benefits Framework, and Workplace Policies", ParagraphStyle('Sub', parent=body_style, fontName='Helvetica-Oblique', fontSize=10, textColor=colors.HexColor('#64748b'))))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Leave and Absence Policy", h1_style))
    story.append(Paragraph("TechCorp provides competitive leave entitlements to support employee work-life balance and overall well-being. All full-time permanent employees are eligible for the following leave categories:", body_style))
    story.append(Paragraph("&bull; <b>Annual Leave:</b> Full-time employees accrue 20 days of paid annual leave per calendar year. Annual leave must be requested at least 7 business days in advance through the HR Portal. A maximum of 5 unused days may be carried over into the first quarter of the subsequent year.", bullet_style))
    story.append(Paragraph("&bull; <b>Sick Leave:</b> Employees receive 10 days of paid sick leave annually. For consecutive absences exceeding 3 business days, a verified medical certificate signed by a licensed physician must be submitted upon return.", bullet_style))
    story.append(Paragraph("&bull; <b>Casual Leave:</b> 6 days of paid casual leave are allocated per calendar year for unforeseen personal matters. Unused casual leaves expire on December 31 and cannot be encashed.", bullet_style))
    story.append(Paragraph("&bull; <b>Parental Leave:</b> Primary caregivers receive 16 weeks of fully paid parental leave, while secondary caregivers receive 4 weeks of fully paid leave.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Health and Wellness Allowance Framework", h1_style))
    story.append(Paragraph("The organization actively fosters physical vitality and mental health across all departments:", body_style))
    story.append(Paragraph("&bull; <b>Gym and Fitness Reimbursement:</b> Employees can claim up to $60 per month towards gym memberships, yoga studios, or athletic equipment. Original itemized receipts must be submitted before the 25th of each month.", bullet_style))
    story.append(Paragraph("&bull; <b>Mental Health Support:</b> Up to 5 complimentary, confidential counseling sessions per year are provided via the MindCare telehealth portal.", bullet_style))
    story.append(Paragraph("&bull; <b>Ergonomic Home Office Stipend:</b> A one-time allowance of $350 is granted to remote workers for certified ergonomic chairs, standing desks, or monitor arms.", bullet_style))

    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. Travel and Expense Reimbursement Policy", h1_style))
    story.append(Paragraph("All business-related domestic and international travel must comply with TechCorp expenditure regulations:", body_style))
    story.append(Paragraph("&bull; <b>Commercial Flights:</b> Domestic flights must be booked in Economy Class at least 14 days in advance through the corporate travel portal. International flights exceeding 7 continuous flight hours qualify for Premium Economy.", bullet_style))
    story.append(Paragraph("&bull; <b>Hotel Lodging Caps:</b> Nightly room rates (excluding local taxes) are capped at $180 per night for Tier 1 metropolitan cities (e.g., New York, San Francisco, London, Tokyo) and $120 per night for Tier 2 cities.", bullet_style))
    story.append(Paragraph("&bull; <b>Daily Meal Allowance (Per Diem):</b> Up to $75 per business travel day is reimbursed for meals and incidental expenses. Receipts are required for all individual expenses exceeding $25.", bullet_style))
    story.append(Paragraph("&bull; <b>Ground Transportation:</b> Ride-sharing services (Uber, Lyft) or public rail transit are preferred. Rental vehicles require prior written manager approval.", bullet_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Information Security, Remote Access, and Data Protection", h1_style))
    story.append(Paragraph("Protecting corporate assets, customer proprietary data, and internal systems is an uncompromising obligation:", body_style))
    story.append(Paragraph("&bull; <b>Remote Access Authentication:</b> Multi-Factor Authentication (MFA) via a hardware security key or corporate authenticator app is strictly mandatory for all endpoints connecting to corporate VPN or cloud environments.", bullet_style))
    story.append(Paragraph("&bull; <b>Password Complexity:</b> Passwords must be at least 12 characters in length, containing uppercase letters, numbers, and symbols, and must be rotated every 90 days. Password reuse is prohibited for 6 consecutive cycles.", bullet_style))
    story.append(Paragraph("&bull; <b>Clean Desk and Screen Lock Policy:</b> Unattended workstations must automatically lock after 5 minutes of inactivity. Physical documents containing sensitive data must be securely locked away overnight.", bullet_style))
    story.append(Paragraph("&bull; <b>Removable Storage Media:</b> The connection of unauthorized USB flash drives or external hard drives to corporate laptops is strictly disabled via centralized endpoint device management.", bullet_style))

    doc.build(story)
    print(f"Created {pdf_path}")

def create_cloud_sla_agreement():
    pdf_path = OUTPUT_DIR / "NovaCloud_Enterprise_Architecture_SLA.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0284c7'),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # Page 1
    story.append(Paragraph("NovaCloud Enterprise Architecture & High-Availability SLA (2026)", title_style))
    story.append(Paragraph("Master Infrastructure Agreement, Availability Targets, and Operational Protocol", ParagraphStyle('Sub', parent=body_style, fontName='Helvetica-Oblique', fontSize=10, textColor=colors.HexColor('#64748b'))))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Service Level Objectives (SLO) & Uptime Commitments", h1_style))
    story.append(Paragraph("NovaCloud guarantees enterprise-tier infrastructure availability and API latency standards across multi-region production clusters:", body_style))
    story.append(Paragraph("&bull; <b>Monthly Uptime Commitment:</b> NovaCloud commits to a 99.99% monthly availability percentage across all core API endpoints, database clusters, and container runtime nodes.", bullet_style))
    story.append(Paragraph("&bull; <b>API Latency Standard:</b> 99th percentile (p99) API response latency must remain strictly below 25 milliseconds under standard operating load of up to 50,000 requests per second.", bullet_style))
    story.append(Paragraph("&bull; <b>Scheduled Maintenance Windows:</b> Maintenance operations are restricted to Sundays between 02:00 AM and 04:00 AM UTC with a mandatory 72-hour advance advisory notice.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Disaster Recovery, Replication, and Resiliency", h1_style))
    story.append(Paragraph("Our distributed architecture provides automated failover and data resilience across geographically isolated availability zones:", body_style))
    story.append(Paragraph("&bull; <b>Cross-Region Replication:</b> Database mutations are replicated asynchronously across primary and secondary regions using distributed consensus with a replication lag below 300ms.", bullet_style))
    story.append(Paragraph("&bull; <b>Recovery Point Objective (RPO):</b> NovaCloud guarantees an RPO of less than 5 minutes for stateful transactional data.", bullet_style))
    story.append(Paragraph("&bull; <b>Recovery Time Objective (RTO):</b> Guaranteed automated cluster failover RTO of less than 15 minutes during total regional failure.", bullet_style))

    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("3. Cryptographic Controls and Data Protection at Rest and Transit", h1_style))
    story.append(Paragraph("Enterprise tenant data is insulated with defense-in-depth cryptographic controls:", body_style))
    story.append(Paragraph("&bull; <b>Encryption at Rest:</b> All block storage, object storage buckets, and relational table snapshots are encrypted using AES-256 GCM.", bullet_style))
    story.append(Paragraph("&bull; <b>Customer Managed Encryption Keys (CMEK):</b> Customers may integrate external Cloud KMS with mandatory cryptographic key rotation every 180 days.", bullet_style))
    story.append(Paragraph("&bull; <b>Transport Layer Security:</b> All incoming and internal inter-service network ingress mandates TLS version 1.3 with forward secrecy ciphers.", bullet_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Incident Severity Classification & SLA Financial Credits", h1_style))
    story.append(Paragraph("In the event of an unscheduled degradation, NovaCloud applies structured incident escalation and financial penalty offsets:", body_style))
    story.append(Paragraph("&bull; <b>Severity 1 (Critical Outage):</b> Total unavailability of core API or database service. Dedicated engineering incident commander responds within 15 minutes 24/7/365.", bullet_style))
    story.append(Paragraph("&bull; <b>Severity 2 (High Degradation):</b> Substantial latency elevation or loss of redundancy. Response SLA is within 45 minutes.", bullet_style))
    story.append(Paragraph("&bull; <b>SLA Remedy Credits:</b> If monthly uptime falls between 99.0% and 99.98%, customer receives a 10% credit. If uptime drops below 99.0%, customer receives a 25% credit applied to next billing invoice.", bullet_style))

    doc.build(story)
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    create_employee_handbook()
    create_cloud_sla_agreement()
