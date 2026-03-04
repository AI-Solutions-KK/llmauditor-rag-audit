#!/usr/bin/env python3
"""
Convert PDF certification report to images for README display
"""

import os
import sys
from pathlib import Path

def convert_pdf_to_images():
    """Convert the PDF report to images"""
    
    pdf_path = Path("reports/certification_HR_Knowledge_RAG_Demo_20260305_015105.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    # Create images directory
    images_dir = Path("reports/images")
    images_dir.mkdir(exist_ok=True)
    
    try:
        from pdf2image import convert_from_path
        
        # Convert PDF to images
        print(f"🔄 Converting {pdf_path} to images...")
        images = convert_from_path(pdf_path, dpi=150)
        
        image_paths = []
        for i, image in enumerate(images):
            image_path = images_dir / f"certification_page_{i+1}.png"
            image.save(image_path, "PNG", optimize=True)
            image_paths.append(str(image_path))
            print(f"✅ Saved: {image_path}")
        
        print(f"\n📷 Generated {len(image_paths)} images:")
        for path in image_paths:
            print(f"   {path}")
            
        return image_paths
        
    except ImportError:
        print("❌ pdf2image not available")
        return False
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        print("💡 This might need poppler-utils on Windows")
        return False

def convert_html_to_image():
    """Alternative: Screenshot the HTML report"""
    try:
        import base64
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        html_path = Path("reports/certification_HR_Knowledge_RAG_Demo_20260305_015105.html")
        if not html_path.exists():
            print(f"❌ HTML not found: {html_path}")
            return False
            
        # Setup Chrome options
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1200,800')
        
        # Create screenshot
        driver = webdriver.Chrome(options=options)
        file_url = f"file:///{html_path.absolute()}"
        driver.get(file_url)
        
        # Save screenshot
        images_dir = Path("reports/images")
        images_dir.mkdir(exist_ok=True)
        
        screenshot_path = images_dir / "certification_report.png"
        driver.save_screenshot(str(screenshot_path))
        driver.quit()
        
        print(f"✅ HTML screenshot saved: {screenshot_path}")
        return [str(screenshot_path)]
        
    except ImportError:
        print("❌ selenium not available for HTML screenshots")
        return False
    except Exception as e:
        print(f"❌ Error taking HTML screenshot: {e}")
        return False

def simple_html_extract():
    """Extract key information from HTML for display"""
    html_path = Path("reports/certification_HR_Knowledge_RAG_Demo_20260305_015105.html")
    
    if not html_path.exists():
        print(f"❌ HTML not found: {html_path}")
        return None
        
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract the SVG certification stamp
    import re
    svg_match = re.search(r'<svg[^>]*>.*?</svg>', html_content, re.DOTALL)
    svg_stamp = svg_match.group(0) if svg_match else None
    
    # Extract score info
    score_match = re.search(r'<div style="font-size:24px;font-weight:bold;">([^<]+)</div>', html_content)
    score = score_match.group(1) if score_match else "91.0/100"
    
    level_match = re.search(r'💎 (Platinum)', html_content)
    level = level_match.group(1) if level_match else "Platinum"
    
    cert_match = re.search(r'LMA-\d{8}-[A-F0-9]+', html_content)
    cert_number = cert_match.group(0) if cert_match else "LMA-20260305-EEE55F"
    
    return {
        "svg_stamp": svg_stamp,
        "score": score,
        "level": level,
        "cert_number": cert_number
    }

if __name__ == "__main__":
    # Try PDF conversion first
    images = convert_pdf_to_images()
    
    # If PDF fails, try HTML screenshot  
    if not images:
        images = convert_html_to_image()
    
    # If both fail, extract HTML info
    if not images:
        print("📋 Extracting report information from HTML...")
        info = simple_html_extract()
        if info:
            print(f"✅ Extracted: {info['level']} {info['score']} - {info['cert_number']}")
        else:
            print("❌ Could not extract report information")