#!/usr/bin/env python3
"""
Scrape the SHL product catalog to extract Individual Test Solutions.
"""

import json
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict


def find_individual_solutions_table(soup: BeautifulSoup):
    """Return the Individual Test Solutions table regardless of page layout."""

    for table in soup.find_all('table'):
        header_cell = table.find('th')
        if header_cell and 'Individual Test Solutions' in header_cell.get_text(' ', strip=True):
            return table
    return None

def scrape_individual_solutions() -> List[Dict]:
    """Scrape all Individual Test Solutions from the SHL catalog."""
    
    all_assessments = []
    base_url = "https://www.shl.com/products/product-catalog/"
    
    # Individual Test Solutions pagination
    # type=1 for Individual Test Solutions
    # 32 pages total, 12 assessments per page
    
    for page in range(1, 33):
        try:
            print(f"Scraping page {page}/32...", end="", flush=True)
            
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}?start={(page-1)*12}&type=1"
            
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = find_individual_solutions_table(soup)

            if table is None:
                print(" (Warning: Individual Test Solutions table not found)")
                continue

            rows = table.find_all('tr')[1:]  # Skip header row
            
            page_count = 0
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 4:
                    continue
                
                # Extract assessment data
                name_cell = cells[0]
                link = name_cell.find('a')
                
                if not link:
                    continue
                
                name = link.text.strip()
                url = link.get('href', '').strip()
                
                # Ensure absolute URL
                if url and not url.startswith('http'):
                    url = 'https://www.shl.com' + url
                
                # Extract test types
                test_type_cell = cells[3]
                test_types = [span.text.strip() for span in test_type_cell.find_all('span', class_='product-catalogue__key')]
                
                # Extract remote testing and adaptive info from cells 1 and 2
                remote_testable = len(cells[1].text.strip()) > 0
                adaptive = len(cells[2].text.strip()) > 0
                
                assessment = {
                    "name": name,
                    "url": url,
                    "test_types": test_types,
                    "remote_testable": remote_testable,
                    "adaptive": adaptive
                }
                
                all_assessments.append(assessment)
                page_count += 1
            
            print(f" found {page_count} assessments")
            time.sleep(0.3)  # Be nice to the server
            
        except Exception as e:
            print(f" ERROR: {e}")
            continue
    
    return all_assessments

if __name__ == "__main__":
    print("Scraping SHL Product Catalog - Individual Test Solutions")
    print("=" * 60)
    
    assessments = scrape_individual_solutions()
    
    print("\n" + "=" * 60)
    print(f"Total assessments scraped: {len(assessments)}")
    
    # Save to JSON
    output_file = "shl_catalog.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")
    
    # Print sample
    print(f"\nSample assessments:")
    for assessment in assessments[:5]:
        print(f"  - {assessment['name']} ({', '.join(assessment['test_types'])})")
