#!/usr/bin/env python3
"""
HTML to Text Converter for Arcanum Wiki Articles

This script extracts titles and content from HTML files with specific structure
and converts them to clean text files while preserving paragraph breaks.
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Ошибка: Требуется установить библиотеку beautifulsoup4")
    print("Выполните: pip install beautifulsoup4")
    sys.exit(1)


def fix_spacing(text):
    """
    Fix common spacing issues in extracted text.
    
    Args:
        text (str): Text with potential spacing issues
        
    Returns:
        str: Text with fixed spacing
    """
    # Fix spacing around em-dashes (but not hyphens in compound words)
    text = re.sub(r'(\w)—(\w)', r'\1 — \2', text)
    
    # Fix spacing around parentheses
    text = re.sub(r'(\w)\(', r'\1 (', text)
    text = re.sub(r'\)(\w)', r') \1', text)
    
    # Fix spacing around common prepositions that got merged (but avoid compound words)
    # Only add spaces if the preposition is followed by a capital letter (new word)
    text = re.sub(r'(\w)в([А-ЯЁ][а-яё])', r'\1 в \2', text)
    text = re.sub(r'(\w)на([А-ЯЁ][а-яё])', r'\1 на \2', text)
    text = re.sub(r'(\w)из([А-ЯЁ][а-яё])', r'\1 из \2', text)
    text = re.sub(r'(\w)от([А-ЯЁ][а-яё])', r'\1 от \2', text)
    text = re.sub(r'(\w)для([А-ЯЁ][а-яё])', r'\1 для \2', text)
    text = re.sub(r'(\w)при([А-ЯЁ][а-яё])', r'\1 при \2', text)
    text = re.sub(r'(\w)под([А-ЯЁ][а-яё])', r'\1 под \2', text)
    text = re.sub(r'(\w)над([А-ЯЁ][а-яё])', r'\1 над \2', text)
    text = re.sub(r'(\w)за([А-ЯЁ][а-яё])', r'\1 за \2', text)
    text = re.sub(r'(\w)до([А-ЯЁ][а-яё])', r'\1 до \2', text)
    text = re.sub(r'(\w)после([А-ЯЁ][а-яё])', r'\1 после \2', text)
    text = re.sub(r'(\w)перед([А-ЯЁ][а-яё])', r'\1 перед \2', text)
    
    # Fix spacing around common conjunctions (only before capital letters)
    text = re.sub(r'(\w)и([А-ЯЁ][а-яё])', r'\1 и \2', text)
    text = re.sub(r'(\w)или([А-ЯЁ][а-яё])', r'\1 или \2', text)
    text = re.sub(r'(\w)но([А-ЯЁ][а-яё])', r'\1 но \2', text)
    text = re.sub(r'(\w)а([А-ЯЁ][а-яё])', r'\1 а \2', text)
    
    # Fix spacing after punctuation (but not before)
    text = re.sub(r'\.([А-ЯЁа-яё])', r'. \1', text)
    text = re.sub(r',([А-ЯЁа-яё])', r', \1', text)
    text = re.sub(r':([А-ЯЁа-яё])', r': \1', text)
    text = re.sub(r';([А-ЯЁа-яё])', r'; \1', text)
    
    # Remove spaces before punctuation (fix the comma issue)
    text = re.sub(r'\s+([,;:.])', r'\1', text)
    
    # Fix compound words that got broken (common Russian compounds)
    text = re.sub(r'из\s+-\s+за', 'из-за', text)
    text = re.sub(r'по\s+-\s+прежнему', 'по-прежнему', text)
    text = re.sub(r'во\s+-\s+первых', 'во-первых', text)
    text = re.sub(r'во\s+-\s+вторых', 'во-вторых', text)
    text = re.sub(r'в\s+-\s+третьих', 'в-третьих', text)
    text = re.sub(r'кое\s+-\s+что', 'кое-что', text)
    text = re.sub(r'кое\s+-\s+как', 'кое-как', text)
    text = re.sub(r'кое\s+-\s+где', 'кое-где', text)
    text = re.sub(r'кто\s+-\s+то', 'кто-то', text)
    text = re.sub(r'что\s+-\s+то', 'что-то', text)
    text = re.sub(r'где\s+-\s+то', 'где-то', text)
    text = re.sub(r'когда\s+-\s+то', 'когда-то', text)
    text = re.sub(r'как\s+-\s+то', 'как-то', text)
    text = re.sub(r'почему\s+-\s+то', 'почему-то', text)
    
    # Fix specific HTML formatting issues
    # Fix "Brummon d" -> "Brummond" (HTML italic tag separation issue)
    text = re.sub(r'Brummon\s+d\b', 'Brummond', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_direct_text_content(element):
    """
    Get only the direct text content of an element, excluding nested ul/ol elements.
    
    Args:
        element: BeautifulSoup element
        
    Returns:
        str: Direct text content only
    """
    # Clone the element to avoid modifying the original
    element_copy = element.__copy__()
    
    # Remove all nested ul and ol elements
    for nested_list in element_copy.find_all(['ul', 'ol']):
        nested_list.decompose()
    
    # Get the text content
    text = element_copy.get_text(separator=' ', strip=True)
    return text


def extract_content_from_html(html_file_path):
    """
    Extract title and content from an HTML file.
    
    Args:
        html_file_path (Path): Path to the HTML file
        
    Returns:
        tuple: (title, content) where both are strings
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title from <h1 class="page-header__title" id="firstHeading">
        title_element = soup.find('h1', {'class': 'page-header__title', 'id': 'firstHeading'})
        if title_element:
            # Get text from the span inside the h1
            title_span = title_element.find('span', class_='mw-page-title-main')
            title = title_span.get_text(strip=True) if title_span else title_element.get_text(strip=True)
        else:
            title = "Заголовок не найден"
        
        # Extract content from <div id="content" class="page-content">
        content_div = soup.find('div', {'id': 'content', 'class': 'page-content'})
        if content_div:
            # Find the main content area within the content div
            main_content = content_div.find('div', class_='mw-content-ltr mw-parser-output')
            if main_content:
                # Extract text while preserving paragraph structure
                content_parts = []
                processed_elements = set()
                
                # First, handle information boxes (pi-data structures)
                info_boxes = main_content.find_all('div', class_=lambda x: x and 'pi-item' in x and 'pi-data' in x)
                if info_boxes:
                    info_pairs = []
                    for box in info_boxes:
                        label_elem = box.find('h3', class_=lambda x: x and 'pi-data-label' in x)
                        value_elem = box.find('div', class_=lambda x: x and 'pi-data-value' in x)
                        
                        if label_elem and value_elem:
                            label = label_elem.get_text(strip=True)
                            value = value_elem.get_text(strip=True)
                            if label and value:
                                label = fix_spacing(label)
                                value = fix_spacing(value)
                                info_pairs.append(f"{label}\t{value}")
                                # Mark these elements as processed
                                processed_elements.add(id(box))
                                processed_elements.add(id(label_elem))
                                processed_elements.add(id(value_elem))
                    
                    if info_pairs:
                        content_parts.append('\n'.join(info_pairs))
                
                # Process each element to preserve paragraph breaks
                for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'div']):
                    # Skip if this element has already been processed as a nested element or info box
                    if id(element) in processed_elements:
                        continue
                    
                    # Skip info box related elements and their parents
                    if element.get('class'):
                        classes = element.get('class')
                        if any('pi-' in cls for cls in classes):
                            continue
                        # Also skip elements that contain info boxes
                        if element.find('div', class_=lambda x: x and 'pi-item' in x and 'pi-data' in x):
                            continue
                        
                    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        # Handle headers
                        text = element.get_text(strip=True)
                        if text and not text.endswith('[]'):  # Skip edit section markers
                            text = fix_spacing(text)
                            content_parts.append(f"\n{text}\n")
                    elif element.name in ['ul', 'ol']:
                        # Handle lists with proper nesting
                        list_items = []
                        for li in element.find_all('li', recursive=False):
                            # Get only the direct text content of this li, not nested lists
                            li_text = get_direct_text_content(li)
                            if li_text:
                                li_text = fix_spacing(li_text)
                                list_items.append(f"• {li_text}")
                            
                            # Process nested lists separately and mark them as processed
                            nested_lists = li.find_all(['ul', 'ol'], recursive=False)
                            for nested_list in nested_lists:
                                processed_elements.add(id(nested_list))
                                for nested_li in nested_list.find_all('li', recursive=False):
                                    nested_text = get_direct_text_content(nested_li)
                                    if nested_text:
                                        nested_text = fix_spacing(nested_text)
                                        list_items.append(f"  • {nested_text}")  # Indent nested items
                        
                        if list_items:
                            content_parts.append('\n'.join(list_items))
                    elif element.name == 'p':
                        # Handle paragraphs - extract only direct text, excluding nested info boxes
                        # First, check if this paragraph contains an info box (aside element)
                        if element.find('aside', class_=lambda x: x and 'portable-infobox' in x):
                            # This paragraph contains an info box, extract only the direct text content
                            # Create a copy and remove the aside element
                            element_copy = element.__copy__()
                            aside_elem = element_copy.find('aside', class_=lambda x: x and 'portable-infobox' in x)
                            if aside_elem:
                                aside_elem.decompose()
                            text = element_copy.get_text(separator=' ', strip=True)
                        else:
                            # Regular paragraph without info box
                            text = element.get_text(separator=' ', strip=True)
                        
                        if text:
                            # Fix common spacing issues
                            text = fix_spacing(text)
                            content_parts.append(text)
                    elif element.name == 'div':
                        # Handle other divs that might contain content
                        # Skip if it's empty or contains only whitespace
                        text = element.get_text(separator=' ', strip=True)
                        if text and len(text) > 10:  # Only process substantial content
                            # Check if it's not a structural div
                            classes = element.get('class', [])
                            if not any(cls in ['mw-parser-output', 'pi-', 'toc'] for cls in classes):
                                text = fix_spacing(text)
                                content_parts.append(text)
                
                content = '\n\n'.join(content_parts)
            else:
                content = content_div.get_text(separator='\n\n', strip=True)
        else:
            content = "Контент не найден"
        
        return title, content
        
    except Exception as e:
        print(f"Ошибка при обработке файла {html_file_path}: {e}")
        return "Ошибка извлечения заголовка", "Ошибка извлечения контента"


def convert_html_to_text(input_dir, output_dir):
    """
    Convert all HTML files in input directory to text files in output directory.
    
    Args:
        input_dir (Path): Directory containing HTML files
        output_dir (Path): Directory where text files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Ошибка: Входная папка '{input_path}' не существует.")
        return False
    
    if not input_path.is_dir():
        print(f"Ошибка: '{input_path}' не является папкой.")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all HTML files
    html_files = list(input_path.glob('*.html'))
    if not html_files:
        print(f"В папке '{input_path}' не найдено HTML-файлов.")
        return False
    
    print(f"Найдено {len(html_files)} HTML-файлов для обработки...")
    
    successful_conversions = 0
    
    for html_file in html_files:
        print(f"Обработка: {html_file.name}")
        
        # Extract content
        title, content = extract_content_from_html(html_file)
        
        # Create output filename (same name but with .txt extension)
        output_filename = html_file.stem + '.txt'
        output_file_path = output_path / output_filename
        
        # Write to text file
        try:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(f"{title}\n\n{content}")
            
            print(f"  ✓ Сохранено: {output_filename}")
            successful_conversions += 1
            
        except Exception as e:
            print(f"  ✗ Ошибка при сохранении {output_filename}: {e}")
    
    print(f"\nОбработка завершена. Успешно конвертировано: {successful_conversions}/{len(html_files)} файлов.")
    return successful_conversions > 0


def main():
    """Main function to handle command line arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description='Конвертирует HTML-файлы в текстовые файлы, извлекая заголовки и контент.'
    )
    parser.add_argument(
        'input_dir',
        help='Путь к папке с HTML-файлами'
    )
    parser.add_argument(
        'output_dir',
        help='Путь к выходной папке для текстовых файлов'
    )
    
    args = parser.parse_args()
    
    # Convert and exit with appropriate code
    success = convert_html_to_text(args.input_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
