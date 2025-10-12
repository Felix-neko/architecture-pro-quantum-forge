#!/usr/bin/env python3
"""
Script to apply automatic text replacements based on YAML mapping file.
Processes all text files in input directory and applies term replacements.
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_replacement_map(yaml_file: Path) -> Dict[str, str]:
    """
    Load replacement mapping from YAML file.
    
    Args:
        yaml_file (Path): Path to YAML file with replacement mappings
        
    Returns:
        Dict[str, str]: Dictionary mapping original terms to replacement terms
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert YAML data to flat replacement dictionary
        replacement_map = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and isinstance(value, str):
                    replacement_map[key] = value
        
        return replacement_map
    
    except Exception as e:
        print(f"Ошибка при загрузке YAML файла {yaml_file}: {e}")
        return {}


def create_replacement_patterns(replacement_map: Dict[str, str]) -> List[Tuple[re.Pattern, str, str]]:
    """
    Create regex patterns for replacements with intelligent matching.
    
    Args:
        replacement_map (Dict[str, str]): Dictionary of replacements
        
    Returns:
        List[Tuple[re.Pattern, str, str]]: List of (compiled regex pattern, replacement, original) tuples
    """
    patterns = []
    
    # Sort by length (longest first) to handle overlapping terms correctly
    sorted_terms = sorted(replacement_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, replacement in sorted_terms:
        # Escape special regex characters in the original term
        escaped_original = re.escape(original)
        
        # Determine if this looks like a stem (incomplete word) or full word
        # Heuristic: if the original term ends with common Cyrillic word endings, treat as full word
        # Otherwise, treat as stem that can have endings
        full_word_endings = ['ий', 'ый', 'ой', 'ая', 'ое', 'ые', 'ых', 'ом', 'ем', 'ам', 'ами', 'ах', 'ей', 'ов', 'ев', 'ь', 'я', 'е', 'и', 'у', 'ю', 'а', 'о', 'ы']
        latin_endings = ['s', 'ed', 'ing', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'ful', 'less']
        
        is_likely_full_word = False
        
        # Check for Cyrillic endings (but be more restrictive - only clear endings)
        clear_cyrillic_endings = ['ий', 'ый', 'ой', 'ая', 'ое', 'ые', 'ых', 'ом', 'ем', 'ами', 'ах', 'ей', 'ов', 'ев']
        for ending in clear_cyrillic_endings:
            if original.lower().endswith(ending.lower()) and len(original) > len(ending) + 2:
                is_likely_full_word = True
                break
        
        # Check for Latin endings
        if not is_likely_full_word:
            for ending in latin_endings:
                if original.lower().endswith(ending.lower()) and len(original) > len(ending) + 2:
                    is_likely_full_word = True
                    break
        
        # Check if it's a single character or very short (likely abbreviation)
        if len(original) <= 2:
            is_likely_full_word = True
        
        # Special case: if it's all uppercase, treat as full word (likely abbreviation)
        if original.isupper() and len(original) <= 5:
            is_likely_full_word = True
        
        # Create appropriate pattern
        if is_likely_full_word:
            # Full word matching with word boundaries
            pattern = rf'\b{escaped_original}\b'
        else:
            # Stem matching - match at word start, allow word endings
            # Use word boundary at start, but allow word characters at end
            pattern = rf'\b{escaped_original}(?=[а-яёА-ЯЁa-zA-Z]*\b)'
        
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            patterns.append((compiled_pattern, replacement, original))
        except re.error as e:
            print(f"Предупреждение: Не удалось создать паттерн для '{original}': {e}")
            continue
    
    return patterns


def apply_replacements(text: str, patterns: List[Tuple[re.Pattern, str, str]]) -> Tuple[str, int]:
    """
    Apply all replacement patterns to the text with intelligent stem/word replacement.
    
    Args:
        text (str): Original text
        patterns (List[Tuple[re.Pattern, str, str]]): List of (regex pattern, replacement, original) tuples
        
    Returns:
        Tuple[str, int]: Modified text and number of replacements made
    """
    modified_text = text
    total_replacements = 0
    
    for pattern, replacement, original in patterns:
        # Find all matches with their positions
        matches = list(pattern.finditer(modified_text))
        if matches:
            # Process matches in reverse order to maintain positions
            for match in reversed(matches):
                matched_text = match.group(0)
                start, end = match.span()
                
                # For stem replacements, we need to preserve the ending
                # Check if this was a stem pattern (doesn't end with \b)
                if not pattern.pattern.endswith(r'\b'):
                    # This is a stem replacement
                    # Extract the stem part and the ending
                    original_lower = original.lower()
                    matched_lower = matched_text.lower()
                    
                    if matched_lower.startswith(original_lower):
                        # Get the ending that was matched
                        ending = matched_text[len(original):]
                        # Create the replacement with the same ending
                        full_replacement = replacement + ending
                    else:
                        # Fallback to simple replacement
                        full_replacement = replacement
                else:
                    # This is a full word replacement
                    full_replacement = replacement
                
                # Replace the matched text
                modified_text = modified_text[:start] + full_replacement + modified_text[end:]
                total_replacements += 1
    
    return modified_text, total_replacements


def process_text_file(input_file: Path, output_file: Path, patterns: List[Tuple[re.Pattern, str, str]]) -> Tuple[bool, int]:
    """
    Process a single text file and apply replacements.
    
    Args:
        input_file (Path): Input text file path
        output_file (Path): Output text file path
        patterns (List[Tuple[re.Pattern, str, str]]): Replacement patterns
        
    Returns:
        Tuple[bool, int]: Success status and number of replacements made
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        # Apply replacements
        modified_text, replacements_count = apply_replacements(original_text, patterns)
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write modified text to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_text)
        
        return True, replacements_count
    
    except Exception as e:
        print(f"Ошибка при обработке файла {input_file}: {e}")
        return False, 0


def process_directory(input_dir: Path, output_dir: Path, yaml_file: Path) -> None:
    """
    Process all text files in input directory and apply replacements.
    
    Args:
        input_dir (Path): Input directory with text files
        output_dir (Path): Output directory for processed files
        yaml_file (Path): YAML file with replacement mappings
    """
    # Load replacement mappings
    print(f"Загрузка словаря замен из: {yaml_file}")
    replacement_map = load_replacement_map(yaml_file)
    
    if not replacement_map:
        print("Ошибка: Словарь замен пуст или не удалось загрузить")
        return
    
    print(f"Загружено {len(replacement_map)} правил замены")
    
    # Create replacement patterns
    patterns = create_replacement_patterns(replacement_map)
    print(f"Создано {len(patterns)} паттернов для замены")
    
    # Find all text files in input directory
    txt_files = list(input_dir.glob('*.txt'))
    
    if not txt_files:
        print(f"Не найдено .txt файлов в директории: {input_dir}")
        return
    
    print(f"Найдено {len(txt_files)} текстовых файлов для обработки")
    print()
    
    # Process each text file
    total_files_processed = 0
    total_files_with_changes = 0
    total_replacements = 0
    
    for txt_file in txt_files:
        print(f"Обработка: {txt_file.name}")
        
        # Create output file path
        output_file = output_dir / txt_file.name
        
        # Process the file
        success, replacements_count = process_text_file(txt_file, output_file, patterns)
        
        if success:
            total_files_processed += 1
            if replacements_count > 0:
                total_files_with_changes += 1
                total_replacements += replacements_count
                print(f"  ✓ Сохранено: {txt_file.name} (замен: {replacements_count})")
            else:
                print(f"  ✓ Сохранено: {txt_file.name} (изменений нет)")
        else:
            print(f"  ✗ Ошибка обработки: {txt_file.name}")
    
    print()
    print("=" * 60)
    print(f"Обработка завершена:")
    print(f"  Всего файлов обработано: {total_files_processed}/{len(txt_files)}")
    print(f"  Файлов с изменениями: {total_files_with_changes}")
    print(f"  Общее количество замен: {total_replacements}")
    print(f"  Результаты сохранены в: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Применить автозамену терминов в текстовых файлах на основе YAML-словаря'
    )
    parser.add_argument('input_dir', help='Путь к папке с исходными .txt файлами')
    parser.add_argument('yaml_file', help='Путь к YAML файлу со словарём замен')
    parser.add_argument('output_dir', help='Путь к выходной папке')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    yaml_file = Path(args.yaml_file)
    output_dir = Path(args.output_dir)
    
    # Validate input paths
    if not input_dir.exists():
        print(f"Ошибка: Входная директория не найдена: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"Ошибка: Путь не является директорией: {input_dir}")
        return 1
    
    if not yaml_file.exists():
        print(f"Ошибка: YAML файл не найден: {yaml_file}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Входная директория: {input_dir}")
    print(f"YAML файл замен: {yaml_file}")
    print(f"Выходная директория: {output_dir}")
    print()
    
    # Process the directory
    process_directory(input_dir, output_dir, yaml_file)
    
    return 0


if __name__ == '__main__':
    exit(main())
