"""
File converter utility for ADAM SLM Database
Handles conversion between different file formats
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging


class FileConverter:
    """
    Universal file converter supporting multiple formats
    
    Supported conversions:
    - Text formats: txt, md, rst, html
    - Data formats: csv, json, jsonl, xml, yaml
    - Image formats: png, jpg, jpeg, gif, bmp, webp, svg
    - Document formats: pdf, doc, docx, rtf
    - Archive formats: zip, tar, gz, bz2, 7z
    - Model formats: pt, pth, onnx, safetensors
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp(prefix="adam_converter_")
        
    def __del__(self):
        """Cleanup temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
            
    def convert_file(
        self,
        input_path: str,
        output_path: str,
        target_format: str,
        options: Dict = None
    ) -> Dict[str, Any]:
        """
        Convert file to target format
        
        Args:
            input_path: Source file path
            output_path: Destination file path
            target_format: Target format (extension without dot)
            options: Conversion options
            
        Returns:
            Conversion result with status and metadata
        """
        if not os.path.exists(input_path):
            return {'success': False, 'error': f'Input file not found: {input_path}'}
            
        input_ext = Path(input_path).suffix.lower()[1:]  # Remove dot
        options = options or {}
        
        try:
            # Route to appropriate converter
            if self._is_text_format(input_ext) and self._is_text_format(target_format):
                return self._convert_text(input_path, output_path, target_format, options)
            elif self._is_data_format(input_ext) and self._is_data_format(target_format):
                return self._convert_data(input_path, output_path, target_format, options)
            elif self._is_image_format(input_ext) and self._is_image_format(target_format):
                return self._convert_image(input_path, output_path, target_format, options)
            elif self._is_document_format(input_ext) and target_format in ['txt', 'md', 'html']:
                return self._convert_document_to_text(input_path, output_path, target_format, options)
            elif self._is_archive_format(input_ext):
                return self._extract_archive(input_path, output_path, options)
            elif input_ext in ['pt', 'pth'] and target_format == 'onnx':
                return self._convert_pytorch_to_onnx(input_path, output_path, options)
            else:
                return {'success': False, 'error': f'Conversion from {input_ext} to {target_format} not supported'}
                
        except Exception as e:
            return {'success': False, 'error': f'Conversion failed: {str(e)}'}
            
    def _is_text_format(self, ext: str) -> bool:
        """Check if format is text-based"""
        return ext in ['txt', 'md', 'rst', 'html', 'htm', 'log']
        
    def _is_data_format(self, ext: str) -> bool:
        """Check if format is data-based"""
        return ext in ['csv', 'json', 'jsonl', 'xml', 'yaml', 'yml', 'tsv']
        
    def _is_image_format(self, ext: str) -> bool:
        """Check if format is image-based"""
        return ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'tiff', 'ico']
        
    def _is_document_format(self, ext: str) -> bool:
        """Check if format is document-based"""
        return ext in ['pdf', 'doc', 'docx', 'rtf', 'odt']
        
    def _is_archive_format(self, ext: str) -> bool:
        """Check if format is archive-based"""
        return ext in ['zip', 'tar', 'gz', 'bz2', '7z', 'rar']
        
    def _convert_text(self, input_path: str, output_path: str, target_format: str, options: Dict) -> Dict:
        """Convert between text formats"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if target_format == 'html':
                # Convert to HTML
                if Path(input_path).suffix.lower() == '.md':
                    # Markdown to HTML
                    try:
                        import markdown
                        html_content = markdown.markdown(content)
                        content = f"<!DOCTYPE html><html><body>{html_content}</body></html>"
                    except ImportError:
                        # Simple conversion without markdown library
                        content = f"<html><body><pre>{content}</pre></body></html>"
                else:
                    # Plain text to HTML
                    content = f"<html><body><pre>{content}</pre></body></html>"
                    
            elif target_format == 'md':
                # Convert to Markdown
                if not content.startswith('#'):
                    content = f"# Document\n\n{content}"
                    
            # Write converted content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return {
                'success': True,
                'output_size': os.path.getsize(output_path),
                'encoding': 'utf-8'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Text conversion failed: {str(e)}'}
            
    def _convert_data(self, input_path: str, output_path: str, target_format: str, options: Dict) -> Dict:
        """Convert between data formats"""
        try:
            input_ext = Path(input_path).suffix.lower()[1:]
            
            # Load data based on input format
            if input_ext == 'csv':
                data = self._load_csv(input_path)
            elif input_ext == 'json':
                data = self._load_json(input_path)
            elif input_ext == 'jsonl':
                data = self._load_jsonl(input_path)
            elif input_ext in ['yaml', 'yml']:
                data = self._load_yaml(input_path)
            elif input_ext == 'xml':
                data = self._load_xml(input_path)
            else:
                return {'success': False, 'error': f'Unsupported input format: {input_ext}'}
                
            # Save data in target format
            if target_format == 'csv':
                self._save_csv(data, output_path)
            elif target_format == 'json':
                self._save_json(data, output_path)
            elif target_format == 'jsonl':
                self._save_jsonl(data, output_path)
            elif target_format in ['yaml', 'yml']:
                self._save_yaml(data, output_path)
            elif target_format == 'xml':
                self._save_xml(data, output_path)
            else:
                return {'success': False, 'error': f'Unsupported output format: {target_format}'}
                
            return {
                'success': True,
                'output_size': os.path.getsize(output_path),
                'record_count': len(data) if isinstance(data, list) else 1
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Data conversion failed: {str(e)}'}
            
    def _convert_image(self, input_path: str, output_path: str, target_format: str, options: Dict) -> Dict:
        """Convert between image formats"""
        try:
            from PIL import Image
            
            with Image.open(input_path) as img:
                # Handle transparency for JPEG
                if target_format.lower() in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                    
                # Apply options
                quality = options.get('quality', 95)
                optimize = options.get('optimize', True)
                
                # Save in target format
                save_kwargs = {}
                if target_format.lower() in ['jpg', 'jpeg']:
                    save_kwargs.update({'quality': quality, 'optimize': optimize})
                elif target_format.lower() == 'png':
                    save_kwargs.update({'optimize': optimize})
                elif target_format.lower() == 'webp':
                    save_kwargs.update({'quality': quality, 'optimize': optimize})
                    
                img.save(output_path, format=target_format.upper(), **save_kwargs)
                
            return {
                'success': True,
                'output_size': os.path.getsize(output_path),
                'original_size': img.size,
                'format': target_format.upper()
            }
            
        except ImportError:
            return {'success': False, 'error': 'PIL/Pillow not available for image conversion'}
        except Exception as e:
            return {'success': False, 'error': f'Image conversion failed: {str(e)}'}
            
    def _convert_document_to_text(self, input_path: str, output_path: str, target_format: str, options: Dict) -> Dict:
        """Convert document to text format"""
        try:
            input_ext = Path(input_path).suffix.lower()[1:]
            extracted_text = ""
            
            if input_ext == 'pdf':
                extracted_text = self._extract_pdf_text(input_path)
            elif input_ext in ['doc', 'docx']:
                extracted_text = self._extract_word_text(input_path)
            elif input_ext == 'rtf':
                extracted_text = self._extract_rtf_text(input_path)
            else:
                return {'success': False, 'error': f'Unsupported document format: {input_ext}'}
                
            if not extracted_text:
                return {'success': False, 'error': 'No text could be extracted'}
                
            # Format output based on target format
            if target_format == 'md':
                # Add markdown formatting
                extracted_text = f"# Extracted Document\n\n{extracted_text}"
            elif target_format == 'html':
                # Add HTML formatting
                extracted_text = f"<html><body><h1>Extracted Document</h1><pre>{extracted_text}</pre></body></html>"
                
            # Write extracted text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
                
            return {
                'success': True,
                'output_size': os.path.getsize(output_path),
                'word_count': len(extracted_text.split()),
                'extraction_method': f'{input_ext}_parser'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Document conversion failed: {str(e)}'}
            
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF with multiple fallback methods"""
        extracted_text = ""

        # Method 1: Try pdfplumber (best for complex layouts)
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n--- Page {i+1} ---\n"
                        extracted_text += page_text + "\n"

            if extracted_text.strip():
                return extracted_text

        except ImportError:
            print("pdfplumber not available, trying PyPDF2...")
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyPDF2...")

        # Method 2: Try PyPDF2 (fallback)
        try:
            import PyPDF2

            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n--- Page {i+1} ---\n"
                        extracted_text += page_text + "\n"

            if extracted_text.strip():
                return extracted_text

        except ImportError:
            print("PyPDF2 not available, trying pymupdf...")
        except Exception as e:
            print(f"PyPDF2 failed: {e}, trying pymupdf...")

        # Method 3: Try pymupdf/fitz (another fallback)
        try:
            import fitz  # pymupdf

            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    extracted_text += f"\n--- Page {i+1} ---\n"
                    extracted_text += page_text + "\n"
            doc.close()

            if extracted_text.strip():
                return extracted_text

        except ImportError:
            print("pymupdf not available")
        except Exception as e:
            print(f"pymupdf failed: {e}")

        # Method 4: Try pdfminer (last resort)
        try:
            from pdfminer.high_level import extract_text
            extracted_text = extract_text(pdf_path)
            if extracted_text.strip():
                return extracted_text

        except ImportError:
            print("pdfminer not available")
        except Exception as e:
            print(f"pdfminer failed: {e}")

        return extracted_text if extracted_text.strip() else "Could not extract text from PDF"
                
    def _extract_word_text(self, doc_path: str) -> str:
        """Extract text from Word document"""
        try:
            import docx
            
            doc = docx.Document(doc_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        except ImportError:
            return ""
            
    def _extract_rtf_text(self, rtf_path: str) -> str:
        """Extract text from RTF document"""
        try:
            from striprtf.striprtf import rtf_to_text
            
            with open(rtf_path, 'r') as f:
                rtf_content = f.read()
            return rtf_to_text(rtf_content)
            
        except ImportError:
            return ""
            
    def _load_csv(self, file_path: str) -> List[Dict]:
        """Load CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except ImportError:
            import csv
            with open(file_path, 'r') as f:
                return list(csv.DictReader(f))
                
    def _load_json(self, file_path: str) -> Any:
        """Load JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
        
    def _load_yaml(self, file_path: str) -> Any:
        """Load YAML file"""
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_xml(self, file_path: str) -> Dict:
        """Load XML file"""
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        return self._xml_to_dict(root)
        
    def _xml_to_dict(self, element) -> Dict:
        """Convert XML element to dictionary"""
        result = {}
        if element.text and element.text.strip():
            result['text'] = element.text.strip()
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        return result
        
    def _save_csv(self, data: List[Dict], file_path: str):
        """Save data as CSV"""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        except ImportError:
            import csv
            if data:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    
    def _save_json(self, data: Any, file_path: str):
        """Save data as JSON"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _save_jsonl(self, data: List[Dict], file_path: str):
        """Save data as JSONL"""
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
    def _save_yaml(self, data: Any, file_path: str):
        """Save data as YAML"""
        import yaml
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
    def _save_xml(self, data: Dict, file_path: str):
        """Save data as XML"""
        import xml.etree.ElementTree as ET
        root = self._dict_to_xml(data, 'root')
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
    def _dict_to_xml(self, data: Any, tag: str):
        """Convert dictionary to XML element"""
        import xml.etree.ElementTree as ET
        element = ET.Element(tag)
        
        if isinstance(data, dict):
            for key, value in data.items():
                child = self._dict_to_xml(value, key)
                element.append(child)
        elif isinstance(data, list):
            for item in data:
                child = self._dict_to_xml(item, 'item')
                element.append(child)
        else:
            element.text = str(data)
            
        return element
