"""
File management system for ADAM SLM Database
Handles all types of files with conversion, import, and processing capabilities
"""

import os
import hashlib
import mimetypes
import json
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging

from database import AdamSLMDatabase


class FileManager:
    """
    Comprehensive file management system for ADAM SLM
    
    Supports:
    - All file types (text, binary, images, documents, models, etc.)
    - File conversion and processing
    - Content extraction and analysis
    - Metadata management
    - Version control
    - Access logging
    """
    
    def __init__(self, db: AdamSLMDatabase, storage_root: str = "./file_storage"):
        self.db = db
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Create storage subdirectories
        self._create_storage_structure()
        
    def _create_storage_structure(self):
        """Create organized storage directory structure"""
        subdirs = [
            "datasets", "models", "configs", "logs", "images", 
            "documents", "archives", "temp", "processed"
        ]
        
        for subdir in subdirs:
            (self.storage_root / subdir).mkdir(exist_ok=True)
            
    def _calculate_checksums(self, file_path: str) -> Tuple[str, str]:
        """Calculate MD5 and SHA256 checksums for file"""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
                
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
        
    def _detect_file_type(self, file_path: str, filename: str) -> Tuple[str, str, str]:
        """Detect file type, format, and MIME type"""
        file_ext = Path(filename).suffix.lower()
        mime_type, _ = mimetypes.guess_type(filename)
        
        # File type classification
        if file_ext in ['.txt', '.md', '.rst', '.log']:
            file_type = 'text'
        elif file_ext in ['.json', '.yaml', '.yml', '.xml', '.toml']:
            file_type = 'config'
        elif file_ext in ['.csv', '.tsv', '.jsonl', '.parquet']:
            file_type = 'dataset'
        elif file_ext in ['.pt', '.pth', '.ckpt', '.safetensors', '.bin']:
            file_type = 'model'
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg']:
            file_type = 'image'
        elif file_ext in ['.pdf', '.doc', '.docx', '.rtf']:
            file_type = 'document'
        elif file_ext in ['.zip', '.tar', '.gz', '.bz2', '.7z']:
            file_type = 'archive'
        elif file_ext in ['.py', '.js', '.cpp', '.java', '.go', '.rs']:
            file_type = 'code'
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            file_type = 'video'
        elif file_ext in ['.mp3', '.wav', '.flac', '.ogg']:
            file_type = 'audio'
        else:
            file_type = 'binary'
            
        file_format = file_ext[1:] if file_ext else 'unknown'
        
        return file_type, file_format, mime_type or 'application/octet-stream'
        
    def _get_storage_path(self, file_type: str, filename: str) -> Path:
        """Get organized storage path for file"""
        # Create subdirectory based on file type
        type_dir = self.storage_root / file_type
        type_dir.mkdir(exist_ok=True)
        
        # Add date-based subdirectory
        date_dir = type_dir / datetime.now().strftime("%Y/%m")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        return date_dir / filename
        
    def register_file(
        self,
        file_path: str,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        description: str = None,
        tags: List[str] = None,
        created_by: int = None,
        metadata: Dict = None,
        copy_to_storage: bool = True,
        process_immediately: bool = True
    ) -> int:
        """
        Register a file in the database with optional storage and processing
        
        Args:
            file_path: Path to the source file
            filename: Custom filename (defaults to original)
            file_type: Override file type detection
            description: File description
            tags: List of tags
            created_by: User ID who uploaded the file
            metadata: Additional metadata
            copy_to_storage: Whether to copy file to managed storage
            process_immediately: Whether to start processing immediately
            
        Returns:
            File ID in database
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Use original filename if not provided
        if filename is None:
            filename = os.path.basename(file_path)
            
        # Detect file properties
        detected_type, file_format, mime_type = self._detect_file_type(file_path, filename)
        if file_type is None:
            file_type = detected_type
            
        # Calculate file size and checksums
        file_size = os.path.getsize(file_path)
        md5_checksum, sha256_checksum = self._calculate_checksums(file_path)
        
        # Determine storage path
        if copy_to_storage:
            storage_path = self._get_storage_path(file_type, filename)
            # Ensure unique filename
            counter = 1
            while storage_path.exists():
                name_parts = filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    new_filename = f"{filename}_{counter}"
                storage_path = self._get_storage_path(file_type, new_filename)
                counter += 1
                
            # Copy file to storage
            shutil.copy2(file_path, storage_path)
            stored_path = str(storage_path)
        else:
            stored_path = file_path
            
        # Prepare metadata
        file_metadata = {
            "original_size": file_size,
            "upload_timestamp": datetime.now().isoformat(),
            "source_path": file_path,
            **(metadata or {})
        }
        
        # Register in database
        file_id = self.db.execute_insert("""
            INSERT INTO file_registry (
                filename, original_path, stored_path, file_type, file_format,
                file_size_bytes, mime_type, checksum_md5, checksum_sha256,
                created_by, description, metadata, tags, processing_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, file_path, stored_path, file_type, file_format,
            file_size, mime_type, md5_checksum, sha256_checksum,
            created_by, description, json.dumps(file_metadata),
            json.dumps(tags or []), 'pending' if process_immediately else 'registered'
        ))
        
        self.logger.info(f"Registered file: {filename} (ID: {file_id}, Type: {file_type})")
        
        # Start processing if requested
        if process_immediately:
            self.queue_processing_job(file_id, 'analyze', created_by)
            
        return file_id
        
    def queue_processing_job(
        self,
        file_id: int,
        job_type: str,
        created_by: int = None,
        job_config: Dict = None,
        priority: int = 5
    ) -> int:
        """Queue a file processing job"""
        
        job_id = self.db.execute_insert("""
            INSERT INTO file_processing_jobs (
                file_id, job_type, created_by, job_config, priority
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            file_id, job_type, created_by,
            json.dumps(job_config or {}), priority
        ))
        
        self.logger.info(f"Queued {job_type} job for file {file_id} (Job ID: {job_id})")
        return job_id
        
    def process_file(self, file_id: int, job_type: str = 'analyze') -> bool:
        """Process a file based on its type and job type"""
        
        # Get file info
        file_info = self.db.execute_query(
            "SELECT * FROM file_registry WHERE id = ?", (file_id,)
        )
        
        if not file_info:
            self.logger.error(f"File not found: {file_id}")
            return False
            
        file_info = file_info[0]
        
        try:
            # Update job status
            self.db.execute_update("""
                UPDATE file_processing_jobs 
                SET job_status = 'running', started_at = CURRENT_TIMESTAMP 
                WHERE file_id = ? AND job_type = ?
            """, (file_id, job_type))
            
            # Process based on file type and job type
            if job_type == 'analyze':
                result = self._analyze_file(file_info)
            elif job_type == 'convert':
                result = self._convert_file(file_info)
            elif job_type == 'extract':
                result = self._extract_content(file_info)
            else:
                result = {'error': f'Unknown job type: {job_type}'}
                
            # Update job completion
            if 'error' in result:
                self.db.execute_update("""
                    UPDATE file_processing_jobs 
                    SET job_status = 'failed', completed_at = CURRENT_TIMESTAMP,
                        error_message = ?, progress_percent = 0
                    WHERE file_id = ? AND job_type = ?
                """, (result['error'], file_id, job_type))
                return False
            else:
                self.db.execute_update("""
                    UPDATE file_processing_jobs 
                    SET job_status = 'completed', completed_at = CURRENT_TIMESTAMP,
                        progress_percent = 100, result_data = ?
                    WHERE file_id = ? AND job_type = ?
                """, (json.dumps(result), file_id, job_type))
                
                # Update file processing status
                self.db.execute_update("""
                    UPDATE file_registry 
                    SET processing_status = 'completed', is_processed = 1
                    WHERE id = ?
                """, (file_id,))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_id}: {e}")
            
            # Update job with error
            self.db.execute_update("""
                UPDATE file_processing_jobs 
                SET job_status = 'failed', completed_at = CURRENT_TIMESTAMP,
                    error_message = ?
                WHERE file_id = ? AND job_type = ?
            """, (str(e), file_id, job_type))
            
            return False
            
    def _analyze_file(self, file_info: Dict) -> Dict:
        """Analyze file and extract metadata"""
        file_path = file_info['stored_path']
        file_type = file_info['file_type']
        
        analysis = {
            'analyzed_at': datetime.now().isoformat(),
            'file_type': file_type,
            'analysis_version': '1.0'
        }
        
        try:
            if file_type == 'text':
                analysis.update(self._analyze_text_file(file_path))
            elif file_type == 'dataset':
                analysis.update(self._analyze_dataset_file(file_path))
            elif file_type == 'image':
                analysis.update(self._analyze_image_file(file_path))
            elif file_type == 'model':
                analysis.update(self._analyze_model_file(file_path))
            elif file_type == 'config':
                analysis.update(self._analyze_config_file(file_path))
            else:
                analysis.update(self._analyze_generic_file(file_path))
                
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
            
        return analysis

    def _analyze_text_file(self, file_path: str) -> Dict:
        """Analyze text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            words = content.split()

            analysis = {
                'line_count': len(lines),
                'word_count': len(words),
                'character_count': len(content),
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'encoding': 'utf-8',
                'language': 'auto-detected'  # Could add language detection
            }

            # Store extracted content
            self._store_file_content(file_path, content, 'text', 'direct')

            return analysis

        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return {'encoding': encoding, 'content_preview': content[:500]}
                except:
                    continue
            return {'error': 'Could not decode text file'}

    def _analyze_dataset_file(self, file_path: str) -> Dict:
        """Analyze dataset file (CSV, JSON, etc.)"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.csv':
                return self._analyze_csv_file(file_path)
            elif file_ext in ['.json', '.jsonl']:
                return self._analyze_json_file(file_path)
            elif file_ext == '.parquet':
                return self._analyze_parquet_file(file_path)
            else:
                return self._analyze_generic_file(file_path)

        except Exception as e:
            return {'error': f'Dataset analysis failed: {str(e)}'}

    def _analyze_csv_file(self, file_path: str) -> Dict:
        """Analyze CSV file structure"""
        try:
            import pandas as pd

            # Read sample of CSV
            df = pd.read_csv(file_path, nrows=1000)

            analysis = {
                'format': 'csv',
                'columns': list(df.columns),
                'column_count': len(df.columns),
                'sample_row_count': len(df),
                'data_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'has_header': True,  # Assumed
                'null_counts': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
            }

            # Get full row count
            with open(file_path, 'r') as f:
                total_rows = sum(1 for line in f) - 1  # Subtract header
            analysis['total_rows'] = total_rows

            return analysis

        except ImportError:
            # Fallback without pandas
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                total_lines = sum(1 for line in f) + 1

            return {
                'format': 'csv',
                'total_rows': total_lines - 1,
                'columns': first_line.split(','),
                'column_count': len(first_line.split(',')),
                'note': 'Basic analysis (pandas not available)'
            }

    def _analyze_json_file(self, file_path: str) -> Dict:
        """Analyze JSON/JSONL file structure"""
        try:
            file_size = os.path.getsize(file_path)

            with open(file_path, 'r') as f:
                first_line = f.readline().strip()

            # Check if JSONL or JSON
            try:
                json.loads(first_line)
                is_jsonl = True
            except:
                is_jsonl = False

            if is_jsonl:
                # JSONL analysis
                with open(file_path, 'r') as f:
                    line_count = sum(1 for line in f)

                # Sample first few records
                sample_records = []
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 5:  # Sample first 5 records
                            break
                        try:
                            sample_records.append(json.loads(line))
                        except:
                            continue

                analysis = {
                    'format': 'jsonl',
                    'record_count': line_count,
                    'sample_records': len(sample_records),
                    'file_size_mb': file_size / (1024 * 1024)
                }

                if sample_records:
                    # Analyze structure of first record
                    first_record = sample_records[0]
                    if isinstance(first_record, dict):
                        analysis['keys'] = list(first_record.keys())
                        analysis['key_count'] = len(first_record.keys())

            else:
                # Regular JSON analysis
                with open(file_path, 'r') as f:
                    data = json.load(f)

                analysis = {
                    'format': 'json',
                    'data_type': type(data).__name__,
                    'file_size_mb': file_size / (1024 * 1024)
                }

                if isinstance(data, list):
                    analysis['item_count'] = len(data)
                    if data and isinstance(data[0], dict):
                        analysis['keys'] = list(data[0].keys())
                elif isinstance(data, dict):
                    analysis['keys'] = list(data.keys())
                    analysis['key_count'] = len(data.keys())

            return analysis

        except Exception as e:
            return {'error': f'JSON analysis failed: {str(e)}'}

    def _analyze_image_file(self, file_path: str) -> Dict:
        """Analyze image file properties"""
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                analysis = {
                    'format': img.format,
                    'mode': img.mode,
                    'width': img.width,
                    'height': img.height,
                    'size': f"{img.width}x{img.height}",
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }

                if hasattr(img, 'info') and img.info:
                    analysis['metadata'] = dict(img.info)

            return analysis

        except ImportError:
            return {'error': 'PIL/Pillow not available for image analysis'}
        except Exception as e:
            return {'error': f'Image analysis failed: {str(e)}'}

    def _analyze_model_file(self, file_path: str) -> Dict:
        """Analyze model file (PyTorch, etc.)"""
        try:
            file_ext = Path(file_path).suffix.lower()
            file_size = os.path.getsize(file_path)

            analysis = {
                'format': file_ext[1:],
                'file_size_mb': file_size / (1024 * 1024),
                'estimated_parameters': self._estimate_model_parameters(file_size)
            }

            if file_ext in ['.pt', '.pth']:
                try:
                    import torch
                    # Load model metadata without loading full model
                    checkpoint = torch.load(file_path, map_location='cpu')

                    if isinstance(checkpoint, dict):
                        analysis['checkpoint_keys'] = list(checkpoint.keys())

                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                            analysis['model_layers'] = len(state_dict)
                            analysis['parameter_count'] = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))

                        if 'optimizer_state_dict' in checkpoint:
                            analysis['has_optimizer'] = True

                        if 'epoch' in checkpoint:
                            analysis['epoch'] = checkpoint['epoch']

                        if 'loss' in checkpoint:
                            analysis['loss'] = checkpoint['loss']

                except ImportError:
                    analysis['note'] = 'PyTorch not available for detailed analysis'
                except Exception as e:
                    analysis['note'] = f'Could not load checkpoint: {str(e)}'

            return analysis

        except Exception as e:
            return {'error': f'Model analysis failed: {str(e)}'}

    def _analyze_config_file(self, file_path: str) -> Dict:
        """Analyze configuration file"""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.json':
                with open(file_path, 'r') as f:
                    config = json.load(f)

                analysis = {
                    'format': 'json',
                    'keys': list(config.keys()) if isinstance(config, dict) else [],
                    'key_count': len(config.keys()) if isinstance(config, dict) else 0,
                    'data_type': type(config).__name__
                }

            elif file_ext in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)

                    analysis = {
                        'format': 'yaml',
                        'keys': list(config.keys()) if isinstance(config, dict) else [],
                        'key_count': len(config.keys()) if isinstance(config, dict) else 0,
                        'data_type': type(config).__name__
                    }
                except ImportError:
                    analysis = {'error': 'PyYAML not available for YAML analysis'}

            else:
                # Generic text analysis
                analysis = self._analyze_text_file(file_path)
                analysis['format'] = file_ext[1:]

            return analysis

        except Exception as e:
            return {'error': f'Config analysis failed: {str(e)}'}

    def _analyze_generic_file(self, file_path: str) -> Dict:
        """Generic file analysis"""
        try:
            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()

            analysis = {
                'format': file_ext[1:] if file_ext else 'unknown',
                'file_size_mb': file_size / (1024 * 1024),
                'is_binary': self._is_binary_file(file_path)
            }

            # Try to read as text for preview
            if not analysis['is_binary']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        preview = f.read(500)  # First 500 characters
                    analysis['text_preview'] = preview
                except:
                    analysis['text_preview'] = 'Could not read as text'

            return analysis

        except Exception as e:
            return {'error': f'Generic analysis failed: {str(e)}'}

    def _is_binary_file(self, file_path: str) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
            return b'\0' in chunk
        except:
            return True

    def _estimate_model_parameters(self, file_size_bytes: int) -> int:
        """Estimate number of parameters from file size"""
        # Rough estimation: 4 bytes per float32 parameter
        return file_size_bytes // 4

    def _store_file_content(self, file_path: str, content: str, content_type: str, method: str):
        """Store extracted file content for search"""
        # This would be implemented to store in file_content table
        pass

    def _convert_file(self, file_info: Dict) -> Dict:
        """Convert file to different format"""
        file_path = file_info['stored_path']
        file_type = file_info['file_type']
        file_format = file_info['file_format']

        conversion_result = {
            'converted_at': datetime.now().isoformat(),
            'original_format': file_format,
            'conversions': []
        }

        try:
            if file_type == 'image':
                conversion_result.update(self._convert_image(file_path))
            elif file_type == 'document':
                conversion_result.update(self._convert_document(file_path))
            elif file_type == 'dataset':
                conversion_result.update(self._convert_dataset(file_path))
            else:
                conversion_result['note'] = f'No conversion available for {file_type}'

        except Exception as e:
            return {'error': f'Conversion failed: {str(e)}'}

        return conversion_result

    def _convert_image(self, file_path: str) -> Dict:
        """Convert image to different formats"""
        try:
            from PIL import Image

            conversions = []
            base_path = Path(file_path).with_suffix('')

            with Image.open(file_path) as img:
                # Convert to common formats
                formats = [('png', 'PNG'), ('jpg', 'JPEG'), ('webp', 'WEBP')]

                for ext, format_name in formats:
                    if not file_path.endswith(f'.{ext}'):
                        output_path = f"{base_path}.{ext}"

                        if format_name == 'JPEG' and img.mode in ('RGBA', 'LA'):
                            # Convert to RGB for JPEG
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                            rgb_img.save(output_path, format_name, quality=95)
                        else:
                            img.save(output_path, format_name)

                        conversions.append({
                            'format': ext,
                            'path': output_path,
                            'size_bytes': os.path.getsize(output_path)
                        })

            return {'conversions': conversions}

        except ImportError:
            return {'error': 'PIL/Pillow not available for image conversion'}
        except Exception as e:
            return {'error': f'Image conversion failed: {str(e)}'}

    def _convert_document(self, file_path: str) -> Dict:
        """Convert document to text format"""
        try:
            file_ext = Path(file_path).suffix.lower()
            base_path = Path(file_path).with_suffix('')

            if file_ext == '.pdf':
                # Convert PDF to text
                try:
                    import PyPDF2

                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ''
                        for page in reader.pages:
                            text += page.extract_text() + '\n'

                    text_path = f"{base_path}.txt"
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                    return {
                        'conversions': [{
                            'format': 'txt',
                            'path': text_path,
                            'size_bytes': os.path.getsize(text_path),
                            'pages_extracted': len(reader.pages)
                        }]
                    }

                except ImportError:
                    return {'error': 'PyPDF2 not available for PDF conversion'}

            return {'note': f'No conversion available for {file_ext}'}

        except Exception as e:
            return {'error': f'Document conversion failed: {str(e)}'}

    def _convert_dataset(self, file_path: str) -> Dict:
        """Convert dataset to different formats"""
        try:
            file_ext = Path(file_path).suffix.lower()
            base_path = Path(file_path).with_suffix('')
            conversions = []

            if file_ext == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)

                    # Convert to JSON
                    json_path = f"{base_path}.json"
                    df.to_json(json_path, orient='records', indent=2)
                    conversions.append({
                        'format': 'json',
                        'path': json_path,
                        'size_bytes': os.path.getsize(json_path)
                    })

                    # Convert to JSONL
                    jsonl_path = f"{base_path}.jsonl"
                    df.to_json(jsonl_path, orient='records', lines=True)
                    conversions.append({
                        'format': 'jsonl',
                        'path': jsonl_path,
                        'size_bytes': os.path.getsize(jsonl_path)
                    })

                except ImportError:
                    return {'error': 'pandas not available for dataset conversion'}

            elif file_ext == '.json':
                try:
                    import pandas as pd
                    df = pd.read_json(file_path)

                    # Convert to CSV
                    csv_path = f"{base_path}.csv"
                    df.to_csv(csv_path, index=False)
                    conversions.append({
                        'format': 'csv',
                        'path': csv_path,
                        'size_bytes': os.path.getsize(csv_path)
                    })

                except ImportError:
                    return {'error': 'pandas not available for dataset conversion'}

            return {'conversions': conversions}

        except Exception as e:
            return {'error': f'Dataset conversion failed: {str(e)}'}

    def _extract_content(self, file_info: Dict) -> Dict:
        """Extract searchable content from file"""
        file_path = file_info['stored_path']
        file_type = file_info['file_type']

        extraction_result = {
            'extracted_at': datetime.now().isoformat(),
            'extraction_method': 'auto',
            'content_types': []
        }

        try:
            if file_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                extraction_result['text_content'] = content
                extraction_result['word_count'] = len(content.split())
                extraction_result['content_types'].append('text')

            elif file_type == 'document':
                # Extract text from documents
                extracted_text = self._extract_document_text(file_path)
                if extracted_text:
                    extraction_result['text_content'] = extracted_text
                    extraction_result['word_count'] = len(extracted_text.split())
                    extraction_result['content_types'].append('text')

            elif file_type == 'config':
                # Extract configuration as structured data
                config_data = self._extract_config_data(file_path)
                if config_data:
                    extraction_result['config_data'] = config_data
                    extraction_result['content_types'].append('config')

            return extraction_result

        except Exception as e:
            return {'error': f'Content extraction failed: {str(e)}'}

    def _extract_document_text(self, file_path: str) -> Optional[str]:
        """Extract text from document files"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                return text
            except ImportError:
                return None

        return None

    def _extract_config_data(self, file_path: str) -> Optional[Dict]:
        """Extract structured data from config files"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(file_path, 'r') as f:
                        return yaml.safe_load(f)
                except ImportError:
                    return None
        except:
            return None

        return None

    def get_file_info(self, file_id: int) -> Optional[Dict]:
        """Get complete file information"""
        file_info = self.db.execute_query(
            "SELECT * FROM file_registry WHERE id = ?", (file_id,)
        )

        if not file_info:
            return None

        file_info = file_info[0]

        # Parse JSON fields
        if file_info['metadata']:
            file_info['metadata'] = json.loads(file_info['metadata'])
        if file_info['tags']:
            file_info['tags'] = json.loads(file_info['tags'])

        # Get processing jobs
        jobs = self.db.execute_query("""
            SELECT * FROM file_processing_jobs
            WHERE file_id = ?
            ORDER BY id DESC
        """, (file_id,))

        file_info['processing_jobs'] = jobs

        # Get relationships
        relationships = self.db.execute_query("""
            SELECT fr.*, f.filename as target_filename
            FROM file_relationships fr
            JOIN file_registry f ON fr.target_file_id = f.id
            WHERE fr.source_file_id = ?
        """, (file_id,))

        file_info['relationships'] = relationships

        return file_info

    def list_files(
        self,
        file_type: str = None,
        file_format: str = None,
        processing_status: str = None,
        created_by: int = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List files with filtering options"""

        conditions = ["is_active = 1"]
        params = []

        if file_type:
            conditions.append("file_type = ?")
            params.append(file_type)

        if file_format:
            conditions.append("file_format = ?")
            params.append(file_format)

        if processing_status:
            conditions.append("processing_status = ?")
            params.append(processing_status)

        if created_by:
            conditions.append("created_by = ?")
            params.append(created_by)

        where_clause = " WHERE " + " AND ".join(conditions)
        params.extend([limit, offset])

        query = f"""
            SELECT * FROM file_registry
            {where_clause}
            ORDER BY uploaded_at DESC
            LIMIT ? OFFSET ?
        """

        files = self.db.execute_query(query, tuple(params))

        # Parse JSON fields
        for file_info in files:
            if file_info['metadata']:
                file_info['metadata'] = json.loads(file_info['metadata'])
            if file_info['tags']:
                file_info['tags'] = json.loads(file_info['tags'])

        return files

    def delete_file(self, file_id: int, remove_from_storage: bool = True) -> bool:
        """Delete file from database and optionally from storage"""
        try:
            # Get file info
            file_info = self.get_file_info(file_id)
            if not file_info:
                return False

            # Remove from storage if requested
            if remove_from_storage and file_info['stored_path']:
                try:
                    os.remove(file_info['stored_path'])
                except OSError:
                    pass  # File might already be deleted

            # Mark as inactive in database
            self.db.execute_update(
                "UPDATE file_registry SET is_active = 0 WHERE id = ?", (file_id,)
            )

            self.logger.info(f"Deleted file: {file_info['filename']} (ID: {file_id})")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {e}")
            return False
