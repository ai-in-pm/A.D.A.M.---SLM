#!/usr/bin/env python3
"""
ADAM SLM Database File Import Tool
Command-line interface for importing files
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from database import AdamSLMDatabase
    from file_manager import FileManager
    from file_converter import FileConverter
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the adam_slm_database directory")
    sys.exit(1)


def print_header():
    """Print tool header"""
    print("🗄️ ADAM SLM Database File Import Tool")
    print("=" * 50)


def print_success(message):
    """Print success message"""
    print(f"✅ {message}")


def print_error(message):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")


def print_step(message):
    """Print step message"""
    print(f"📋 {message}")


def detect_file_type(file_path):
    """Detect file type from extension"""
    extension = Path(file_path).suffix.lower()[1:]  # Remove dot
    
    type_mapping = {
        # Text files
        'txt': 'text', 'md': 'text', 'rst': 'text', 'log': 'text',
        # Data files
        'csv': 'dataset', 'jsonl': 'dataset', 'parquet': 'dataset', 'tsv': 'dataset',
        # Config files
        'json': 'config', 'yaml': 'config', 'yml': 'config', 'toml': 'config', 'ini': 'config',
        # Images
        'png': 'image', 'jpg': 'image', 'jpeg': 'image', 'gif': 'image', 'bmp': 'image', 'webp': 'image', 'svg': 'image',
        # Documents
        'pdf': 'document', 'doc': 'document', 'docx': 'document', 'rtf': 'document', 'html': 'document', 'htm': 'document',
        # Models
        'pt': 'model', 'pth': 'model', 'ckpt': 'model', 'safetensors': 'model', 'onnx': 'model', 'bin': 'model',
        # Code
        'py': 'code', 'js': 'code', 'cpp': 'code', 'java': 'code', 'go': 'code', 'rs': 'code', 'c': 'code', 'h': 'code',
        # Archives
        'zip': 'archive', 'tar': 'archive', 'gz': 'archive', 'bz2': 'archive', '7z': 'archive', 'rar': 'archive',
        # Audio/Video
        'mp3': 'audio', 'wav': 'audio', 'flac': 'audio', 'ogg': 'audio',
        'mp4': 'video', 'avi': 'video', 'mov': 'video', 'mkv': 'video'
    }
    
    return type_mapping.get(extension, 'binary')


def show_file_info(file_path):
    """Show file information"""
    if not os.path.exists(file_path):
        print_error(f"File not found: {file_path}")
        return False
        
    file_size = os.path.getsize(file_path)
    file_type = detect_file_type(file_path)
    
    print_info("File Information:")
    print(f"   📁 Path: {file_path}")
    print(f"   📊 Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"   🏷️  Detected Type: {file_type}")
    print(f"   📅 Modified: {os.path.getmtime(file_path)}")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Import files into ADAM SLM database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python file_import_tool.py dataset.csv
  python file_import_tool.py -f json dataset.csv
  python file_import_tool.py -t model -d "Trained ADAM SLM" model.pt
  python file_import_tool.py -T "research,experiment" -u alice data.jsonl
  python file_import_tool.py --format png --output ./converted image.jpg

Supported file types:
  📝 Text: txt, md, rst, html, log
  📊 Data: csv, json, jsonl, xml, yaml, parquet
  🖼️  Images: png, jpg, jpeg, gif, bmp, webp, svg
  📄 Documents: pdf, doc, docx, rtf
  🤖 Models: pt, pth, ckpt, safetensors, onnx
  ⚙️  Config: json, yaml, yml, toml, ini
  📦 Archives: zip, tar, gz, bz2, 7z
  💻 Code: py, js, cpp, java, go, rs
  🎵 Audio: mp3, wav, flac, ogg
  🎬 Video: mp4, avi, mov, mkv
        """
    )
    
    parser.add_argument('file_path', help='Path to file to import')
    parser.add_argument('-t', '--type', help='Override file type detection')
    parser.add_argument('-f', '--format', help='Convert to specific format before import')
    parser.add_argument('-d', '--description', help='File description')
    parser.add_argument('-T', '--tags', help='Comma-separated tags')
    parser.add_argument('-u', '--user', default='admin', help='Username for ownership (default: admin)')
    parser.add_argument('-c', '--copy', action='store_true', default=True, help='Copy to managed storage (default: true)')
    parser.add_argument('-p', '--process', action='store_true', default=True, help='Process immediately (default: true)')
    parser.add_argument('-o', '--output', help='Output directory for conversions')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--info', action='store_true', help='Show file info only (no import)')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print_error(f"File not found: {args.file_path}")
        return 1
    
    # Show file info if requested
    if args.info:
        return 0 if show_file_info(args.file_path) else 1
    
    # Show file info if verbose
    if args.verbose:
        show_file_info(args.file_path)
        
    try:
        print_step("Initializing database connection...")
        
        # Initialize database and file manager
        db = AdamSLMDatabase("../databases/adamslm_sophisticated.sqlite")
        file_manager = FileManager(db)
        
        print_success("Database connection established")
        
        # Get user ID
        user = db.get_user(username=args.user)
        if not user:
            print_error(f"User not found: {args.user}")
            return 1
            
        user_id = user['id']
        print_info(f"Using user: {args.user} (ID: {user_id})")
        
        # Parse tags
        tags = []
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(',')]
            print_info(f"Tags: {', '.join(tags)}")
            
        # Convert file if format specified
        input_path = args.file_path
        if args.format:
            print_step(f"Converting to {args.format}...")
            converter = FileConverter()
            
            # Determine output path
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                output_path = os.path.join(args.output, f"{Path(args.file_path).stem}.{args.format}")
            else:
                output_path = f"{Path(args.file_path).stem}.{args.format}"
                
            result = converter.convert_file(input_path, output_path, args.format)
            
            if result['success']:
                print_success(f"Converted successfully: {output_path}")
                if args.verbose:
                    print_info(f"Output size: {result.get('output_size', 0):,} bytes")
                input_path = output_path
            else:
                print_error(f"Conversion failed: {result['error']}")
                return 1
                
        # Import file
        print_step(f"Importing file: {os.path.basename(input_path)}")
        
        file_id = file_manager.register_file(
            file_path=input_path,
            file_type=args.type,
            description=args.description,
            tags=tags,
            created_by=user_id,
            copy_to_storage=args.copy,
            process_immediately=args.process
        )
        
        print_success("File imported successfully!")
        print_info(f"File ID: {file_id}")
        
        # Show file info if verbose
        if args.verbose:
            file_info = file_manager.get_file_info(file_id)
            print_info("Import Details:")
            print(f"   🏷️  Type: {file_info['file_type']}")
            print(f"   📋 Format: {file_info['file_format']}")
            print(f"   📊 Size: {file_info['file_size_bytes']:,} bytes")
            print(f"   📁 Stored: {file_info['stored_path']}")
            print(f"   ⚡ Status: {file_info['processing_status']}")
            
            if file_info['processing_jobs']:
                latest_job = file_info['processing_jobs'][0]
                print(f"   🔄 Job: {latest_job['job_type']} ({latest_job['job_status']})")
                
        print_info("Next Steps:")
        print("   🔍 View files: python file_import_demo.py")
        print("   📊 Check database: python integration_example.py")
        print("   🌐 List files: python -c \"from file_manager import *; fm = FileManager(AdamSLMDatabase('adamslm_sophisticated.sqlite')); print(len(fm.list_files()), 'files in database')\"")
        
        return 0
        
    except Exception as e:
        print_error(f"Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
