#!/bin/bash

# ADAM SLM Database File Import Script
# Converts and imports files of all types into the sophisticated database

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATABASE_DIR="$SCRIPT_DIR"
PYTHON_SCRIPT="$DATABASE_DIR/file_import_tool.py"
LOG_FILE="$DATABASE_DIR/import.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}🗄️  ADAM SLM File Import Tool${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${PURPLE}📋 $1${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <file_path>"
    echo ""
    echo "Import and convert files into ADAM SLM database"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Override file type detection (text, dataset, image, model, etc.)"
    echo "  -f, --format FORMAT    Convert to specific format before import"
    echo "  -d, --description DESC Add description for the file"
    echo "  -T, --tags TAGS        Add comma-separated tags"
    echo "  -u, --user USER        Username for file ownership (default: admin)"
    echo "  -c, --copy             Copy file to managed storage (default: true)"
    echo "  -p, --process          Process file immediately (default: true)"
    echo "  -o, --output DIR       Output directory for conversions"
    echo "  -v, --verbose          Verbose output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dataset.csv"
    echo "  $0 -f json dataset.csv"
    echo "  $0 -t model -d \"Trained ADAM SLM\" model.pt"
    echo "  $0 -T \"research,experiment\" -u alice data.jsonl"
    echo "  $0 --format png --output ./converted image.jpg"
    echo ""
    echo "Supported file types:"
    echo "  📝 Text: txt, md, rst, html, log"
    echo "  📊 Data: csv, json, jsonl, xml, yaml, parquet"
    echo "  🖼️  Images: png, jpg, jpeg, gif, bmp, webp, svg"
    echo "  📄 Documents: pdf, doc, docx, rtf"
    echo "  🤖 Models: pt, pth, ckpt, safetensors, onnx"
    echo "  ⚙️  Config: json, yaml, yml, toml, ini"
    echo "  📦 Archives: zip, tar, gz, bz2, 7z"
    echo "  💻 Code: py, js, cpp, java, go, rs"
    echo "  🎵 Audio: mp3, wav, flac, ogg"
    echo "  🎬 Video: mp4, avi, mov, mkv"
}

# Function to check dependencies
check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if the Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_warning "Python import tool not found, creating it..."
        create_python_tool
    fi
    
    print_success "Dependencies checked"
}

# Function to create the Python import tool
create_python_tool() {
    cat > "$PYTHON_SCRIPT" << 'EOF'
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

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from database import AdamSLMDatabase
    from file_manager import FileManager
    from file_converter import FileConverter
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the adam_slm_database directory")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Import files into ADAM SLM database')
    parser.add_argument('file_path', help='Path to file to import')
    parser.add_argument('-t', '--type', help='Override file type detection')
    parser.add_argument('-f', '--format', help='Convert to specific format')
    parser.add_argument('-d', '--description', help='File description')
    parser.add_argument('-T', '--tags', help='Comma-separated tags')
    parser.add_argument('-u', '--user', default='admin', help='Username for ownership')
    parser.add_argument('-c', '--copy', action='store_true', default=True, help='Copy to storage')
    parser.add_argument('-p', '--process', action='store_true', default=True, help='Process immediately')
    parser.add_argument('-o', '--output', help='Output directory for conversions')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"❌ File not found: {args.file_path}")
        return 1
        
    try:
        # Initialize database and file manager
        db = AdamSLMDatabase("adamslm_sophisticated.sqlite")
        file_manager = FileManager(db)
        
        # Get user ID
        user = db.get_user(username=args.user)
        if not user:
            print(f"❌ User not found: {args.user}")
            return 1
            
        user_id = user['id']
        
        # Parse tags
        tags = []
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(',')]
            
        # Convert file if format specified
        input_path = args.file_path
        if args.format:
            print(f"🔄 Converting to {args.format}...")
            converter = FileConverter()
            
            # Determine output path
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                output_path = os.path.join(args.output, f"{Path(args.file_path).stem}.{args.format}")
            else:
                output_path = f"{Path(args.file_path).stem}.{args.format}"
                
            result = converter.convert_file(input_path, output_path, args.format)
            
            if result['success']:
                print(f"✅ Converted successfully: {output_path}")
                input_path = output_path
            else:
                print(f"❌ Conversion failed: {result['error']}")
                return 1
                
        # Import file
        print(f"📥 Importing file: {input_path}")
        
        file_id = file_manager.register_file(
            file_path=input_path,
            file_type=args.type,
            description=args.description,
            tags=tags,
            created_by=user_id,
            copy_to_storage=args.copy,
            process_immediately=args.process
        )
        
        print(f"✅ File imported successfully!")
        print(f"   File ID: {file_id}")
        
        # Show file info if verbose
        if args.verbose:
            file_info = file_manager.get_file_info(file_id)
            print(f"   Type: {file_info['file_type']}")
            print(f"   Format: {file_info['file_format']}")
            print(f"   Size: {file_info['file_size_bytes']:,} bytes")
            print(f"   Stored: {file_info['stored_path']}")
            
        return 0
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x "$PYTHON_SCRIPT"
    print_success "Created Python import tool"
}

# Function to validate file
validate_file() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        print_error "File does not exist: $file_path"
        return 1
    fi
    
    if [ ! -r "$file_path" ]; then
        print_error "File is not readable: $file_path"
        return 1
    fi
    
    # Check file size (warn if > 1GB)
    local file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo 0)
    local max_size=$((1024 * 1024 * 1024))  # 1GB
    
    if [ "$file_size" -gt "$max_size" ]; then
        print_warning "Large file detected ($(($file_size / 1024 / 1024))MB). Import may take time."
    fi
    
    return 0
}

# Function to detect file type
detect_file_type() {
    local file_path="$1"
    local extension="${file_path##*.}"
    extension=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
    
    case "$extension" in
        txt|md|rst|html|htm|log)
            echo "text"
            ;;
        csv|tsv|jsonl|parquet)
            echo "dataset"
            ;;
        json|yaml|yml|toml|ini)
            echo "config"
            ;;
        png|jpg|jpeg|gif|bmp|webp|svg|tiff)
            echo "image"
            ;;
        pdf|doc|docx|rtf|odt)
            echo "document"
            ;;
        pt|pth|ckpt|safetensors|onnx|bin)
            echo "model"
            ;;
        zip|tar|gz|bz2|7z|rar)
            echo "archive"
            ;;
        py|js|cpp|java|go|rs|c|h)
            echo "code"
            ;;
        mp3|wav|flac|ogg|m4a)
            echo "audio"
            ;;
        mp4|avi|mov|mkv|webm)
            echo "video"
            ;;
        *)
            echo "binary"
            ;;
    esac
}

# Function to show file info
show_file_info() {
    local file_path="$1"
    local file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo 0)
    local file_type=$(detect_file_type "$file_path")
    
    print_info "File Information:"
    echo "   📁 Path: $file_path"
    echo "   📊 Size: $(($file_size / 1024))KB ($(printf "%'d" $file_size) bytes)"
    echo "   🏷️  Type: $file_type"
    echo "   📅 Modified: $(stat -f%Sm "$file_path" 2>/dev/null || stat -c%y "$file_path" 2>/dev/null || echo 'Unknown')"
}

# Main function
main() {
    # Parse command line arguments
    POSITIONAL_ARGS=()
    FILE_TYPE=""
    CONVERT_FORMAT=""
    DESCRIPTION=""
    TAGS=""
    USERNAME="admin"
    COPY_TO_STORAGE=true
    PROCESS_IMMEDIATELY=true
    OUTPUT_DIR=""
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                FILE_TYPE="$2"
                shift 2
                ;;
            -f|--format)
                CONVERT_FORMAT="$2"
                shift 2
                ;;
            -d|--description)
                DESCRIPTION="$2"
                shift 2
                ;;
            -T|--tags)
                TAGS="$2"
                shift 2
                ;;
            -u|--user)
                USERNAME="$2"
                shift 2
                ;;
            -c|--copy)
                COPY_TO_STORAGE=true
                shift
                ;;
            -p|--process)
                PROCESS_IMMEDIATELY=true
                shift
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*|--*)
                print_error "Unknown option $1"
                show_usage
                exit 1
                ;;
            *)
                POSITIONAL_ARGS+=("$1")
                shift
                ;;
        esac
    done
    
    # Restore positional parameters
    set -- "${POSITIONAL_ARGS[@]}"
    
    # Check if file path provided
    if [ $# -eq 0 ]; then
        print_error "No file path provided"
        show_usage
        exit 1
    fi
    
    FILE_PATH="$1"
    
    # Print header
    print_header
    
    # Check dependencies
    check_dependencies
    
    # Validate file
    print_step "Validating file..."
    if ! validate_file "$FILE_PATH"; then
        exit 1
    fi
    print_success "File validation passed"
    
    # Show file info if verbose
    if [ "$VERBOSE" = true ]; then
        show_file_info "$FILE_PATH"
    fi
    
    # Build Python command arguments
    PYTHON_ARGS=("$FILE_PATH")
    
    if [ -n "$FILE_TYPE" ]; then
        PYTHON_ARGS+=("-t" "$FILE_TYPE")
    fi
    
    if [ -n "$CONVERT_FORMAT" ]; then
        PYTHON_ARGS+=("-f" "$CONVERT_FORMAT")
    fi
    
    if [ -n "$DESCRIPTION" ]; then
        PYTHON_ARGS+=("-d" "$DESCRIPTION")
    fi
    
    if [ -n "$TAGS" ]; then
        PYTHON_ARGS+=("-T" "$TAGS")
    fi
    
    if [ -n "$USERNAME" ]; then
        PYTHON_ARGS+=("-u" "$USERNAME")
    fi
    
    if [ "$COPY_TO_STORAGE" = true ]; then
        PYTHON_ARGS+=("-c")
    fi
    
    if [ "$PROCESS_IMMEDIATELY" = true ]; then
        PYTHON_ARGS+=("-p")
    fi
    
    if [ -n "$OUTPUT_DIR" ]; then
        PYTHON_ARGS+=("-o" "$OUTPUT_DIR")
    fi
    
    if [ "$VERBOSE" = true ]; then
        PYTHON_ARGS+=("-v")
    fi
    
    # Execute Python import tool
    print_step "Executing import..."
    
    # Log the command
    echo "$(date): Importing $FILE_PATH" >> "$LOG_FILE"
    
    if python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"; then
        print_success "Import completed successfully!"
        echo "$(date): Import successful for $FILE_PATH" >> "$LOG_FILE"
        
        # Show next steps
        echo ""
        print_info "Next steps:"
        echo "   🔍 View file: python3 -c \"from database import AdamSLMDatabase; from file_manager import FileManager; fm = FileManager(AdamSLMDatabase('adamslm_sophisticated.sqlite')); print(fm.list_files(limit=1))\""
        echo "   📊 Check database: python3 demo.py"
        echo "   🌐 Start web interface: python3 web_interface.py (if available)"
        
    else
        print_error "Import failed!"
        echo "$(date): Import failed for $FILE_PATH" >> "$LOG_FILE"
        exit 1
    fi
}

# Run main function
main "$@"
