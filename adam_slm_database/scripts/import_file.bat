@echo off
REM ADAM SLM Database File Import Script for Windows
REM Converts and imports files of all types into the sophisticated database

setlocal enabledelayedexpansion

REM Script configuration
set "SCRIPT_DIR=%~dp0"
set "DATABASE_DIR=%SCRIPT_DIR%"
set "PYTHON_SCRIPT=%DATABASE_DIR%file_import_tool.py"
set "LOG_FILE=%DATABASE_DIR%import.log"

REM Colors (using Windows color codes)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "PURPLE=[95m"
set "CYAN=[96m"
set "NC=[0m"

REM Function to print colored output
:print_header
echo %BLUE%================================%NC%
echo %BLUE%🗄️  ADAM SLM File Import Tool%NC%
echo %BLUE%================================%NC%
goto :eof

:print_success
echo %GREEN%✅ %~1%NC%
goto :eof

:print_error
echo %RED%❌ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠️  %~1%NC%
goto :eof

:print_info
echo %CYAN%ℹ️  %~1%NC%
goto :eof

:print_step
echo %PURPLE%📋 %~1%NC%
goto :eof

REM Function to show usage
:show_usage
echo Usage: %~nx0 [OPTIONS] ^<file_path^>
echo.
echo Import and convert files into ADAM SLM database
echo.
echo Options:
echo   -t TYPE        Override file type detection (text, dataset, image, model, etc.)
echo   -f FORMAT      Convert to specific format before import
echo   -d DESC        Add description for the file
echo   -T TAGS        Add comma-separated tags
echo   -u USER        Username for file ownership (default: admin)
echo   -c             Copy file to managed storage (default: true)
echo   -p             Process file immediately (default: true)
echo   -o DIR         Output directory for conversions
echo   -v             Verbose output
echo   -h             Show this help message
echo.
echo Examples:
echo   %~nx0 dataset.csv
echo   %~nx0 -f json dataset.csv
echo   %~nx0 -t model -d "Trained ADAM SLM" model.pt
echo   %~nx0 -T "research,experiment" -u alice data.jsonl
echo   %~nx0 -f png -o ./converted image.jpg
echo.
echo Supported file types:
echo   📝 Text: txt, md, rst, html, log
echo   📊 Data: csv, json, jsonl, xml, yaml, parquet
echo   🖼️  Images: png, jpg, jpeg, gif, bmp, webp, svg
echo   📄 Documents: pdf, doc, docx, rtf
echo   🤖 Models: pt, pth, ckpt, safetensors, onnx
echo   ⚙️  Config: json, yaml, yml, toml, ini
echo   📦 Archives: zip, tar, gz, bz2, 7z
echo   💻 Code: py, js, cpp, java, go, rs
echo   🎵 Audio: mp3, wav, flac, ogg
echo   🎬 Video: mp4, avi, mov, mkv
goto :eof

REM Function to check dependencies
:check_dependencies
call :print_step "Checking dependencies..."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is required but not installed"
    exit /b 1
)

REM Check if the Python script exists
if not exist "%PYTHON_SCRIPT%" (
    call :print_warning "Python import tool not found, creating it..."
    call :create_python_tool
)

call :print_success "Dependencies checked"
goto :eof

REM Function to create the Python import tool
:create_python_tool
(
echo #!/usr/bin/env python3
echo """
echo ADAM SLM Database File Import Tool
echo Command-line interface for importing files
echo """
echo.
echo import sys
echo import os
echo import argparse
echo import json
echo from pathlib import Path
echo.
echo # Add current directory to path
echo sys.path.append^(os.path.dirname^(__file__^)^)
echo.
echo try:
echo     from database import AdamSLMDatabase
echo     from file_manager import FileManager
echo     from file_converter import FileConverter
echo except ImportError as e:
echo     print^(f"❌ Error importing modules: {e}"^)
echo     print^("Make sure you're running from the adam_slm_database directory"^)
echo     sys.exit^(1^)
echo.
echo.
echo def main^(^):
echo     parser = argparse.ArgumentParser^(description='Import files into ADAM SLM database'^)
echo     parser.add_argument^('file_path', help='Path to file to import'^)
echo     parser.add_argument^('-t', '--type', help='Override file type detection'^)
echo     parser.add_argument^('-f', '--format', help='Convert to specific format'^)
echo     parser.add_argument^('-d', '--description', help='File description'^)
echo     parser.add_argument^('-T', '--tags', help='Comma-separated tags'^)
echo     parser.add_argument^('-u', '--user', default='admin', help='Username for ownership'^)
echo     parser.add_argument^('-c', '--copy', action='store_true', default=True, help='Copy to storage'^)
echo     parser.add_argument^('-p', '--process', action='store_true', default=True, help='Process immediately'^)
echo     parser.add_argument^('-o', '--output', help='Output directory for conversions'^)
echo     parser.add_argument^('-v', '--verbose', action='store_true', help='Verbose output'^)
echo     
echo     args = parser.parse_args^(^)
echo     
echo     # Check if file exists
echo     if not os.path.exists^(args.file_path^):
echo         print^(f"❌ File not found: {args.file_path}"^)
echo         return 1
echo         
echo     try:
echo         # Initialize database and file manager
echo         db = AdamSLMDatabase^("adamslm_sophisticated.sqlite"^)
echo         file_manager = FileManager^(db^)
echo         
echo         # Get user ID
echo         user = db.get_user^(username=args.user^)
echo         if not user:
echo             print^(f"❌ User not found: {args.user}"^)
echo             return 1
echo             
echo         user_id = user['id']
echo         
echo         # Parse tags
echo         tags = []
echo         if args.tags:
echo             tags = [tag.strip^(^) for tag in args.tags.split^(','^)]
echo             
echo         # Convert file if format specified
echo         input_path = args.file_path
echo         if args.format:
echo             print^(f"🔄 Converting to {args.format}..."^)
echo             converter = FileConverter^(^)
echo             
echo             # Determine output path
echo             if args.output:
echo                 os.makedirs^(args.output, exist_ok=True^)
echo                 output_path = os.path.join^(args.output, f"{Path^(args.file_path^).stem}.{args.format}"^)
echo             else:
echo                 output_path = f"{Path^(args.file_path^).stem}.{args.format}"
echo                 
echo             result = converter.convert_file^(input_path, output_path, args.format^)
echo             
echo             if result['success']:
echo                 print^(f"✅ Converted successfully: {output_path}"^)
echo                 input_path = output_path
echo             else:
echo                 print^(f"❌ Conversion failed: {result['error']}"^)
echo                 return 1
echo                 
echo         # Import file
echo         print^(f"📥 Importing file: {input_path}"^)
echo         
echo         file_id = file_manager.register_file^(
echo             file_path=input_path,
echo             file_type=args.type,
echo             description=args.description,
echo             tags=tags,
echo             created_by=user_id,
echo             copy_to_storage=args.copy,
echo             process_immediately=args.process
echo         ^)
echo         
echo         print^(f"✅ File imported successfully!"^)
echo         print^(f"   File ID: {file_id}"^)
echo         
echo         # Show file info if verbose
echo         if args.verbose:
echo             file_info = file_manager.get_file_info^(file_id^)
echo             print^(f"   Type: {file_info['file_type']}"^)
echo             print^(f"   Format: {file_info['file_format']}"^)
echo             print^(f"   Size: {file_info['file_size_bytes']:,} bytes"^)
echo             print^(f"   Stored: {file_info['stored_path']}"^)
echo             
echo         return 0
echo         
echo     except Exception as e:
echo         print^(f"❌ Import failed: {e}"^)
echo         if args.verbose:
echo             import traceback
echo             traceback.print_exc^(^)
echo         return 1
echo.
echo.
echo if __name__ == "__main__":
echo     sys.exit^(main^(^)^)
) > "%PYTHON_SCRIPT%"

call :print_success "Created Python import tool"
goto :eof

REM Main script
call :print_header

REM Parse command line arguments
set "FILE_PATH="
set "FILE_TYPE="
set "CONVERT_FORMAT="
set "DESCRIPTION="
set "TAGS="
set "USERNAME=admin"
set "COPY_TO_STORAGE=true"
set "PROCESS_IMMEDIATELY=true"
set "OUTPUT_DIR="
set "VERBOSE=false"

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="-h" goto :show_help
if "%~1"=="-t" (
    set "FILE_TYPE=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-f" (
    set "CONVERT_FORMAT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-d" (
    set "DESCRIPTION=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-T" (
    set "TAGS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-u" (
    set "USERNAME=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-c" (
    set "COPY_TO_STORAGE=true"
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set "PROCESS_IMMEDIATELY=true"
    shift
    goto :parse_args
)
if "%~1"=="-o" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-v" (
    set "VERBOSE=true"
    shift
    goto :parse_args
)
REM If no flag, assume it's the file path
if "%FILE_PATH%"=="" (
    set "FILE_PATH=%~1"
    shift
    goto :parse_args
)
shift
goto :parse_args

:args_done

REM Check if file path provided
if "%FILE_PATH%"=="" (
    call :print_error "No file path provided"
    call :show_usage
    exit /b 1
)

REM Check dependencies
call :check_dependencies
if errorlevel 1 exit /b 1

REM Validate file
call :print_step "Validating file..."
if not exist "%FILE_PATH%" (
    call :print_error "File does not exist: %FILE_PATH%"
    exit /b 1
)
call :print_success "File validation passed"

REM Build Python command
set "PYTHON_CMD=python "%PYTHON_SCRIPT%" "%FILE_PATH%""

if not "%FILE_TYPE%"=="" set "PYTHON_CMD=%PYTHON_CMD% -t "%FILE_TYPE%""
if not "%CONVERT_FORMAT%"=="" set "PYTHON_CMD=%PYTHON_CMD% -f "%CONVERT_FORMAT%""
if not "%DESCRIPTION%"=="" set "PYTHON_CMD=%PYTHON_CMD% -d "%DESCRIPTION%""
if not "%TAGS%"=="" set "PYTHON_CMD=%PYTHON_CMD% -T "%TAGS%""
if not "%USERNAME%"=="" set "PYTHON_CMD=%PYTHON_CMD% -u "%USERNAME%""
if "%COPY_TO_STORAGE%"=="true" set "PYTHON_CMD=%PYTHON_CMD% -c"
if "%PROCESS_IMMEDIATELY%"=="true" set "PYTHON_CMD=%PYTHON_CMD% -p"
if not "%OUTPUT_DIR%"=="" set "PYTHON_CMD=%PYTHON_CMD% -o "%OUTPUT_DIR%""
if "%VERBOSE%"=="true" set "PYTHON_CMD=%PYTHON_CMD% -v"

REM Execute Python import tool
call :print_step "Executing import..."

REM Log the command
echo %date% %time%: Importing %FILE_PATH% >> "%LOG_FILE%"

%PYTHON_CMD%
if errorlevel 1 (
    call :print_error "Import failed!"
    echo %date% %time%: Import failed for %FILE_PATH% >> "%LOG_FILE%"
    exit /b 1
) else (
    call :print_success "Import completed successfully!"
    echo %date% %time%: Import successful for %FILE_PATH% >> "%LOG_FILE%"
    
    echo.
    call :print_info "Next steps:"
    echo    🔍 View files: python demo.py
    echo    📊 Check database: python integration_example.py
    echo    🌐 Start web interface: python web_interface.py (if available)
)

goto :eof

:show_help
call :show_usage
exit /b 0
