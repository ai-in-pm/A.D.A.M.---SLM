# A.D.A.M. SLM Database System

🗄️ **Sophisticated SQLite database system for AI model lifecycle management**

## 📁 **Project Structure**

```
adam_slm_database/
├── 📊 Core System
│   ├── __init__.py                 # Package initialization
│   ├── database.py                 # Core database operations
│   ├── manager.py                  # High-level database manager
│   ├── analytics.py               # Advanced analytics engine
│   ├── migrations.py              # Schema migration system
│   ├── models.py                   # Data model classes
│   ├── schema.sql                  # Complete database schema
│   ├── file_manager.py            # File management system
│   └── file_converter.py          # Universal file converter
│
├── 📁 databases/                   # Database files
│   ├── adamslm_sophisticated.sqlite    # Main production database
│   ├── adam_slm_demo.db               # Demo database
│   ├── adamslm.sqlite                 # Original simple database
│   └── adamslm_backup_*.sqlite        # Database backups
│
├── 📋 demos/                       # Demonstration scripts
│   ├── demo.py                     # Main database demo
│   ├── file_import_demo.py         # File management demo
│   └── integration_example.py     # Integration examples
│
├── 📚 docs/                        # Documentation
│   ├── README.md                   # Main documentation
│   ├── DATABASE_SUMMARY.md         # Database system overview
│   └── FILE_MANAGEMENT_SUMMARY.md  # File management guide
│
├── 🔧 scripts/                     # Utility scripts
│   ├── import_file.sh              # Unix/Linux import script
│   ├── import_file.bat             # Windows import script
│   ├── update_schema.py            # Schema update utility
│   └── upgrade_database.py         # Database upgrade tool
│
├── 🛠️ tools/                       # Command-line tools
│   └── file_import_tool.py         # File import CLI tool
│
├── 🧪 tests/                       # Test files and data
│   ├── test_file.txt               # Sample text file
│   ├── test_data.csv               # Sample CSV data
│   └── test_data.json              # Sample JSON data
│
└── 📦 file_storage/                # Organized file storage
    ├── text/2025/07/               # Date-organized storage
    ├── dataset/2025/07/
    ├── config/2025/07/
    ├── image/2025/07/
    ├── document/2025/07/
    ├── model/2025/07/
    └── temp/                       # Temporary files
```

## 🚀 **Quick Start**

### 1. **Run Database Demo**
```bash
cd demos
python demo.py
```

### 2. **Import Files**
```bash
cd tools
python file_import_tool.py ../tests/test_file.txt -v
```

### 3. **File Management Demo**
```bash
cd demos
python file_import_demo.py
```

### 4. **Integration Examples**
```bash
cd demos
python integration_example.py
```

## 📊 **Core Features**

### 🗄️ **Database System**
- **15+ interconnected tables** with proper relationships
- **User management** with roles and sessions
- **Model versioning** and lineage tracking
- **Training run** monitoring and metrics
- **Dataset management** with splits and statistics
- **Experiment tracking** and comparison
- **Performance benchmarking** and analytics

### 📁 **File Management**
- **Universal file support** - 20+ file formats
- **Automatic conversion** between formats
- **Content extraction** and analysis
- **Organized storage** with version control
- **Processing pipeline** with async jobs
- **Search capabilities** across content and metadata

### 🛠️ **Tools & Scripts**
- **Cross-platform CLI tools** for file import
- **Comprehensive demos** showcasing all features
- **Migration system** for database updates
- **Analytics engine** for insights and reporting

## 📋 **Supported File Types**

| Category | Formats | Features |
|----------|---------|----------|
| 📝 **Text** | txt, md, rst, html, log | Content extraction, analysis |
| 📊 **Data** | csv, json, jsonl, xml, yaml | Schema analysis, conversion |
| 🖼️ **Images** | png, jpg, gif, bmp, webp | Format conversion, metadata |
| 📄 **Documents** | pdf, doc, docx, rtf | Text extraction, processing |
| 🤖 **Models** | pt, pth, ckpt, safetensors | Parameter analysis, metadata |
| ⚙️ **Config** | json, yaml, toml, ini | Structure analysis, validation |
| 📦 **Archives** | zip, tar, gz, bz2, 7z | Extraction, content listing |
| 💻 **Code** | py, js, cpp, java, go | Syntax analysis, metrics |

## 🎯 **Usage Examples**

### **Database Operations**
```python
from database import AdamSLMDatabase
from manager import DatabaseManager

# Initialize
db = AdamSLMDatabase("databases/adamslm_sophisticated.sqlite")
manager = DatabaseManager("databases/adamslm_sophisticated.sqlite")

# Get dashboard stats
stats = manager.get_dashboard_stats()
print(f"Models: {stats['models']['total_models']}")
```

### **File Management**
```python
from file_manager import FileManager

# Initialize file manager
fm = FileManager(db)

# Import file
file_id = fm.register_file(
    file_path="data.csv",
    description="Training dataset",
    tags=["training", "v1.0"]
)

# List files
files = fm.list_files(file_type="dataset")
```

### **File Conversion**
```bash
# Convert CSV to JSON
python tools/file_import_tool.py data.csv -f json

# Import with metadata
python tools/file_import_tool.py model.pt \
  -t model \
  -d "Trained A.D.A.M. SLM v1.0" \
  -T "production,shakespeare"
```

## 📈 **Analytics & Reporting**

```python
from analytics import DatabaseAnalytics

analytics = DatabaseAnalytics(db)

# Model performance comparison
comparison = analytics.get_model_performance_comparison([1, 2, 3])

# Training trends
trends = analytics.get_training_trends(days=30)

# System performance
performance = analytics.get_system_performance_report(hours=24)
```

## 🔧 **Database Management**

### **Schema Updates**
```bash
cd scripts
python update_schema.py
```

### **Database Migration**
```bash
cd scripts
python upgrade_database.py
```

### **Backup & Restore**
```python
from migrations import DatabaseMigrations

migrations = DatabaseMigrations(db)
migrations.backup_database("backup.db")
```

## 🎯 **Key Benefits**

### 🏗️ **Enterprise-Grade Architecture**
- **Scalable design** supporting large-scale operations
- **Data integrity** with foreign key constraints
- **Performance optimization** with strategic indexing
- **Security features** with user management and audit trails

### 🔄 **Complete ML Lifecycle**
- **Data ingestion** with automatic processing
- **Model training** with comprehensive tracking
- **Experiment management** with comparison tools
- **Deployment monitoring** with performance analytics

### 🛠️ **Developer-Friendly**
- **Simple APIs** for common operations
- **Comprehensive documentation** with examples
- **Cross-platform tools** for all environments
- **Extensible architecture** for custom features

## 📚 **Documentation**

- **[Database System Overview](docs/DATABASE_SUMMARY.md)** - Complete system documentation
- **[File Management Guide](docs/FILE_MANAGEMENT_SUMMARY.md)** - File management features
- **[API Reference](docs/README.md)** - Detailed API documentation

## 🚀 **Production Ready**

The A.D.A.M. SLM database system is ready for:
- ✅ **Research environments** with experiment tracking
- ✅ **Production deployments** with monitoring and analytics
- ✅ **Team collaboration** with user management
- ✅ **Enterprise integration** with comprehensive APIs
- ✅ **Scalable operations** with optimized performance

---

**A.D.A.M. SLM Database System** - Where sophisticated data management meets cutting-edge AI! 🗄️✨
