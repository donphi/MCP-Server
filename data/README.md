# 📁 Data Directory

Place your Markdown (.md) and text (.txt) files in this directory to be processed by the MCP Server pipeline.

## 📥 Adding Files

Simply copy your files to this directory before running the pipeline. The pipeline will process all supported files in this directory and its subdirectories.

## 📄 Supported File Types

By default, the following file types are supported:
- Markdown files (.md)
- Text files (.txt)

You can configure additional file extensions in the `config/pipeline_config.json` file or by setting the `SUPPORTED_EXTENSIONS` environment variable.

## ⭐ Priority Files

You can configure priority files in the `config/pipeline_config.json` file. These files will be processed first if they exist.

## 🔄 File Processing

Files are processed in the following order:
1. Priority files (if configured)
2. All other supported files (in no particular order)

Once a file has been processed, it will not be processed again unless it changes.

---

Created with ❤️ by [donphi](https://github.com/donphi)