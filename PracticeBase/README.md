# Python Environment Setup

Simply use the command "Python: Create Environment" in Cursor, and choose python 3.9.6. Then choose the requirements.txt file for installation.
This will automatically:

- Create a virtual environment
- Install all required dependencies
- Set up everything needed to run the tests

After the environment is created, you can run the tests with:

```bash
python test.py
```

## Required Packages

The project uses the following main packages:

- tensorflow>=2.15.0
- transformers>=4.36.0
- gensim>=4.3.0
- tf-keras>=2.15.0

## Testing the Installation

After installation, you can verify everything is working by running:

```bash
python test.py
```

This will:

1. Test gensim by loading a word embedding model
2. Test transformers with a sentiment analysis model
3. Test tensorflow with a simple matrix operation

## Troubleshooting

If you encounter any issues:

1. **SSL Warning**: If you see an SSL warning about LibreSSL, it's a known issue with urllib3 and doesn't affect functionality.

2. **Package Conflicts**: If you encounter package conflicts, try:

   ```bash
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

3. **Memory Issues**: If you run into memory issues while loading models, try using smaller models or increasing your system's swap space.

## Notes

- The first run of `test.py` will download necessary models, which might take some time depending on your internet connection.
- TensorFlow will show optimization warnings on first run - these are informational and don't affect functionality.
