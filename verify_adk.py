import adk

try:
    print(f"ADK version: {adk.__version__}")
    print("ADK installed successfully!")
except ImportError as e:
    print(f"Error importing ADK: {e}")
    print("ADK might not be installed correctly in the current environment.")