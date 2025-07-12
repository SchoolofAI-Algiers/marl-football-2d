# Project Setup

Follow these steps to set up your development environment:

## 1. Create a Virtual Environment
Run the following command to create a virtual environment:

```sh
uv venv
```

## 2. Activate the Virtual Environment
- **Windows:**
  ```sh
  .venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source .venv/bin/activate
  ```

## 3. Install Dependencies
Run the following command to install all required dependencies:

```sh
uv pip install -e .
```

If you want to use your own GPU for training, install Torch with CUDA support

```sh
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Your environment is now set up and ready for development!
