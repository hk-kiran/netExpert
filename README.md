# netExpert

Model = llama3.2

### Setup

1. Setup conda environment
    ```sh
    conda create --name cs260 --file ./requirements.txt

    conda activate cs260
    ```

2. Install ollama on your system
3. Setup database `python prepare_db.py`

### Run the app

```sh
streamlit run streamlit.py --browser.serverAddress localhost
```

The app will be exposed on port 8501 
