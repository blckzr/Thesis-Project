# deepfake-detection

For bimodal hybrid detection of deepfake videos.

# Installing

Run `uv sync` to install all dependencies.
If you have multiple nonsensical errors after, run `source .venv/bin/activate` and run your code editor of choice in that terminal.

# Development

`api.py` is where the actual development of the backend will be done, while `prototype.py` will be used for testing the models.
Any other one-off scripts will be put under the `scripts/` folder (dataset parsing, scraping websites into a JSON, etc.)

To run `api.py`, do `uv run api.py`.

To run `prototype.py`, do `streamlit run prototype.py`.
