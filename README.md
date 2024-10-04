# GDELT ML Pipeline

## Description

GDELT ML Pipeline is a machine learning project that processes GDELT (Global Database of Events, Language, and Tone) data to predict stock market movements. It combines GDELT event data with stock market information to create a predictive model.

## Features

- Fetches and processes GDELT data for a specified country and time range
- Retrieves stock market data for a given ticker symbol
- Performs feature engineering on GDELT and stock market data
- Generates embeddings for GDELT events using OpenAI's API
- Supports multiple machine learning models including LSTM, Temporal Fusion Transformer (TFT), Logistic Regression, Random Forest, and XGBoost
- Implements caching mechanisms for GDELT data and embeddings to improve performance
- Provides model evaluation and interpretability tools

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/gdelt_ml_pipeline.git
   cd gdelt_ml_pipeline
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Configuration

1. Create a `.env` file in the project root and add the following environment variables:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
   OPENAI_API_KEY=your_openai_api_key
   ```

2. Adjust the configuration parameters in `gdelt_ml_pipeline/config.py` as needed.

## Usage

Run the main script:

```
python -m gdelt_ml_pipeline.main
```

## Project Structure

- `gdelt_ml_pipeline/`
  - `config.py`: Configuration settings for the project
  - `data_loader.py`: Functions to fetch GDELT and stock market data
  - `preprocessor.py`: Data preprocessing functions
  - `feature_engineering.py`: Feature engineering and data preparation
  - `embeddings.py`: Functions to generate and manage embeddings
  - `models.py`: Machine learning model definitions
  - `train.py`: Model training procedures
  - `evaluate.py`: Model evaluation and interpretation tools
  - `main.py`: Main execution script

## Dependencies

Key dependencies include:
- PyTorch and PyTorch Forecasting
- Transformers
- yfinance
- pandas and numpy
- Google Cloud BigQuery
- OpenAI API

For a complete list of dependencies, refer to the `pyproject.toml` file.

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Raffay Rana] - [raffay.rana@gmail.com]

Project Link: [https://github.com/RRaffay/gdelt_ml_pipeline](https://github.com/RRaffay/gdelt_ml_pipeline)
