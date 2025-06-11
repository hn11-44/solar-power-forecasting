# ☀️ Solar Power Generation Forecasting

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hrmn1/solar-forecasting-app)

A complete end-to-end machine learning project that predicts the power generation of a solar plant based on weather data. This application is built with Scikit-learn and deployed as an interactive web app using Gradio on Hugging Face Spaces.

## 🚀 Live Demo

The application is deployed and live. You can test it here:

**[➡️ Live Solar Power Forecasting App](https://huggingface.co/spaces/hrmn1/solar-forecasting-app)**

## 📖 Problem Statement

With the global shift towards renewable energy, accurately forecasting the output of solar power plants is essential for grid stability and energy management. This project aims to build a simple but effective model to predict the DC power output from a solar plant using real-time weather sensor readings, such as ambient temperature and solar irradiation.

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-learn
* **Web Framework / UI:** Gradio
* **Deployment:** Hugging Face Spaces
* **Version Control:** Git & GitHub

## 📂 Project Structure

The repository is organized as follows:
```text
solar-power-forecasting/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                  # The Gradio application script
├── solar_model.joblib      # The serialized, trained model
│
├── data/                   # Contains the raw data (ignored by git)
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
│
├── notebooks/              # Contains the exploratory data analysis
│   └── 01-EDA-and-Model-Building.ipynb
│
└── src/                    # Source code directory
    └── train.py            # Script to train the model and save it
```

## ⚙️ Running Locally

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hn11-44/solar-power-forecasting.git](https://github.com/hn11-44/solar-power-forecasting.git)
    cd solar-power-forecasting
    ```

2.  **Set up a virtual environment:**
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate

    # Or using conda
    conda create -n solar_env python=3.10
    conda activate solar_env
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the model:**
    (Note: You must have the data files from Kaggle's [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) inside a `data/` folder for this step).
    ```bash
    python src/train.py
    ```
    This will generate the `solar_model.joblib` file in the root directory.

5.  **Run the Gradio application:**
    ```bash
    python app.py
    ```
    Open your browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

## 🤖 Model Details

* **Model Type:** `LinearRegression` from `scikit-learn`.
* **Features:** `AMBIENT_TEMPERATURE` (°C) and `IRRADIATION` (W/m²).
* **Target:** `DC_POWER` (kW).

The model was trained on the full dataset to create the production `solar_model.joblib` file. The exploratory notebook contains a train-test split for performance evaluation, which yielded a Mean Absolute Error (MAE) demonstrating the model's viability.
