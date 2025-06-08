import gradio as gr
import joblib
import pandas as pd


# Load the saved model 
model = joblib.load('solar_model.joblib')

# Prediction function 

def predict_solar_power(ambient_temperature, irradiation): 
    
    """
    Takes two values, ambient temperature and irradiation to give a predicted 
    DC power.
    """
    
    input_data = pd.DataFrame([[ambient_temperature, irradiation]],
                              columns = ["AMBIENT_TEMPERATURE", "IRRADIATION"])

    prediction = model.predict(input_data)
    
    return round(prediction[0].item(), 2)

# Build the Gradio Interface 
# gr.Interface for UI from our function
# fn the function to wrap 
# inputs for user to provide input 
# outputs to disply result 
# title and description for the text to disply on UI

with gr.Blocks() as demo:
    gr.Markdown(
        
    """
    # ☀️ Solar Power Generation Forecast
    Enter the weather conditions to predict the power (in kW) of the solar plant.
    This model was trained on data from a real solar plant
    """
    )
    
    with gr.Row():
        temp = gr.Number(label = "Ambient Temperature (°C)")
        irrad = gr.Number(label = "Solar Irradiation W/m²)")
    
    output = gr.Textbox(label = "Predicted DC Power (kW)")
    
    
    predict_btn = gr.Button("Predict")
    predict_btn.click(fn = predict_solar_power, inputs = [temp, irrad], outputs = output, api_name = "predict")
    
    
    gr.Markdown("----")
    gr.Markdown("### Example Values")
    gr.Examples(
        examples = [
        [25, 0], # Night time
        [30, 0.5], # Cloud day 
        [35, 1.0], # Sunny day
        ],
        inputs = [temp, irrad]
        )
    
    # Launching 
    if __name__ == "__main__":
        demo.launch()