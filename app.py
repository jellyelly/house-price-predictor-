import gradio as gr
import joblib
import numpy as np
import pandas as pd

model = joblib.load('model.joblib')
unique_values = joblib.load('unique_values.joblib')
neighborhood_values = unique_values['Neighborhood']

# Define the prediction function
def predict(neighborhood, house_size, num_rooms):
    # Convert inputs to appropriate types
    house_size = float(house_size)
    num_rooms = int(num_rooms)
    
    # Prepare the input array for prediction
    input_data = pd.DataFrame({
        'Neighborhood': [neighborhood],
        'House Size': [house_size],
        'Number of Rooms': [num_rooms]
    })
    
    # Perform the prediction
    prediction = model.predict(input_data)
    
    return f"The predicted house price is ${prediction[0]:,.2f}"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=list(neighborhood_values), label="Neighborhood"),
        gr.Textbox(label="House Size (in square feet)"),
        gr.Textbox(label="Number of Rooms")
    ],
    outputs="text",
    title="House Price Predictor",
    description="Enter the neighborhood, house size, and number of rooms to predict the house price."
)



# Launch the app
interface.launch()