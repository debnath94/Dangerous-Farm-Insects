# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:57:14 2023

@author: debna
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import wikipedia
import pickle

# Load the model using pickle
#with open('pest.pickle', 'rb') as f:
    #model = pickle.load(f)

model = tf.keras.models.load_model("pest_mobilenet.h3")

def get_wikipedia_summary(page_title):
    try:
        page = wikipedia.page(page_title)
        return page.summary
    except wikipedia.exceptions.PageError:
        return "No information found on Wikipedia."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous page title. Please be more specific: {', '.join(e.options)}."

def get_pesticide_suggestions(predicted_label):
    pesticide_suggestions = {
        'Armyworms': ['Bacillus thuringiensis (Bt)', 'Pyrethroids', 'Spinosad'],
        'Thrips': ['Neonicotinoids', 'Organophosphates', 'Insecticidal soaps'],
        'Aphids': ['Pyrethroids', 'Neonicotinoids', 'Insecticidal oils'],
        'Brown Marmorated Stink Bugs': ['Pyrethroids', 'Insecticidal soaps', 'Botanical insecticides'],
        'Cabbage Loopers': ['Bacillus thuringiensis (Bt)', 'Spinosad', 'Insecticidal soaps'],
        'Citrus Canker': ['Copper-based fungicides', 'Streptomycin'],
        'Colorado Potato Beetles': ['Neonicotinoids', 'Pyrethroids', 'Spinosad'],
        'Corn Borers': ['Bacillus thuringiensis (Bt)', 'Pyrethroids', 'Spinosad'],
        'Corn Earworms': ['Bacillus thuringiensis (Bt)', 'Pyrethroids', 'Spinosad'],
        'Fall Armyworms': ['Bacillus thuringiensis (Bt)', 'Pyrethroids', 'Spinosad'],
        'Fruit Flies': ['Fruit fly bait', 'Insecticidal traps', 'Methyl eugenol lure'],
        'Spider Mites': ['Acaricides', 'Insecticidal soaps', 'Neem oil'],
        'Tomato Hornworms': ['Bacillus thuringiensis (Bt)', 'Spinosad', 'Insecticidal soaps'],
        'Western Corn Rootworms': ['Bacillus thuringiensis (Bt)', 'Pyrethroids', 'Soil insecticides'],
        'Africanized Honey Bees Killer Bees': ['Professional beekeeping services', 'Honey bee relocation'],
        # Add more pest labels and their corresponding pesticide suggestions here
    }

    return pesticide_suggestions.get(predicted_label, [])

def main():
    st.title("Upload an insect image for pesticide suggestions tailored to your farm's insect menace")
    st.sidebar.title("Options")

    # Add input options to the sidebar
    upload_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if upload_file is not None:
        image = Image.open(upload_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predict_button = st.sidebar.button('Predict')

        if predict_button:
            # Preprocess the image
            image = np.array(image.resize((100, 100)))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Make predictions
            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            classes = ['Africanized Honey Bees Killer Bees', 'Aphids', 'Armyworms', 'Brown Marmorated Stink Bugs',
                       'Cabbage Loopers', 'Citrus Canker', 'Colorado Potato Beetles',
                       'Corn Borers', 'Corn Earworms', 'Fall Armyworms', 'Fruit Flies',
                       'Spider Mites', 'Thrips', 'Tomato Hornworms', 'Western Corn Rootworms']
            predicted_label = classes[class_index]

            # Fetch details from Wikipedia
            wikipedia_summary = get_wikipedia_summary(predicted_label)

            # Fetch pesticide suggestions
            pesticide_suggestions = get_pesticide_suggestions(predicted_label)

            st.success(f"Prediction: {predicted_label}")
            st.info(f"Wikipedia Summary: {wikipedia_summary}")

            if pesticide_suggestions:
                st.info("Pesticide Suggestions:")
                for suggestion in pesticide_suggestions:
                    st.write(f"- {suggestion}")

if __name__ == '__main__':
    main()
