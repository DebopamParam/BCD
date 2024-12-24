import streamlit as st
import os
import math

# Set the path to the evaluation folder
evaluation_folder = "static/evaluation"

# Get a list of image file names in the evaluation folder
image_names = [
    filename
    for filename in os.listdir(evaluation_folder)
    if filename.endswith(
        (".png", ".jpg", ".jpeg", ".gif")
    )  # Filter for common image types
]

# Create the full path for the image files
image_paths = [os.path.join(evaluation_folder, filename) for filename in image_names]


# Display images in a 2x2 grid or more rows if necessary.
num_images = len(image_paths)

if num_images > 0:
    num_cols = 2  # Display in 2 columns
    num_rows = math.ceil(
        num_images / num_cols
    )  # Get how many rows we need to display the images

    # Create a list of st.columns, the length of the list will be equivalent to the amount of columns we want to display
    cols = st.columns(num_cols)

    # Loop over each row and column
    for row in range(num_rows):
        for col in range(num_cols):
            image_index = row * num_cols + col
            if image_index < num_images:
                with cols[col]:
                    st.image(image_paths[image_index], use_container_width=True)

else:
    st.write("No images found in the evaluation folder.")
