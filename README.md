# DisguisedFaceGeneration
Generation of Photo-realistic Disguised Faces from Sketches using GANs


# Project Scope
The scope of this project is to design and develop an AI-powered system that can generate
realistic human faces from police sketches and create variations of these faces with disguises, such
as adding beards, glasses, or hats. The generated images will be stored in a database that will be
fed into subjective systems and can be matched against real-world images captured by surveillance
or even immigration cameras. This system aims to assist law enforcement agencies in identifying
suspects more effectively and bridging the gap between traditional sketch-based identification and
modern technological advancements.

# Functional Requireements / Modules

1. Web Scraping Module
Responsible for collecting a large dataset of face images, specifically focusing on Indian
facial features, from employee database website. It uses web scraping techniques to gather these
images from URLs in an automated manner, ensuring diversity in the dataset. The module fetches
images from websites and stores them for further processing in the following stages.
2. Face Detection and Cropping Module
This module uses the HAAR Cascade Classifier for face detection to locate faces within the
collected images. Once a face is detected, the module automatically crops the squared image to
focus on the face, discarding irrelevant portions of the image. This ensures that only the face is
used for further processing, improving the accuracy and relevance of the data for training the AI
models.
3. Sketch Generation Module
Utilizes OpenCV to convert the cropped face images into abstract sketches. Using
OpenCV’s image processing techniques, such as edge detection and smoothing filters, the module
generates pencil-drawn sketches that serve as input for the GAN model. These sketches retain key
facial features necessary for the AI model to accurately generate realistic faces from them.
4. Sketch-to-Image Generation Module
Uses an Unconditional Pix2Pix Generative Adversarial Network (GAN) to convert the
abstract sketches into realistic human faces. This module is trained on the dataset of sketches and
corresponding real face images. After evaluating several models namely Conditional and
Unconditional Pix2Pix, CycleGAN, Autoencoder with GANs mentioned, and more, the best-
performing GAN was selected due to its ability to generate high-quality, visually coherent faces
while being computationally efficient. The model produces a realistic face based on the features
outlined in the sketch.
5. Disguise Variation Module
Utilizes Stable Diffusion v1-5 to generate variations of the generated faces, adding
disguises such as beards, glasses, and hats based on the prompts given by the user through text.
The model was fine-tuned and retrained to suit the specific needs of face variation generation. This
module allows for creating multiple variations of the same face, which can simulate different
appearances or disguises, making it useful for law enforcement in identifying suspects who may
have altered their appearance. For this model, dataset creation was required for fine tuning the
model for specified prompts.
6. Face Verification Module
The Face Verification Module integrates Facenet, PyTorch-based face recognition algorithm
to compare the generated face images with those captured. This module processes the generated
and real-world face images to identify matches, ensuring accurate identification of suspects.
7. User Interface Module
The User Interface (UI) is developed using Streamlit, providing a simple, interactive
platform for law enforcement personnel to input sketches, view generated face images, and apply
disguise variations (through text prompts generic as well as custom prompts) for the user, adjust
model parameters, slide through different tabs and verify the detected image efficiently. The UI is
designed for ease of use, enabling users to access all functionalities with minimal effort.



