import streamlit as st
from test2_streamlit_gan import gan
from torch import device, set_grad_enabled, sigmoid
from torch.cuda import is_available
from torch.optim import Adam
import dataset, losses
from tqdm import tqdm
import new_components as components
from PIL import Image
from torchvision import transforms as T
import torch
import numpy as np
from torch.utils.data import DataLoader
from diffusers import StableDiffusionImg2ImgPipeline
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import base64
import time

st.set_page_config(page_title="FaceTrace", page_icon=":house:", layout="centered", initial_sidebar_state="collapsed")

page_bg = """
<style>

[data-testid="stAppViewContainer"] {
    background-image: url('./bkg.jpeg');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSpinner"] {
    scale: 2;
    margin-top: 50px;
    margin-left: 550px;
}
h1 {
  scale: 1.5;
}

[data-testid="stTabs"]{
  margin-left: -175px;
  margin-top: 50px;
  width: 1000px;
}
.st-emotion-cache-uef7qa.e1nzilvr5 > p{
  scale:1.25;
  width: 300px;
  margin-bottom: 5px;
}
[data-testid="stWidgetLabel"] {
    margin-top: 10px;
    margin-left: 40px;
}
[data-testid="stToastContainer"]{
  scale: 1.2;
  margin-right:200px;
}


</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        img_data = img_file.read()
    b64_img = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bkg.jpg")

st.title("**FaceTrace**")

tab1, tab2, tab3 = st.tabs(["**Face Generation**","**Disguised Face Generation**", "**Face Verification**"])
device_name = device('cuda:0') if is_available() else device('cpu')

model_path = "/home/user/sk2df/Code Files/diffusers-main/examples/dreambooth/1000/" # "/home/user/stable-diffusion-v1-5/"  /home/user/sk2df/Code Files/diffusers-main/examples/dreambooth/1000/
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="casia-webface")
resnet.eval()

generated_image = None
output1 = None
output2 = None
distance = 0

def verify(image1, image2):
    face_image_1 = image1.convert("RGB")
    face_image_2 = image2.convert("RGB")
    
    faces1, _ = mtcnn.detect(image1)
    faces2, _ = mtcnn.detect(image2)
    
    if faces1 is not None and faces2 is not None:
        aligned1 = mtcnn(face_image_1)
        aligned1 = aligned1.unsqueeze(0) 
        aligned2 = mtcnn(face_image_2)
        aligned2 = aligned2.unsqueeze(0)
        
        embeddings1 = resnet(aligned1).detach()
        embeddings2 = resnet(aligned2).detach()
        
        distance = (embeddings1 - embeddings2).norm().item()
    
    return distance


with tab1:
    st.header("**Face Generation**")
    transforms = T.Compose([T.Grayscale(), T.Resize(512), T.ToTensor(), ])

    uploaded_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        sketch = Image.open(uploaded_file)
        sketch_loader = transforms(sketch).to(device_name)
        st.subheader("Pre-processing the sketch...")
        st.write("Enhancing and modifying the sketch")
        progress_bar = st.progress(0)
        for i in range(2):
            progress_bar.progress((i+1)*50)
            time.sleep(1)
            
        st.subheader("Generating human face...")
        st.write("The system will now use GAN to synthesize a face. This might take a few moments.")
        progress_bar1 = st.progress(0)
            
        with set_grad_enabled(gan.training):
            generated_image = gan(sketch_loader)
           
        generated_image = generated_image.cpu().permute(1, 2, 0).numpy()
        generated_image = np.clip(generated_image, 0, 1)
        generated_image = (generated_image * 255).astype(np.uint8)
       
        sketch = sketch.resize((512, 512))
        col1, col2 = st.columns(2)
        with col1:
            st.image(sketch, caption='Uploaded Sketch')
            
        for i in range(4):
            progress_bar1.progress((i+1)*25)
            time.sleep(1)
            
        with col2:
            st.image(generated_image, caption='Generated Image')
            
        st.success("Generated successfully!")
        
    verification_file = st.file_uploader("Choose an image to verify...", type=["jpg", "jpeg", "png"])
    if verification_file is not None:
        st.toast(":green[**Check verification tab for results**]")


with tab2:
    st.header("**Disguised Face Generation**")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float, safety_checker = None)
    pipe = pipe.to(device_name)
    # st.success("Pipeline Built!")

    if "selected_option" not in st.session_state:
        st.session_state.selected_option = "None"

    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if generated_image is not None:
        init_image = Image.fromarray(generated_image)
        init_image = init_image.convert("RGB")
        init_image = init_image.resize((512, 512))

        st.sidebar.header("Feature Selection")

        prompt_mapping1 = {
            "Eyeglasses": "give eyeglasses.",
            "Beard": "give long,thick beard.",
            "Mustache": "give thick mustache.",
            "Makeup": "give heavy makeup,red lipstick.",
            "Scar": "give scar.",
            "Blue Eyes": "give blue eyes.",
            "Straight Hair": "give straight hair.",
            "Curly Hair": "give curly hair.",
            "Short hair": "give short hair.",
            "Smile": "add smile.",
            "Anger": "make face angry.",
            "Sad": "make sad.",
            "Amused": "make amused face.",
            "Annoyed": "make sad.",
            "Old age": "add wrinkles, gray hair.",
            "American ethnicity": "fair skin, make American.",
            "European ethnicity": "fair skin, make European.",
            "Asian ethnicity": "asian.",
            "Bald": "photo of a rajasi bald person."
            
            
        }

        st.sidebar.markdown("""
            <style>
            .full-width-button {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
                background-color: #f0f0f5;
                color: black;
                border: none;
                border-radius: 4px;
                padding: 5px 0;
                margin-bottom: 10px;
                font-size: 16px;
                cursor: pointer;
            }
            .full-width-button:hover {
                background-color: #30a65b;
            }
            </style>
        """, unsafe_allow_html=True)
        
        selected_option = None
        
        custom = st.sidebar.toggle("Custom prompt")
        strength = st.sidebar.slider("Strength", min_value=0.3, max_value=1.0, value=0.35, step=0.05, label_visibility="visible")
        inference = st.sidebar.slider("Inference steps", min_value=50, max_value=1000, value=200, step=50, label_visibility="visible")
        if custom:
            selected_option = st.sidebar.text_area(r"$\textsf{\normalsize Enter prompt}$", None)
            prompt = selected_option
            st.sidebar.warning("Please enter valid prompts for better results")
            
        else:
            st.sidebar.header("Physical Attributes")
            for option in ["Eyeglasses", "Beard", "Mustache","Curly Hair", "Makeup", "Scar", "Blue Eyes", "Bald"]:
                if st.sidebar.button(option, key=option, use_container_width=True):
                    selected_option = option
            
            st.sidebar.header("Emotional Attributes")
            for option in ["Smile", "Anger", "Sad"]:
                if st.sidebar.button(option, key=option, use_container_width=True):
                    selected_option = option
            
            st.sidebar.header("Demographic Attributes")
            for option in ["Old age", "American ethnicity", "European ethnicity", "Asian ethnicity"]:
                if st.sidebar.button(option, key=option, use_container_width=True):
                    selected_option = option
            prompt = prompt_mapping1.get(selected_option)


        
        if prompt is not None:
            container = st.container()
            with container:
                with st.spinner():
                    # prompt = prompt + "hd quality,dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD"
                    N = 2
                    init_image_batch = [init_image] * N
                    print(init_image.size)
                    negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, teeth"
        
                    images = pipe(prompt=prompt, image=init_image, strength=strength,num_inference_steps=inference, num_images_per_prompt = N, negative_prompt = negative_prompt).images   
                    output1 = images[0].resize((512, 512))
                    output2 = images[1].resize((512, 512))
                    init_image = init_image.resize((512, 512))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(init_image, caption='Uploaded Image')
            with col2:
                st.image(output1, caption='Generated Image 1')
            with col3:
                st.image(output2, caption='Generated Image 2')

            st.success("Generated successfully!")
            st.info("Not satisfied with the results ? Try again or use the custom prompt option")

            st.toast(":green[**Check verification tab for results**]")
            
with tab3:
    st.header("**Face Verification**")
    if verification_file is not None and generated_image is not None:
        uploaded_face = Image.open(verification_file)
        p_generated_image = Image.fromarray(generated_image.astype('uint8'))
        distance = verify(uploaded_face, p_generated_image)
        uploaded_face = uploaded_face.resize((512, 512))
        
        col1, col2 = st.columns(2)
        
        if distance < 0.91:
            with col1: 
              st.image(uploaded_face, caption='Verification image')
            with col2:
              st.image(generated_image, caption='Generated Image')
              st.write("Distance: ", format(distance, ".2f"))
            st.success("**Match Found**")
        else:
            st.error("**Match Not Found**")
     
    if verification_file is not None and output1 is not None and output2 is not None:     
        uploaded_face = Image.open(verification_file)
        distance1 = verify(uploaded_face, output1)
        distance2 = verify(uploaded_face, output2)
        output1 = output1.resize((512, 512))
        output2 = output2.resize((512, 512))
        uploaded_face = uploaded_face.resize((512, 512))
        
        flag1 = False 
        flag2 = False
        print(distance1, distance2)
        col1, col2, col3 = st.columns(3)

        if distance1 < 1.0:
          flag1 = True
          with col1: 
            st.image(uploaded_face, caption='Verification image')
          with col2:
            st.image(output1, caption='Generated Image 1')
            st.write("Distance: ", format(distance1, ".2f"))
        if distance2 < 1.0:
          flag2 = True
          if flag1 == False: 
            with col1: 
              st.image(uploaded_face, caption='Verification image')
            with col2:
              st.image(output2, caption='Generated Image 2')
              st.write("Distance: ", format(distance2, ".2f"))
          else: 
            with col3:
              st.image(output2, caption='Generated Image 2')
              st.write("Distance: ", format(distance2, ".2f"))
        if flag1==False and flag2==False:
            st.error("**Match Not Found**")
        else:
            st.success("**Match Found**")


#    
