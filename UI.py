import streamlit as st
from PIL import Image
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

scaler = MinMaxScaler()

path = r"C:/Users/gaura/OneDrive/Desktop/DESKtop/BTS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_"

# MAIN FUNCTION:
# ----------------------------------------------------

rows = 2
columns = 2

slice_w = 25
n_slice = 55

st.set_page_config(layout="wide")

st.write(""" # 3D Brain Tumor Image Segmentation:
Please upload the required image. """)

uploaded_files = st.sidebar.file_uploader(
    "Choose a Brain Tumor 3D Image:", accept_multiple_files=True)


if uploaded_files is not None:

    st.session_state['initial_image'] = True
    for uploaded_file in uploaded_files:

        if str(uploaded_file.name)[-9::] == 'flair.nii':

            number = uploaded_file.name[-13:-10]
            number = str(number)
            complete_path = path + number + "/" + uploaded_file.name
            st.session_state['image_flair'] = nib.load(
                complete_path).get_fdata()
            st.session_state['image_flair'] = scaler.fit_transform(
                st.session_state['image_flair'].reshape(-1,  st.session_state['image_flair'].shape[-1])).reshape(st.session_state['image_flair'].shape)

        elif str(uploaded_file.name)[-7::] == 'seg.nii':
            number = uploaded_file.name[-11:-8]
            number = str(number)
            complete_path = path + number + "/" + uploaded_file.name
            st.session_state['image_seg'] = nib.load(
                complete_path).get_fdata()

        elif str(uploaded_file.name)[-6::] == 't1.nii':
            number = uploaded_file.name[-10:-7]
            number = str(number)
            complete_path = path + number + "/" + uploaded_file.name
            st.session_state['image_t1'] = nib.load(complete_path).get_fdata()
            st.session_state['image_t1'] = scaler.fit_transform(
                st.session_state['image_t1'].reshape(-1,  st.session_state['image_t1'].shape[-1])).reshape(st.session_state['image_t1'].shape)

        elif str(uploaded_file.name)[-8::] == 't1ce.nii':
            number = uploaded_file.name[-12:-9]
            number = str(number)
            complete_path = path + number + "/" + uploaded_file.name
            st.session_state['image_t1ce'] = nib.load(
                complete_path).get_fdata()
            st.session_state['image_t1ce'] = scaler.fit_transform(
                st.session_state['image_t1ce'] .reshape(-1,  st.session_state['image_t1ce'] .shape[-1])).reshape(st.session_state['image_t1ce'].shape)

        elif str(uploaded_file.name)[-6::] == 't2.nii':
            number = uploaded_file.name[-10:-7]
            number = str(number)
            complete_path = path + number + "/" + uploaded_file.name
            st.session_state['image_t2'] = nib.load(complete_path).get_fdata()
            st.session_state['image_t2'] = scaler.fit_transform(
                st.session_state['image_t2'].reshape(-1,   st.session_state['image_t2'].shape[-1])).reshape(st.session_state['image_t2'].shape)

    if st.session_state['initial_image'] == True:

        if st.sidebar.button('Show Image'):

            st.session_state['initial_image'] = True
            fig = plt.figure(figsize=(30, 20))

            fig.add_subplot(rows, columns, 1)
            plt.imshow(st.session_state['image_flair'][:, :,
                                                       st.session_state['image_flair'].shape[0]//2-slice_w], cmap='gray')
            plt.axis('off')
            plt.title("' Flair '", fontsize=35)

            fig.add_subplot(rows, columns, 2)
            plt.imshow(st.session_state['image_t1ce'][:, :,
                                                      st.session_state['image_t1ce'].shape[0]//2-slice_w], cmap='gray')
            plt.axis('off')
            plt.title("' T1ce '", fontsize=35)

            fig.add_subplot(rows, columns, 3)
            plt.imshow(st.session_state['image_t2'][:, :,
                                                    st.session_state['image_t2'].shape[0]//2-slice_w], cmap='gray')
            plt.axis('off')
            plt.title("' T2 '", fontsize=35)

            fig.add_subplot(rows, columns, 4)
            plt.imshow(st.session_state['image_t1'][:, :,
                                                    st.session_state['image_t1'].shape[0]//2-slice_w], cmap='gray')
            plt.axis('off')
            plt.title("' T1 '", fontsize=35)

            plt.savefig("Images/complete.png")
            image = Image.open('Images/complete.png')
            st.image(image)

            st.session_state['initial_image'] = False
            st.session_state['get_results'] = True

    if 'get_results' in st.session_state:

        if st.sidebar.button('Get Results..'):

            # The preprocessing & model run code via object oriented way.
            combined_image = np.stack([st.session_state['image_flair'],
                                       st.session_state['image_t1ce'], st.session_state['image_t2']], axis=3)

            combined_image = combined_image[56:184, 56:184, 13:141]
            test_mask = st.session_state['image_seg'].astype(np.uint8)
            test_mask[test_mask == 4] = 3
            print(np.unique(test_mask))

            test_mask = test_mask[56:184, 56:184, 13:141]

            test_img_input = np.expand_dims(combined_image, axis=0)
            test_mask = to_categorical(test_mask, num_classes=4)

            test_mask_argmax = np.argmax(test_mask, axis=3)

            model = load_model('')
            test_prediction = model.predict(test_img_input)

            test_prediction_argmax = np.argmax(
                test_prediction, axis=4)[0, :, :, :]

            n_slice = 55
            plt.figure(figsize=(12, 8))
            plt.subplot(231)
            plt.title('Testing Image')
            plt.imshow(combined_image[:, :, n_slice, 1], cmap='gray')
            plt.subplot(232)
            plt.title('Testing Label')
            plt.imshow(test_mask_argmax[:, :, n_slice])
            plt.subplot(233)
            plt.title('Prediction on test image')
            plt.imshow(test_prediction_argmax[:, :, n_slice])
            plt.savefig('images/results.png')

            image = Image.open('Images/results.png')
            st.image(image)
