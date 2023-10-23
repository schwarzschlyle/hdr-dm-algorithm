from PIL import Image
from PIL.ExifTags import TAGS
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cProfile
import colorsys
import tqdm
import pickle


# To save time, we can extract metadata algorithmically
# Afterwards, we compile them into an index .txt file
# placed at the same directory as our dataset

def extract_exif_data(image_path):
    try:
        #open the image file
        with Image.open(image_path) as img:
            # extract Exif data
            exif_data = img._getexif()

            # Check existence of Exif data
            if exif_data is not None:
                # IF meron, itereate
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    
                    # camera details can be accessed via tags
                    if "ExposureTime" in tag_name:
                        exposure_time = str(value)
                    elif "ShutterSpeedValue" in tag_name:
                        shutter_speed = str(value)

                return exposure_time, shutter_speed

    except Exception as e:
        print(f"Error: {e}")
    
    return None, None



for i in range(1, 10):
    image_path = f"exposure_change/et_0{i}.jpg"
    exposure_time, shutter_speed = extract_exif_data(image_path)

    if exposure_time and shutter_speed:
        print(f"et_0{i}")
        print(f"Exposure Time: {exposure_time}")
        print(f"Shutter Speed: {shutter_speed}")
    else:
        print("Exif data not found or invalid.")

        
for i in range(10, 20):
    image_path = f"exposure_change/et_{i}.jpg"
    exposure_time, shutter_speed = extract_exif_data(image_path)

    if exposure_time and shutter_speed:
        print(f"et_{i}")
        print(f"Exposure Time: {exposure_time}")
        print(f"Shutter Speed: {shutter_speed}")
    else:
        print("Exif data not found or invalid.")
        
        
        
        
    
# we will be dealing with the reciprocal of the shutter speed
# I know this is poor code hygiene but I'm already really tired
print(1/29.8973)




# Apparently, upon directly dealing with jpg files, the code becomes buggy or idk
# PNG works best for me

# converting jpg to png


def convert_jpg_to_png(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img.save(output_path, 'PNG')
            print(f"Conversion successful. PNG image saved at {output_path}")
    # error handling is always a good practice
    except Exception as e:
        print(f"Error: {e}")

def batch_convert_jpg_to_png(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        # .jpg regex
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_directory, filename)
            # Change the extension to PNG for the output file
            # converting them to .png. 
            # TBH, I ACTUALLY don't know if this converts the images into an actual PNG file
            # But a .png suffix seems to work idk
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".png")

            # Convert JPG to PNG
            convert_jpg_to_png(input_path, output_path)

# just dumping it in the same data directory lol
input_directory = 'exposure_change'
output_directory = 'exposure_change'

batch_convert_jpg_to_png(input_directory, output_directory)



# ------------------------------------------------

def load_exposures(source_dir, channel=0):
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        print(line[0])
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]
    
    img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
    img_list = [img[:,:,channel] for img in img_list]
    exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times)


# ----------------------------------------------------

# Equation 3

def response_curve_solver(Z, B, l, w):
    """
    This response solver is an implementation of the Debevec and Malik paper
    refactored from gh repo clone SSARCandy/HDR-imaging
    """
    
    n = 256
    A = np.zeros(shape=(np.size(Z, 0)*np.size(Z, 1)+n+1, n+np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # First term of the system of equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = Z[j][i]
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij*B[j]
            k += 1
    
    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Second term of the system of equations
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    print(np.shape(A))
    print(np.shape(b))
    
    # Solved via singular value decomposition
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


# ------------------------------------------


def hdr_debvec(img_list, exposure_times):
    B = [math.log(e,2) for e in exposure_times]
    l = 10
    w = [z if z <= 0.5*255 else 255-z for z in range(256)] 
    
    
    small_img = [cv2.resize(img, (10, 10)) for img in img_list]
    Z = [img.flatten() for img in small_img]
    
    plt.imshow(small_img[0], cmap='gray')
    plt.imshow(small_img[1], cmap='gray')
    plt.imshow(small_img[2], cmap='gray')
    plt.show()
  
    return response_curve_solver(Z, B, l, w)




# ----------------------------------------------



# Equation 6

def construct_radiance_map(g, Z, ln_t, w):
    acc_E = [0]*len(Z[0])
    ln_E = [0]*len(Z[0])
    
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z]*(g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else acc_E[i]
        acc_w = 0
        if i % 100000 == 0:
            print (f"{100*float(i)/float(pixels)}")
    
    return ln_E


# ---------------------------------------------



def construct_hdr(img_list, response_curve, exposure_times):
    # Construct radiance map for each channels
    img_size = img_list[0][0].shape
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]
    ln_t = np.log2(exposure_times)

    vfunc = np.vectorize(lambda x:math.exp(x))
    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

    # construct radiance map for BGR channels
    for i in range(3):
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(vfunc(E), img_size)

    return hdr


# ----------------------------------------


# Loading exposure images into a list
dirname = 'exposure_change'
img_list_b, exposure_times = load_exposures(dirname, 0)
img_list_g, exposure_times = load_exposures(dirname, 1)
img_list_r, exposure_times = load_exposures(dirname, 2)

gb, _ = hdr_debvec(img_list_b, exposure_times)
gg, _ = hdr_debvec(img_list_g, exposure_times)
gr, _ = hdr_debvec(img_list_r, exposure_times)




# -----------------------------------------------------


plt.figure(figsize=(10,10))
plt.plot(gr,range(256), 'rx')
plt.plot(gg,range(256), 'gx')
plt.plot(gb,range(256), 'bx')
plt.ylabel('pixel value Z')
plt.xlabel('log exposure X')
plt.show()




# ---------------------------------------------

hdr = construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)

# This took me about, idk, 2 hours? So it's best to pickle dump it:

with open('output_hdr.pkl', 'wb') as f:
    pickle.dump(hdr, f)
    
    
    
# ---------------------------------------------


# to store the HDR image, we can convert it into Radiance HDR



# initiallizing a numpy array as container,
image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')


# storing each channel
image[..., 0] = hdr[..., 2]
image[..., 1] = hdr[..., 1]
image[..., 2] = hdr[..., 0]



# now, we open the generated hdr image,


f = open("167.hdr", "wb")
f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
header = "-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]) 
f.write(bytes(header, encoding='utf-8'))


#sweep acorss all channels to find the highest pixel value (brightest)
brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])



mantissa = np.zeros_like(brightest)
exponent = np.zeros_like(brightest)
np.frexp(brightest, mantissa, exponent)
scaled_mantissa = mantissa * 256.0 / brightest
rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
rgbe[...,3] = np.around(exponent + 128)

rgbe.flatten().tofile(f)
f.close()

# ----------------------------------------------------


# References:

# Most of the algorithms was extracted from the following repo: 
# gh repo clone SSARCandy/HDR-imaging

