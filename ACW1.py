from io import BytesIO #substitute bytes as a regular file to use
from PIL import Image #for png/jpg files
import streamlit as st
import numpy as np
import base64 #convert binary to text format
import wave #for wav/mp3 files

buttonstyle = """
display: inline;
text-align: center;
vertical-align: center;
background: #F63366;
width: 200px;
height: 40px;
line-height: 25px;
padding: 9px;
margin: 8px;
border-radius: 5px;
color: white;
text-decoration: none;
"""


def get_image_download_link(filename, img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = '<a href="data:file/png;base64,'+img_str+'" download=' + \
        filename+' style="'+buttonstyle+'" target="_blank">Download Encoded Image</a>'
    return href


#print diff in img resolution since not obvious
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


#encode based on number of bits to hide
#parameters are image to be encoded, data to be encoded (txt file), and k as number of lsbs selected
def encode_enc_lsb(img, data, k):
    #converts each character in data string into 8-bit binary representation 
    #join together into a single string called datalist
    datalist = "".join([format(ord(i), '08b') for i in data])
    #check if length of datalist is a multiple of 3  
    rem = len(datalist) % (3*k)
    #if not zero, then append zero to make length become a multiple of 3 
    if(rem != 0):
        datalist += '0'*(3*k-rem)
    #ideal length should be even distribution across RGB after all necessary zero padding
    lendata = len(datalist)
    #get width of image
    w = img.size[0]
    #initialize starting image coordinates for iteration of modified pixels
    (x, y) = (0, 0)
    for i in range(0, lendata, k*3):
        #each time extract 3 bits from datalist to encode as RGB
        kbit_r, kbit_g, kbit_b = datalist[i:i +k], datalist[i+k:i+2*k], datalist[i+2*k:i+3*k]
        r, g, b = img.getpixel((x, y))[:3]
        #get existing rgb at image coordinate
        r = ((r >> k) << k)+int(kbit_r, 2)
        g = ((g >> k) << k)+int(kbit_g, 2)
        b = ((b >> k) << k)+int(kbit_b, 2)
        # least significant k bits of each color channel are replaced with the corresponding k 
        # bits from datalist. >> and << operators are used for bitwise shifting to clear the least 
        # significant bits and then replace them with the data bits
        img.putpixel((x, y), (r, g, b))
        #update coordinate with new rgb
        #when reach end of row, reset to 0 otherwise incremented to next row so +1  
        if (x == w - 1):
            x = 0
            y += 1
        #move to next pixel in image, w-1 means width reached so go to next row
        else:
            x += 1
    return img


#encode text file data into the selected number of least significant bits of an image color channel
def encode_lsb(filename, image, k, bytes):
    #for manual text input
    #data = d1.text_area("Enter text to be encoded ", max_chars=bytes)
    #for text file upload
    uploaded_text_file = d1.file_uploader("Upload a text file for encoding", type=["txt"])
    if not uploaded_text_file:
        d1.warning("Accepted File Types: TXT")
        return
    data = uploaded_text_file.read().decode('utf-8')
    #calculate size of encoded file
    encoded_file_size = (len(data) + len("$lsb$")) * k // 8  
    if(d1.button('Encode', key="3")):
        if (len(data) == 0):
            d1.error("Text field is empty.")
        else:
            d2.markdown('##')
            result = "The entered text is succesfully encoded in the cover image."
            d2.success(result)
            d2.markdown("## Encoded Image")
            newimg = image.copy()
            encode_enc_lsb(newimg, data+"$lsb$", k)
            #display the encoded image (newimg) with the color channels in BGR (Blue, Green, Red) format
            d2.image(newimg, channels="BGR")
            filename = 'encoded_'+filename
            #convert the original and encoded images to numpy arrays before calculating MSE
            #why array? for enhanced image processing for element/coordinated operations 
            image_np = np.array(image)
            newimg_np = np.array(newimg)
            MSE = mse(image_np, newimg_np)
            msg = "Mean Squared Error (MSE) is the sum of the squared difference between the original and encoded image. The lower the MSE value, the more similarity between the two images. MSE calculated based on the selected LSB value is "+str(MSE)+"."
            msg += f"\n The encoded file size is now {encoded_file_size} bytes."
            d2.warning(msg)
            d2.markdown(get_image_download_link(filename, newimg), unsafe_allow_html=True)


#encode based on number of bits used to hide
#parameters are image previously encoded, and k as number of lsbs selected
def decode_lsb(img, k):
    #global function to interact with streamlit ui by displaying messages and errors to the user
    global d1 
    #get width and height of image
    w, h = img.size
    x, y = 0, 0
    rawdata = ''
    #iterate through image pixels by extracting the rgb components of each pixel
    #for each color component, pixel value is converted to an 8-bit binary string using format
    #append the last k bits of each binary string to the rawdata string
    #why? encoding process hide text in bits have the least impact on the overall appearance of the image
    #making it less likely to be visually noticeable and since these lsbs are usually located at the end
    #of the binary string, the decoding process will start concatentating from that coordinate
    while(x < w and y < h):
        r, g, b = img.getpixel((x, y))[:3]
        rawdata += format(r, '08b')[-k:]
        rawdata += format(g, '08b')[-k:]
        rawdata += format(b, '08b')[-k:]
        #update coordinate with new rgb
        #when reach end of row, reset to 0 otherwise incremented to next row so +1 
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1
    #generate binary string to mark the end of hidden message
    sec_key = ""
    #converts each character in $lsb$ to its ASCII value and then to an 8-bit binary string
    for i in "$lsb$":
        sec_key += format(ord(i), '08b')
    index = rawdata.find(sec_key)
    if(index == -1):
        d1.error(
            "Please try again! Decoding failed as selected LSB value does not match for encoded and decoded image.")
        return None
    #rawdata string is truncated to remove the special key
    rawdata = rawdata[:index]
    data = ''
    #convert to an integer using int(..., 2) then to its corresponding ASCII character using chr(...)
    #reconstructs the original hidden message
    for i in range(0, len(rawdata), 8):
        data += chr(int(rawdata[i:i+8], 2))
    return data


def encode_text_in_audio(text, audio_file_path):
    with wave.open(audio_file_path, "rb") as audio_file:
        #convert byte to np array of 16 bit integers audio sample until end of file
        frames = audio_file.readframes(-1)
        frames = np.frombuffer(frames, dtype=np.int16)
        # convert text to binary
        binary_text = "".join(format(ord(char), "08b") for char in text)
        # add a null terminator to mark the end
        binary_text += "00000000" 
        # embed binary text into lsbs of audio frames
        encoded_frames = frames.copy()
        text_index = 0
        for i in range(len(encoded_frames)):
            # check if there is more binary text data to embed
            # ensures no additional embedding of text data than is available
            if text_index < len(binary_text):
                # current audio frame = clear and set lsb of the audio frame to 0
                # gets the next bit (0 or 1) from the binary text and converts it to an integer
                # text index increments to move to next bit in binary text
                encoded_frames[i] = (encoded_frames[i] & ~1) | int(binary_text[text_index])
                text_index += 1
    # autosave encoded audio file
    with wave.open("encoded_audio.wav", "wb") as encoded_audio_file:
        encoded_audio_file.setparams(audio_file.getparams())
        encoded_audio_file.writeframes(encoded_frames)
        

def decode_text_from_audio(audio_file_path):
    with wave.open(audio_file_path, "rb") as audio_file:
        frames = audio_file.readframes(-1)
        frames = np.frombuffer(frames, dtype=np.int16)
        # extract binary text from predefined lsbs
        binary_text = ""
        for frame in frames:
            binary_text += str(frame & 1)
        # find null terminator to mark the end of the text
        null_index = binary_text.find("00000000")
        if null_index != -1:
            binary_text = binary_text[:null_index]
        # check if binary text has a length that is a multiple of 8 for valid character conversion
        if len(binary_text) % 8 != 0:
            binary_text = binary_text[:-(len(binary_text) % 8)]
        # convert binary text back to characters
        decoded_text = ""
        for i in range(0, len(binary_text), 8):
            byte = binary_text[i:i + 8]
            decoded_text += chr(int(byte, 2))
    return decoded_text


#main ui using streamlit widgets and above functions (nothing else atas) 
def main():
    global d1, d2
    st.set_page_config(page_title="ACW1 Steganography", initial_sidebar_state="expanded", layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.sidebar.title("ACW1 Steganography")
    info = """
    This LSB Replacement Python program supports steganographic encoding and decoding of a TEXT payload with different image and audio-visual files.
    """
    st.sidebar.markdown(info, unsafe_allow_html=True)
    fileTypes1 = ["png", "jpg"]
    st.markdown("""<style> img{max-width: 100%;} </style>""", unsafe_allow_html=True)
    st.subheader("LSB Replacement Algorithm")
    d1, d2 = st.columns(2)
    choice = d1.selectbox(
        'Select', ["Encode Image", "Decode Image", "Encode Audio", "Decode Audio"])
    #encode image
    if(choice == "Encode Image"):
        d1, d2 = st.tabs(['Step 1: Upload', 'Step 2: Download'])
        #number of LSBs to use be selectable from cover object bits 1 – 8 
        k = d1.slider("Select Value of LSB for Encoding", min_value=1, max_value=8)
        file = d1.file_uploader("Upload Cover Image", type=fileTypes1, key="fu3")
        show_file = d1.empty()
        if not file:
            show_file.warning("Accepted File Types: " + ", ".join(["PNG", "JPG"]))
            return
        im = Image.open(BytesIO(file.read()))
        filename = file.name
        w, h = im.size
        bits = (w*h*3*k)-40
        bytes = (bits)//8
        encode_lsb(filename, im, k, bytes)
        if isinstance(file, BytesIO):
            show_file.image(file)
            file.close()
    #decode image
    elif(choice == "Decode Image"):
        d1, d2 = st.tabs(['Step 1: Upload', 'Step 2: View Decoded Text'])
        k = d1.slider("Select Value of LSB for Decoding", min_value=1, max_value=8)
        file = d1.file_uploader("Upload Encoded Image", type=fileTypes1, key="fu4")
        show_file = d1.empty()
        if not file:
            show_file.warning("Accepted file types: " + ", ".join(["PNG", "JPG"]))
            return
        im = Image.open(BytesIO(file.read()))
        if(d1.button('Decode', key="2")):
            data = decode_lsb(im, k)
            if(data != None):
                d2.subheader("Decoded Text: " + data)
        if isinstance(file, BytesIO):
            show_file.image(file)
        file.close()
    #encode audio
    elif(choice == "Encode Audio"):
        audio_file = st.file_uploader(
            "Upload an audio file (WAV)", type=["wav"])
        secret_text_file = st.file_uploader(
            "Upload a text file for encoding", type=["txt"])
        if st.button("Encode"):
            if secret_text_file is None:
                st.warning("No text file uploaded.")
            elif audio_file is None:
                st.warning("Please upload an audio file.")
            else:
                secret_text = secret_text_file.read().decode()
                encode_text_in_audio(secret_text, audio_file)
                st.success("The entered text is succesfully encoded in the cover audio. Modified audio has been downloaded to the ImgStego folder.")
    #decode audio
    elif(choice == "Decode Audio"):
        audio_file = st.file_uploader(
            "Upload Encoded Audio", type=["wav"], key="fu3")
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
        else:
            st.error("No encoded audio file uploaded.")
        if st.button("Decode Text"):
            decoded_audio_text = decode_text_from_audio(audio_file)
            if decoded_audio_text:
                st.subheader("Decoded Text: " + decoded_audio_text)


if __name__ == "__main__":
    main()
