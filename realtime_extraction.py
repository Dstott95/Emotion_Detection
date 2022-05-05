import cv2
import torchvision.transforms as transforms
import torch
import mediapipe as mp
import pyaudio
#from model import LSTM # import NN model for classification

'''RAVDESS video is 720p H.264, AAC, 48kHz, .mp4'''

# Set up video source and device
video = cv2.VideoCapture(0) # for real-time webcam usage
# cam = cv2.VideoCapture('./video/locations/video.mp4') # for saved video
device = torch.device('cuda')

# Define video parameters
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width, height) # print for debugging
video.set(cv2.CAP_PROP_FPS, 30)
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set up audio using pyaudio
FRAMESIZE = 1024
HOPLENGTH = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
update_Time = 3 # in seconds
CHUNK = RATE * update_Time
p = pyaudio.PyAudio()

print("* recording")

# function to move a tensor to a device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# function to detect face(s) and define a box around it
def faceBox(frame):
    findFace = mp.solutions.face_detection.FaceDetection()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = findFace.process(frameRGB)
    myFaces = []
    if results.detections != None:
        for face in results.detections:
            bBox = face.location_data.relative_bounding_box
            myFaces.append(bBox)
    return myFaces

'''use below if classifying using a trained model'''
# loads in pretrained weights
# w = '.\weight\location\LSTM.pth'
#model = LSTM.to(device)
#if str(device) == 'cpu':
    #model.load_state_dict(torch.load(w, map_location=torch.device('cpu')))
#if str(device) == 'cpu':
    #model.load_state_dict(torch.load(w, map_location=torch.device('cuda')))

while True:
    # read in video frame camera source
    _ , frame = video.read()
    # find face(s) using faceBox function
    bBox = faceBox(frame)
    '''# read in audio
    audio_frames = []
    audio = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)'''

    if len(bBox) > 0:
        for box in bBox:
            # extract face area from bBox
            x, y = int(box.xmin*width), int(box.ymin*height)
            w, h = int(box.width*width), int(box.height*height)
            # create visible box around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceExp = frame[y:y+h, x:x+w]
            try:
                # convert video into a tensor, so it can be input into the model
                faceExpResized = cv2.resize(faceExp, (80, 80))
                transform = transforms.ToTensor()
                faceExpResizedTensor = transform(faceExpResized)
            except:
                continue
            # Use model to predict emotion
            #prediction = LSTM.predict_image(data, model, device)
            prediction = 'Test'
            # Display prediction above facebox
            cv2.putText(frame, prediction, (x, y),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1,
                                color=(0, 0, 0),
                                thickness=2 )

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xff == ord('q'): # to quit the camera press q
        print('End')
        break

# close video capture
video.release()
cv2.destroyWindow('Emotion Detection')

'''# close audio capture
stream.stop_stream()
stream.close()
p.terminate()'''
