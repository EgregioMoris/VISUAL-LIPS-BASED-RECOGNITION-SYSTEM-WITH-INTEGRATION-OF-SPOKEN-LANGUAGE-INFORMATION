"""
extract_video.py permette di ritagliare a partire da uno o più file .mp4 un certo numero di video di tot secondi contenenti una persona
È necessario avere ffmpeg installato

Carmine Ippolito
"""
import os
from tqdm import tqdm
import multiprocessing
import dlib
import cv2
import numpy as np
from scipy.spatial import distance
import subprocess

VIDEO_INPUT_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\Video_interi\Francese\Uomo\Under 30"
PREDICTOR_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\shape_predictor_68_face_landmarks.dat"
DURATION = 10
TRIMMED_VIDEOS = 5
VIDEO_OUTPUT_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\Video_10_secondi\Francese\Uomo\Under 30"
TXT_ERORR_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset"


def extract_video(root, filename):
    # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # Calculate the fps and the number of frames of a trimmed video
    fps = int(filename.split('.')[0].split('_')[4])
    frames_number = fps * DURATION
    positions = []

    frame_count = 0
    # Skip the first 10 frames of the input video to avoid transitions and special effects
    frame_index = -10
    distance_array = []

    cap = cv2.VideoCapture(root + '/' + filename)
    while cap.isOpened() and len(positions) != TRIMMED_VIDEOS:
        # Check 5 extra frame to avoid occlusion
        while frame_index < frames_number + 5:
            ret, image = cap.read()
            # The frame is read correctly if ret is True
            if ret != True:
                break

            face = detector(image, 0)
            if len(face) != 1:
                frame_count += 1
                # Skip the next 5 frames of the input video
                frame_index = -5
                distance_array.clear()
                continue

            landmarks = predictor(image, face[0])

            # Calculate the Euclidean distances between the 62nd and 68th landmarks for the trimmed video
            if 0 <= frame_index < frames_number:
                # Convert the landmark (x, y) coordinates to a NumPy array
                # Initialize the list of (x, y) coordinates
                coordinates = np.zeros((68, 2), dtype="int")
                # Loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y) coordinates
                for i in range(0, 68):
                    coordinates[i] = (landmarks.part(i).x, landmarks.part(i).y)

                distance_array.append(round(distance.euclidean(
                    [coordinates[62][0], coordinates[62][1]], [coordinates[66][0], coordinates[66][1]])))

            # Check to avoid still images (standard_deviation) and sections with little movement in the lip area (relative_standard_deviation)
            if len(distance_array) == fps * 2:
                standard_deviation = np.std(distance_array)
                relative_standard_deviation = (
                    standard_deviation / np.mean(distance_array)) * 100
                distance_array.clear()
                if standard_deviation < 1 or relative_standard_deviation < 30:
                    frame_count += 1
                    frame_index = 0
                    continue

            frame_count += 1
            frame_index += 1

        if ret != True:
            break

        frame_index = 0
        distance_array.clear()
        positions.append((frame_count - frames_number - 5) / fps)

    cap.release()

    if len(positions) == TRIMMED_VIDEOS:
        # Save the trimmed videos in .mp4 files
        os.chdir(VIDEO_OUTPUT_PATH)
        for i, position in enumerate(positions):
            # Need to have ffmpeg installed
            subprocess.run(["ffmpeg", "-ss", str(position), "-i", root + "\\" + filename, "-c:v", "libx264",
                           "-an", "-frames:v", str(frames_number), filename.split('.')[0] + '_' + str(i + 1) + ".mp4"])
    else:
        raise ValueError(filename)


if __name__ == "__main__":
    # Count the number of videos and save their path
    N = 0
    videos = []
    for root, dirs, files in os.walk(VIDEO_INPUT_PATH):
        for filename in files:
            if filename.endswith(".mp4"):
                print(filename)
                N += 1
                videos.append((root, filename))

    print("Extracting video")
    print(N)
    # Initialize the progress bar
    pbar = tqdm(total=N)
    errors = []

    def update(*a):
        pbar.update()

    def exception_handler(err):
        errors.append(str(err))

    # Start cpu_count() worker processes
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for root, filename in videos:
            # Execute extract_video asynchronously
            pool.apply_async(extract_video, (root, filename),
                             callback=update, error_callback=exception_handler)
        pool.close()
        pool.join()
    pbar.close()
    print("END")
    if errors:
        # Save the errors in a .txt file
        os.chdir(TXT_ERORR_PATH)
        with open("extract_video_errors.txt", mode='w', newline='') as txtfile:
            print("Unextracted videos\n", file=txtfile)
            for error in errors:
                print(error, file=txtfile)

        print("Unable to extract the lip area of " +
              str(N - pbar.n) + " videos.")
        print("See the errors.txt file for more information.")
