"""
extract_data.py permette di estrarre la zona labiale e calcolare la distanza di Manhattan e quella euclidea tra i landmark d'interesse (mouth_intern o mouth_extern)
a partire da uno o più file .mp4

Carmine Ippolito
"""
import os
from tqdm import tqdm
import multiprocessing
import dlib
import cv2
import numpy as np
from scipy.spatial import distance
import csv

VIDEO_INPUT_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\Video_10_secondi"
PREDICTOR_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\shape_predictor_68_face_landmarks.dat"
FRAME_SIZE = (300, 200)
# Draw the landmarks on the ROI
DRAW_LANDMARKS = False
# Extract the coordinates of mouth_intern [60:68] or mouth_extern [48:60]
MOUTH_COORDINATES = (60, 68)
VIDEO_OUTPUT_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\Solo_labbra"
CSV_MANHATTAN_OUTPUT_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\Csv_distanza_di_Manhattan"
CSV_EUCLIDEAN_OUTPUT_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco\Dataset\Csv_distanza_euclidea"
TXT_ERORR_PATH = r"G:\.shortcut-targets-by-id\13TCYIMgFy4pRrdq3WYYh3Ctpv67MbwS5\Ricco"


def extract_data(root, filename):
    # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    video_array = []
    distance_manhattan_matrix = []
    distance_euclidean_matrix = []
    T = False

    cap = cv2.VideoCapture(root + '/' + filename)
    while cap.isOpened():
        ret, image = cap.read()
        # The frame is read correctly if ret is True
        if ret != True:
            break

        face = detector(image, 0)
        if len(face) != 1:
            if T == False:
                T = True
                continue
            cap.release()
            raise ValueError(filename)

        landmarks = predictor(image, face[0])

        # Convert the landmark (x, y) coordinates to a NumPy array
        # Initialize the list of (x, y) coordinates
        coordinates = np.zeros((68, 2), dtype="int")
        # Loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y) coordinates
        for i in range(0, 68):
            coordinates[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Extract the coordinates of the mouth
        (x, y, w, h) = cv2.boundingRect(np.array([coordinates[48:68]]))
        # Extract and resize the ROI according to frame_size
        roi = image[y - 10: y + h + 10, x - 10: x + w + 10]
        roi = cv2.resize(roi, dsize=(FRAME_SIZE[0], FRAME_SIZE[1]),
                         interpolation=cv2.INTER_CUBIC)

        if DRAW_LANDMARKS != False or CSV_MANHATTAN_OUTPUT_PATH != None or CSV_EUCLIDEAN_OUTPUT_PATH != None:
            scaled_coordinates = []
            # Extract the coordinates of mouth_intern [60:68] or mouth_extern [48:60]
            for (xc, yc) in coordinates[MOUTH_COORDINATES[0]:MOUTH_COORDINATES[1]]:
                # Scale xc and yc according to frame_size
                scaled_xc = round(((xc - (x - 10)) * FRAME_SIZE[0]) / (w + 20))
                scaled_yc = round(((yc - (y - 10)) * FRAME_SIZE[1]) / (h + 20))
                scaled_coordinates.append((scaled_xc, scaled_yc))

                if DRAW_LANDMARKS != False:
                    # Draw the landmarks on the ROI
                    cv2.circle(roi, (scaled_xc, scaled_yc), 1, (0, 0, 255), -1)

            if CSV_MANHATTAN_OUTPUT_PATH != None:
                # Calculate Manhattan distance
                distance_array = [round(distance.cityblock([x1, y1], [x2, y2]))
                                  for i, (x1, y1) in enumerate(scaled_coordinates) for (x2, y2) in scaled_coordinates[i + 1:]]
                if T == True:
                    if (len(distance_manhattan_matrix) != 0):
                        distance_manhattan_matrix.append(distance_manhattan_matrix[-1])
                    else:
                        distance_manhattan_matrix.append(distance_array)
                distance_manhattan_matrix.append(distance_array)

            if CSV_EUCLIDEAN_OUTPUT_PATH != None:
                # Calculate Euclidean distance
                distance_array = [round(distance.euclidean([x1, y1], [x2, y2]))
                                  for i, (x1, y1) in enumerate(scaled_coordinates) for (x2, y2) in scaled_coordinates[i + 1:]]
                if T == True:
                    if (len(distance_euclidean_matrix) != 0):
                        distance_euclidean_matrix.append(distance_euclidean_matrix[-1])
                    else:
                        distance_euclidean_matrix.append(distance_array)
                distance_euclidean_matrix.append(distance_array)

        if T == True:
            if (len(video_array) != 0):
                video_array.append(video_array[-1])
            else:
                video_array.append(roi)
        video_array.append(roi)

        if T == True:
            T = False

    cap.release()

    if VIDEO_OUTPUT_PATH != None:
        # Save the ROI in a .avi file
        os.chdir(VIDEO_OUTPUT_PATH)
        fps = int(filename.split('.')[0].split('_')[4])
        outputVideo = cv2.VideoWriter((filename.split(
            '.')[0]) + "_m" + ".avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, FRAME_SIZE)
        for image in video_array:
            outputVideo.write(image)
        outputVideo.release()

    def save_csv(csv_output_path, distance_matrix):
        # Save the distances in a .csv file
        os.chdir(csv_output_path)
        with open(filename.split('.')[0] + "_m" + ".csv", mode='w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for row in distance_matrix:
                spamwriter.writerow(row)

    if CSV_MANHATTAN_OUTPUT_PATH != None:
        save_csv(CSV_MANHATTAN_OUTPUT_PATH, distance_manhattan_matrix)

    if CSV_EUCLIDEAN_OUTPUT_PATH != None:
        save_csv(CSV_EUCLIDEAN_OUTPUT_PATH, distance_euclidean_matrix)


if __name__ == "__main__":
    # Count the number of videos and save their path
    N = 0
    videos = []
    for root, dirs, files in os.walk(VIDEO_INPUT_PATH):
        for filename in files:
            if filename.endswith(".mp4"):
                N += 1
                videos.append((root, filename))

    print("Extracting data")
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
            # Execute extract_data asynchronously
            pool.apply_async(extract_data, (root, filename),
                             callback=update, error_callback=exception_handler)
        pool.close()
        pool.join()
    pbar.close()

    if errors:
        # Save the errors in a .txt file
        os.chdir(TXT_ERORR_PATH)
        with open("extract_data_errors.txt", mode='w', newline='') as txtfile:
            print("Unextracted data\n", file=txtfile)
            for error in errors:
                print(error, file=txtfile)

        print("Unable to extract the lip area of " +
              str(N - pbar.n) + " videos.")
        print("See the errors.txt file for more information.")
