from imutils.video import FPS
import cv2
import dlib
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
from tracking_with_object.draw import draw_path, show_list


# information of each face in a frame
class Person:
    def __init__(self, name, closest_distance):
        self.name = name
        self.distance = closest_distance


class TrackingObject:
    def __init__(self):
        # list of persons the tracking_object contains
        self.persons = []
        # list of positions
        self.positions = []

    def update_positions(self, tracker):

        # update position list
        start_x, start_y, end_x, end_y = position_of_tracker(tracker)
        self.positions.append((start_x, start_y, end_x, end_y))

    def __str__(self):
        result = ""
        for i in range(len(self.persons)):
            result += self.persons[i].name + '- '
        # for i in range(len(self.positions)):
        #     result += '\n' + str(self.positions[i][0]) + ',' + str(self.positions[i][1])
        return result


# this function is used to get position of tracker
def position_of_tracker(tracker):
    # get position of tracker
    start_x = tracker.get_position().left()
    start_y = tracker.get_position().top()
    end_x = tracker.get_position().right()
    end_y = tracker.get_position().bottom()

    return start_x, start_y, end_x, end_y


# check whether a face with locations as parameter is one of the faces in trackers or not
# this function is very accurate because it just return if two faces are very close together
# two rectangles are overlapped ==> found a matched face
def is_the_same(top, right, bottom, left, trackers, persons):
    # get central of the face
    cx1 = (left + right) / 2
    cy1 = (top + bottom) / 2

    # loop over each tracker in trackers to compare
    for face_id in trackers.keys():
        # get central of the face in tracker
        tracker = trackers[face_id]
        start_x, start_y, end_x, end_y = position_of_tracker(tracker)
        cx2 = (start_x + end_x) / 2
        cy2 = (start_y + end_y) / 2

        # main step: compare two faces
        if ((left <= cx2 <= right) and
                (top <= cy2 <= bottom) and
                (start_x <= cx1 <= end_x) and
                (start_y <= cy1 <= end_y)):
            return face_id, persons[face_id]
    return None


# check whether a face with locations as parameter is one of the faces in trackers or not
# this function is not very accurate because it just return if two faces are close together
# two rectangles have an area in common ==> found a matched face
def is_the_same1(top, right, bottom, left, trackers, persons):
    li = []
    # loop over each tracker in trackers to compare
    for face_id in trackers:
        # get position of tracker
        tracker = trackers[face_id]
        start_x, start_y, end_x, end_y = position_of_tracker(tracker)

        # main step: compare two faces
        if start_y <= top <= end_y:
            if start_x <= left <= end_x:
                li.append((face_id, (end_x - left)*(end_y - top)))
            elif start_x <= right <= end_x:
                li.append((face_id, (right - start_x)*(end_y - top)))
            elif start_x >= left and end_x <= right:
                li.append((face_id, (end_x - start_x)*(end_y - top)))
            elif start_x <= left and end_x >= right:
                li.append((face_id, (right - left)*(end_y - top)))
        elif start_y <= bottom <= end_y:
            if start_x <= left <= end_x:
                li.append((face_id, (end_x - left)*(bottom - start_y)))
            elif start_x <= right <= end_x:
                li.append((face_id, (right - start_x)*(bottom - start_y)))
            elif start_x >= left and end_x <= right:
                li.append((face_id, (end_x - start_x)*(bottom - start_y)))
            elif start_x <= left and end_x >= right:
                li.append((face_id, (right - left) * (bottom - start_y)))
        elif start_x <= left <= end_x:
            if start_y >= top and end_y <= bottom:
                li.append((face_id, (end_y - start_y)*(end_x - left)))
            elif start_y <= top and end_y >= bottom:
                li.append((face_id, (bottom - top)*(bottom - start_y)))
        elif start_x <= right <= end_x:
            if start_y >= top and end_y <= bottom:
                li.append((face_id, (end_y - start_y)*(right - start_x)))
            elif start_y <= top and end_y >= bottom:
                li.append((face_id,  (bottom - top)*(right - start_x)))

    # get the tracker with the biggest area in common
    li.sort(key=lambda x: x[1])
    if len(li) > 0:
        face_id = li[-1][0]
        return face_id, persons[face_id]
    return None


# train knn model and the save the model
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    x = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_locations) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                x.append(face_recognition.face_encodings(image, known_face_locations=face_locations)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


class FaceRecognition:

    def __init__(self, model_path):
        # dictionary contains trackers in each frame
        # {face_id1: tracker1, face_id2: tracker2, ...}
        self.trackers = {}

        # dictionary contains TrackingObject object
        # {face_id1: tracking_object1, face_id2: tracking_object2, ...}
        self.tracking_objects = {}

        # the last current trackers used to solve: poor quality tracking and poor locations
        # {face_id1: last_tracker1, face_id2: last_tracker2, ...}
        self.last_trackers = {}

        # the last current persons used to solve: poor quality tracking and poor locations
        # {face_id1: last_person1, face_id2: last_person2, ...}
        self.last_persons = {}

        # dictionary contains Person object (name, closest_distance)
        # {face_id_1: object1, face_id_2: object2, ...}
        self.persons = {}

        # ----
        self.out_left_faces = {}
        self.out_top_faces = {}
        self.out_right_faces = {}
        self.out_bottom_faces = {}

        # create new face_id for new face appear in image
        self.current_id = 0

        # count frame
        self.frame_count = 0

        # get the knn model
        with open(model_path, 'rb') as f:
            self.knn_clf = pickle.load(f)

    # use to recover face due to poor tracking quality
    def get_same_face(self, type, top, right, bottom, left, trackers, persons):
        if type == 0:
            same_face_object = is_the_same(top, right, bottom, left, trackers, persons)
        elif type == 1:
            same_face_object = is_the_same1(top, right, bottom, left, trackers, persons)

        if same_face_object is not None:
            face_id, person = same_face_object
            # add person into self.person list (we will add tracker later (restart tracking before adding))
            self.persons[face_id] = person

            # delete tracker and person of face_id in self.last_trackers and self.last_persons
            self.last_trackers.pop(face_id)
            self.last_persons.pop(face_id)

            return face_id

    def delete_face_from_tracking_object(self, face_id):
        self.last_trackers.pop(face_id)
        self.last_persons.pop(face_id)
        self.tracking_objects.pop(face_id)

    # check a deleted tracker to control the face that have high probability to go out of camera
    # if it have ability to go out, add it into self.out_faces
    def check_deleted_tracker(self, face_id):
        left, top, right, bottom = position_of_tracker(self.last_trackers[face_id])

        if face_id in self.out_left_faces.keys():
            self.out_left_faces[face_id].append(left)

            # when we have enough 5 positions
            if len(self.out_left_faces[face_id]) == 5:
                x1 = sum(self.out_left_faces[face_id]) / 5
                if x1 < 10:
                    self.delete_face_from_tracking_object(face_id)
                self.out_left_faces.pop(face_id)

        elif face_id in self.out_top_faces.keys():
            self.out_top_faces[face_id].append(top)

            # when we have enough 5 positions
            if len(self.out_top_faces[face_id]) == 5:
                x2 = sum(self.out_top_faces[face_id]) / 5
                if x2 < 10:
                    self.delete_face_from_tracking_object(face_id)
                self.out_top_faces.pop(face_id)

        elif face_id in self.out_right_faces.keys():
            self.out_right_faces[face_id].append(right)

            # when we have enough 5 positions
            if len(self.out_right_faces[face_id]) == 5:
                y1 = sum(self.out_right_faces[face_id]) / 5
                if y1 > 630:
                    self.delete_face_from_tracking_object(face_id)
                self.out_right_faces.pop(face_id)

        elif face_id in self.out_bottom_faces.keys():
            self.out_bottom_faces[face_id].append(bottom)

            # when we have enough 5 positions
            if len(self.out_bottom_faces) == 5:
                y2 = sum(self.out_bottom_faces) / 5
                if y2 > 470:
                    self.delete_face_from_tracking_object(face_id)
                self.out_bottom_faces.pop(face_id)

        else:
            if left < 0:
                self.out_left_faces[face_id] = [left]
            elif top < 0:
                self.out_top_faces[face_id] = [top]
            elif right > 640:
                self.out_right_faces[face_id] = [right]
            elif bottom > 480:
                self.out_bottom_faces[face_id] = [bottom]

    def predict_and_track(self, image, distance_threshold=0.4):
        # increase frame_count
        self.frame_count += 1

        # store locations of new faces and ids
        new_face_locations = []
        new_face_ids = []

        # store tracker that have poor quality
        face_id_to_delete_in_trackers = []
        # save trackers removed to recover in the next frame
        # [(Person_1, start_x1, start_y1, end_x1, end_y2), (Person_2, start_x2, start_y2, end_x2, end_y2), ...]
        last_face_locations = []

        # update last_trackers
        for face_id in self.last_trackers:
            # update
            self.last_trackers[face_id].update(image)

            # check to delete tracking_objects
            self.check_deleted_tracker(face_id)

            # get the position of poor trackers that were deleted in the previous frame
            # (not recovered in the previous frame)----------
            print('[INFO] face_id {} => last_tracking position: {}'.
                  format(face_id, self.last_trackers[face_id].get_position()))

        # remove trackers that have poor quality from the list of trackers
        # save trackers that have poor quality into self.last_trackers and respondent self.last_persons
        for face_id in self.trackers.keys():
            # get position of tracker before poor tracking filter stage(just in self.trackers-
            # not in self.last_trackers)----------
            print('[INFO] face_id {} => tracking position (before poor tracking filter stage): {}'.
                  format(face_id, self.trackers[face_id].get_position()))

            # update position of trackers
            tracking_quality = self.trackers[face_id].update(image)
            # get the quality of tracking ----------
            print('[INFO] face_id {} => tracking quality: {}'.format(face_id, tracking_quality))

            if tracking_quality < 10:
                # store face_id to delete trackers
                face_id_to_delete_in_trackers.append(face_id)

                # store the position of deleted face to recovery later
                start_x, start_y, end_x, end_y = position_of_tracker(self.trackers[face_id])

                # store people of trackers to recover
                last_face_locations.append((face_id, self.persons[face_id], start_x, start_y, end_x, end_y))

        # remove trackers that have poor quality from the list of trackers
        # and remove respondent person in the list of person
        # add removed tracker and removed person into last_trackers and last_persons (delete if recovering successfully)
        for face_id in face_id_to_delete_in_trackers:
            self.last_trackers[face_id] = self.trackers[face_id]
            self.last_persons[face_id] = self.persons[face_id]
            self.trackers.pop(face_id)
            self.persons.pop(face_id)

        # get position of all trackers remain after filter poor tracker stage (just in self.trackers not
        # in self.last_trackers) ----------
        for face_id in self.trackers.keys():
            print('[INFO] face_id {} => tracking position (after filter poor tracker): {}'
                  .format(face_id, self.trackers[face_id].get_position()))

        # get locations of all faces in the frame
        face_locations = face_recognition.face_locations(image)

        # check face locations ----------
        print('[INFO] face locations: ', face_locations)

        # loop on faces of current frame to find new faces
        for top, right, bottom, left in face_locations:
            # check whether a face is new or not by comparing with current trackers in self.trackers
            obj = is_the_same(top, right, bottom, left, self.trackers, self.persons)
            matched_face_id = obj[0] if (obj is not None) else None

            # check with the old face (we removed because the tracking quality is poor)
            # comparing with current trackers in self.last_tracker
            # poor tracking quality- good face locations (the found face and the face in tracker of self.last_trackers
            # are still overlapped)
            if matched_face_id is None:
                matched_face_id = self.get_same_face(0, top, right, bottom, left, self.last_trackers, self.last_persons)

            # still check with the old face (we removed because the tracking quality is poor)
            # comparing with current trackers in self.last_tracker
            # poor tracking quality- bad face locations (the two faces are just still in common, not overlapped)
            if matched_face_id is None:
                matched_face_id = self.get_same_face(1, top, right, bottom, left, self.last_trackers, self.last_persons)

            # still check with the old face (we removed because the tracking quality is poor)
            # poor tracking quality- no face locations in some next frames
            if matched_face_id is None:
                # check if two faces are overlapped
                matched_face_id = self.get_same_face(0, top, right, bottom, left, self.last_trackers, self.last_persons)

                # check if the two faces are in common
                if matched_face_id is None:
                    matched_face_id = self.get_same_face(1, top, right, bottom, left, self.last_trackers, self.last_persons)

            # the last check to recover face_id
            # compare the number of tracking_objects with the number of trackers
            if matched_face_id is None:
                face_ids_in_tracking_objects = set(self.tracking_objects.keys())
                face_ids_in_trackers = set(self.trackers.keys())
                recover_face_id = face_ids_in_tracking_objects - face_ids_in_trackers
                if len(recover_face_id) == 1:
                    matched_face_id = list(recover_face_id)[0]
                    # add person into self.person list (we will add tracker later (restart tracking before adding))
                    self.persons[matched_face_id] = self.last_persons[matched_face_id]

                    # delete tracker and person of face_id in self.last_trackers and self.last_persons
                    self.last_trackers.pop(matched_face_id)
                    self.last_persons.pop(matched_face_id)

            # new face
            if matched_face_id is None:
                # increase current_id
                self.current_id += 1

                # start to track a new face
                tracker = dlib.correlation_tracker()
                tracker.start_track(image, dlib.rectangle(left, top, right, bottom))
                self.trackers[self.current_id] = tracker

                # an unknown person is created and store
                self.persons[self.current_id] = Person("unknown", -1)
                new_face_locations.append((top, right, bottom, left))
                new_face_ids.append(self.current_id)

            # if we recognize that this face is tracking, continue to track it, but restart tracking with new position
            # after 20 frames, we re-track and recognize face again
            else:
                # if tracking quality is ok and we just re-track it with current face_location
                if matched_face_id in self.trackers.keys():
                    self.trackers[matched_face_id].start_track(image, dlib.rectangle(left, top, right, bottom))

                # we recover the old face, tracking with the old face_id (create again the key match_face_id)
                else:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(left, top, right, bottom))
                    self.trackers[matched_face_id] = tracker

                # recognize face again if we still do not know the face or more than 20 frames
                if self.persons[matched_face_id].distance == -1 or self.frame_count >= 20:
                    new_face_locations.append((top, right, bottom, left))
                    new_face_ids.append(matched_face_id)

                    # reset self.frame when it reaches 20 frames
                    if self.frame_count >= 20:
                        self.frame_count = 0

        # get position of all trackers remain after recover and finding new face stage (just in self.trackers-
        # not in self.last_trackers) ----------
        for face_id in self.trackers.keys():
            print('[INFO] face_id: {} => tracking position (after recover stage- already updated): {}'.
                  format(face_id, self.trackers[face_id].get_position()))

        # predict name of new faces
        if(len(new_face_locations)) > 0:
            # Find encodings for new faces in image
            faces_encodings = face_recognition.face_encodings(image, known_face_locations=new_face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)

            are_matches = [closest_distances[0][i][0] < distance_threshold for i in range(len(new_face_locations))]
            for face_id, name, flag_match, location, distance in zip(new_face_ids, self.knn_clf.predict(faces_encodings),
                                                                     are_matches, new_face_locations, closest_distances[0]):
                # save Person object into the persons list
                person = Person(name, distance[0]) if flag_match else Person("unknown", -1)
                self.persons[face_id] = person

                # update persons list of tracking_object with face_id
                # create a new tracking object and add into tracking_object list
                if face_id not in self.tracking_objects.keys():

                    # create a tracking_object
                    tracking_object = TrackingObject()

                    # update information of tracking_object
                    tracking_object.persons.append(person)

                    # add tracking_object into tracking_objects
                    self.tracking_objects[face_id] = tracking_object

                # tracking object of this face_id already exist
                else:
                    self.tracking_objects[face_id].persons.append(person)

        # get all tracking faces and return
        return [(self.persons[face_id].name, face_id, self.trackers[face_id].get_position()) for face_id in self.trackers.keys()]

    def tracking_object_management(self):

        for face_id in self.trackers.keys():
            # update positions of tracking object
            self.tracking_objects[face_id].update_positions(self.trackers[face_id])

        # get information of each tracking objects ----------
        for face_id in self.tracking_objects.keys():
            print('[INFO] face_id: ', face_id)
            print('[INFO] list of name: ', self.tracking_objects[face_id])
            print('[INFO] the number of positions: ', len(self.tracking_objects[face_id].positions))


# use this function to draw boxes and names of people in frame
def draw(frame, persons):
    for name, face_id, location in persons:
        top, right, bottom, left = int(location.top()), int(location.right()), int(location.bottom()), int(location.left())
        color = (255, 0, 0)

        # Some information computed to draw more flexible
        height = bottom - top
        label_height = int(height/7)
        text_start = int(label_height/4)
        font_size = (height/150)*0.5 if len(name.split(' ')) <= 3 else (height/150)*0.4

        # print font size ----------
        print('[INFO] font size: ', font_size)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

        # Draw a label
        cv2.rectangle(frame, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)

        # Draw name in the label
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + ' ' + str(face_id), (left + text_start, bottom - text_start), font, font_size, (255, 255, 255), 1)


def run_camera():

    # Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    # classifier = train('E:\\data\\tts_face_recognition\\train\\',
    #                    model_save_path='E:\\data\\tts_face_recognition\\model.p', n_neighbors=5)
    print("Training complete!")

    # create a FaceRecognition object to use predict_and_track function
    recognize = FaceRecognition('E:\\data\\tts_face_recognition\\model.p')

    # capture video from camera
    vs = cv2.VideoCapture(0)

    # create object to save video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('D:\\out.avi', fourcc, 5, (640, 480))

    fps = FPS().start()
    # count the number of frames
    real_frame_counter = 0

    # check the sum of each loop's time
    sum = 0

    while True:
        # grab the next frame from the video file
        (grabbed, frame) = vs.read()

        # time to start a frame
        start_frame = time.time()

        # increase real_frame_counter by 1
        real_frame_counter += 1

        # frame from BGR to RGB ordering (dlib needs RGB ordering)
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        persons = recognize.predict_and_track(frame)
        recognize.tracking_object_management()

        # get some necessary information ----------
        print('[INFO] self.frame: ', recognize.frame_count)
        print('[INFO] frame: ', real_frame_counter)
        print('[INFO] name, face_id and trackers: ', persons)
        print('[INFO] current_id: ', recognize.current_id)
        print('[INFO] check trackers: ', len(recognize.trackers))

        draw(frame, persons)

        # write frame into out object
        out.write(frame)

        # time to end a frame
        end_frame = time.time()
        print('[INFO] process in: {}'.format(end_frame - start_frame))
        print('\n')
        sum += end_frame - start_frame

        # show the output frame
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF

        # reset current_id, last_trackers and last_persons if tracking_objects is empty
        if len(recognize.tracking_objects) == 0:
            recognize.current_id = 0
            recognize.last_trackers = {}
            recognize.persons = {}

        # if the "q key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] the number of frames: {}".format(real_frame_counter))
    print('[INFO] sum: ', sum)

    # Release handle to the webcam
    vs.release()
    out.release()
    cv2.destroyAllWindows()

    # draw paths of tracking objects ----------
    for face_id in recognize.tracking_objects.keys():
        # get the path of each axis
        position_list = recognize.tracking_objects[face_id].positions
        start_xs = [x[0] for x in position_list]
        start_ys = [x[1] for x in position_list]
        end_xs = [x[2] for x in position_list]
        end_ys = [x[3] for x in position_list]

        # print position of each axis
        print('[INFO] face_id {} => position list (left): '.format(face_id), end='')
        show_list(start_xs)
        print('[INFO] face_id {} => position list (top): '.format(face_id), end='')
        show_list(start_ys)
        print('[INFO] face_id {} => position list (right): '.format(face_id), end='')
        show_list(end_xs)
        print('[INFO] face_id {} => position list (bottom): '.format(face_id), end='')
        show_list(end_ys)

        # draw
        draw_path(start_xs, face_id, 'left')
        draw_path(start_ys, face_id, 'top')
        draw_path(end_xs, face_id, 'right')
        draw_path(end_ys, face_id, 'bottom')


if __name__ == "__main__":
    run_camera()


'''
Sometime, we can not detection faces, but we can still track faces because we do not remove any trackers in the
list of trackers
when current_id increase, it means that new person appear or the face move very fast (we can not track it)
improve: reset current_id if trackers is empty
self.trackers[face_id].update(image): update the position by its own function (in one frame, it can update many time)
and return the quality of check
'''



