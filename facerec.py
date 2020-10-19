import numpy as np
import cv2
import imutils
import time
import tkinter
import PIL.Image, PIL.ImageTk


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.resizable(False, False)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)


        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=300, height=200)
        self.canvas.grid(row=0,column=0)

        self.canvas2 = tkinter.Canvas(window, width=300, height=200)
        self.canvas2.grid(row=1,column=0)

        self.canvas3 = tkinter.Canvas(window, width=300, height=200)
        self.canvas3.grid(row=1,column=1)

        self.canvas4 = tkinter.Canvas(window, width=300, height=200)
        self.canvas4.grid(row=0,column=1)

        self.canvas5 = tkinter.Canvas(window, width=600, height=400)
        self.canvas5.grid(row=0, column=2, columnspan=2, rowspan=2,
               sticky=tkinter.W)

        self.Button_snap = tkinter.Button(window, text='Snapshot', command=self.snapshot, anchor=tkinter.CENTER, width=50)
        self.Button_snap.grid(row=2, column=2, sticky=tkinter.E)
        # Button that lets the user take a snapshot


        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret2, frame2 = self.vid.get_frame_binary()
        ret3, frame3 = self.vid.get_frame_ero()
        ret4, frame4 = self.vid.get_frame_dilt()
        ret5, frame5 = self.vid.detection()

        if ret2 and ret3 and ret4 and ret5:
            cv2.imwrite("frame_binary-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", frame2)
            cv2.imwrite("frame_erode-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", frame3)
            cv2.imwrite("frame_diltate-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", frame4)
            cv2.imwrite("frame_face-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame5, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        ret2, frame2 = self.vid.get_frame_binary()
        ret3, frame3 = self.vid.get_frame_ero()
        ret4, frame4 = self.vid.get_frame_dilt()
        ret5, frame5 = self.vid.detection()

        if ret and ret2 and ret3 and ret4 and ret5:
            self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame2))
            self.canvas2.create_image(150, 100, image=self.photo2)
            self.canvas2.create_text(50, 15, fill="blue", font="Times 20 italic bold",
                                    text="Binary")
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(150, 100, image=self.photo)
            self.canvas.create_text(55, 15, fill="blue", font="Times 20 italic bold",
                                    text="Webcam")

            self.photo3 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame3))
            self.canvas3.create_image(150, 100, image=self.photo3)
            self.canvas3.create_text(50, 15, fill="blue", font="Times 20 italic bold",
                                    text="Erode")
            self.photo4 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame4))
            self.canvas4.create_image(150, 100, image=self.photo4)
            self.canvas4.create_text(50, 15, fill="blue", font="Times 20 italic bold",
                                    text="Dilate")
            self.photo5 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame5))
            self.canvas5.create_image(250, 200, image=self.photo5)
            self.canvas5.create_text(165, 15, fill="blue", font="Times 20 italic bold",
                                    text="Detection and Recognition")
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt',
                                       'mod.caffemodel')
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame,(300,200))

            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)

    def detection(self):
        if self.vid.isOpened():
            ret, image = self.vid.read()

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return(ret,cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    def get_frame_binary(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (300, 200))
            if ret:
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret, None)

    def get_frame_ero(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (300, 200))
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                frame= cv2.erode(frame, kernel, iterations=2)
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret, None)

    def get_frame_dilt(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (300, 200))
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                frame = cv2.dilate(frame, kernel, iterations=2)
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)
            else:
                return (ret, None)
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()




# Create a window and pass it to the Application object
#App(tkinter.Tk(), "Tkinter and OpenCV")
svm = cv2.ml.SVM_create()
svm = cv2.ml.SVM_load('svm.xml')

