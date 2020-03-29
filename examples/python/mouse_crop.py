import cv2
import numpy as np

cropping = False
FinishROI = 0
roi = []
coordinates = []
x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = None
oriImage = None


def mouse_crop(image_np,image_raw, num_of_crop, text=''):

    global x_start, y_start, x_end, y_end, cropping, roi, FinishROI, coordinates

    cropping = False
    FinishROI = 0
    roi = []
    coordinates = []
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    def mouse_crop_aux(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping, roi, FinishROI, coordinates

        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False  # cropping is finished

            refPoint = [(x_start, y_start), (x_end, y_end)]

            if len(refPoint) == 2:  # when two points were found

                if x_start > x_end:
                    tmp = x_start
                    x_start = x_end
                    x_end = tmp

                if y_start > y_end:
                    tmp = y_start
                    x_start = y_end
                    y_end = tmp


                # roi.append(imageRaw[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]])
                roi.append(imageRaw[y_start:y_end, x_start:x_end])
                coordinates.append([x_start, y_start, x_end, y_end])
                #cv2.imshow("Cropped", roi[-1])
                FinishROI += 1



    image =image_np
    oriImage = image.copy()
    imageRaw = image_raw.copy()
    cv2.namedWindow("image")

    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 255, 0)
    cv2.putText(image, text, (10, 50), font, 1, color, 3)

    cv2.imshow("image", image)

    while FinishROI<num_of_crop:
        cv2.setMouseCallback("image", mouse_crop_aux)

        i = image.copy()

        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        cv2.waitKey(1)

    # close all open windows
    cv2.destroyAllWindows()

    # roi_resize=[]
    # for single_roi in roi:
    #     roi_resize.append(cv2.resize(single_roi, ( 100, 32)))

    return roi, coordinates


if __name__ == '__main__':
    image = cv2.imread('demo_image/Screenshot_20200322-140545_WhatsApp.jpg')
    oriImage = image.copy()
    R =mouse_crop(image)
    a=1
    # while True:
    #     i = image.copy()
    #
    #     if not cropping:
    #         cv2.imshow("image", image)
    #
    #     elif cropping:
    #         cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    #         cv2.imshow("image", i)
    #         # return i;
    #
    #     cv2.waitKey(1)
    #
    # # close all open windows
    # cv2.destroyAllWindows()