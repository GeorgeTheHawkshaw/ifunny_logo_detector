#import packages
import numpy as np
import argparse
import imutils
import glob
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-t","--template", required=True, help="Path to the template image")
ap.add_argument("-i","--images", required=True, help="Path to image")
ap.add_argument("-v", "--visualize",help="Flag to indicate visualization")
args = vars(ap.parse_args())

#load image, convert to grayscale, detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50,200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template",template)

#loop over images to find template
for imagePath in glob.glob(args["images"] + "/*.png"):
    #load image, convert to gray, keep track of matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    #loop over scales of image
    for scale in np.linspace(0.2,1.0,20)[::-1]:
        #resize along scale, and keep track of ratio
        resized = imutils.resize(gray,width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        #if image is smaller than template, break
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        #detect edges
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        #check if visualization 
        if args.get("visualize", False):
            #draw rectangle over detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0,0,255),2)
            cv2.imshow("Visualize",clone)
            cv2.waitKey(0)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# draw a rectangle
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.imwrite("new.png",image)


    # crop
    image_obj = Image.open("new.png")
    cropped_image = image_obj.crop((0,0,endX,startY))
    cropped_image.save("cropped.jpg")
    cropped_image.show()
    cv2.waitKey(0)
