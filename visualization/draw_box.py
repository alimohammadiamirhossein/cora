import cv2, argparse

bbox_start = (-1, -1)
bbox_end = (-1, -1)
drawing = False  




def draw_bbox(event, x, y, flags, param):
    global bbox_start, bbox_end, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        bbox_start = (x, y)
        bbox_end = bbox_start

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image = image_copy.copy()
            cv2.rectangle(image, bbox_start, (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox_end = (x, y)
        cv2.rectangle(image, bbox_start, bbox_end, (0, 0, 255), 2)
        # print(f"BBox Coordinates: Start: {bbox_start}, End: {bbox_end}\n")
        print(f"bbx_start_point= ({bbox_start[0]}, {bbox_start[1]}), ")
        print(f"bbx_end_point= ({bbox_end[0]}, {bbox_end[1]})")



def main(image_path):
    global bbox_start, bbox_end, drawing, image_copy, image
    image = cv2.resize(cv2.imread(image_path), (512, 512))
    image_copy = image.copy()

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_bbox)

    while True:
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive image masking")
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()
    main(args.image_path)