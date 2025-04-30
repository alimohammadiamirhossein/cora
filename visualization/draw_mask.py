import cv2, numpy as np, argparse
from PIL import Image
drawing = False
ix, iy = -1, -1
mask = None
brush_size = 25
cur_x, cur_y = -100, -100

def draw_mask(event, x, y, flags, param):
    global drawing, mask, brush_size, cur_x, cur_y
    cur_x, cur_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), brush_size, (255, 255, 255), -1)

def main(image_path):
    global mask, brush_size, cur_x, cur_y

    image = Image.open(image_path).convert('RGB').resize((512, 512))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image is None:
        print("Could not open or find the image"); return

    mask = np.zeros_like(image)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_mask)

    print("[q] save&quit   [+]/[-] brush size   [r] redo")

    while True:
        blended = cv2.addWeighted(image, 0.7, mask, 0.3, 0)


        cv2.circle(blended, (cur_x, cur_y), brush_size, (0, 255, 0), 1)

        cv2.imshow("image", blended)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            brush_size += 5
        elif key == ord('-') and brush_size > 1:
            brush_size = max(1, brush_size - 5)
        elif key == ord('r'):
            mask[:] = 0

    # save mask to the same directory as the image
    mask_path = image_path.rsplit('.', 1)[0] + "_mask.png"
    cv2.imwrite(mask_path, mask)
    print("Mask saved to {}".format(mask_path))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive image masking")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--brush_size", type=int, default=25)
    args = parser.parse_args()

    brush_size = max(1, args.brush_size)
    main(args.image_path)
