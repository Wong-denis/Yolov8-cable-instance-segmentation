from ultralytics import YOLO
import cv2
import numpy as np

# model = YOLO("model.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
model = YOLO('runs/segment/train2/weights/best.pt')

def predict_on_image(result, img, conf):
    # result = model(img, conf=conf)[0]

    # detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

    # segmentation
    masks = result.masks.masks.cpu().numpy()     # masks, (N, H, W)
    # masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
    # rescale masks to original image
    # masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
    # masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    # return boxes, masks, cls, probs
    return boxes, masks

# from cv2
img = cv2.imread("test_data/cable_86.jpg")
img_copy = img.copy()
results = model.predict(source=img, save=True, exist_ok=True)  # save predictions as labels
print("This is the length of the result", len(results))
img = img_copy.copy()
for result in results:
    boxes, masks = predict_on_image(result, img, conf=0.6)
    # masks = result.masks
    # masks = masks.reshape(img.shape)
    w,h = img.shape[0], img.shape[1]
    for m in range(masks.shape[0]):
        mask = masks[m,:,:].astype(np.uint8)
        mask = np.repeat(mask.reshape((w,h,1)), 3, axis=2)*255
        # print(mask.shape)

        image_masked = cv2.addWeighted(mask, 0.5, img, 1, 0)

        cv2.imshow("masks", image_masked)
        cv2.waitKey()