from ultralytics import YOLO

### train 
cfg = 'yolov8l-seg_custom.yaml'
# model = YOLO("yolov8x-seg_custom.yaml")  # build a new model from scratch
model = YOLO("yolov8l-seg.pt")  # load a pretrained model (recommended for training)
results = model.train(data="custom_data.yaml", epochs=1000, workers=1, batch=6,imgsz=512, save_period=200)  # train the model

# ### predict 
# model = YOLO("runs/segment/train8/weights/best.pt")

# #model.predict(source="0") # accepts all formats - img/folder/vid.*(mp4/format). 0 for webcam

# model.predict(source="test_data") # Display preds. Accepts all yolo predict arguments