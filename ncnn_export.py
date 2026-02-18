from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.export(format="ncnn")
ncnn_model = YOLO("./yolo11n_ncnn_model")
results = ncnn_model("https://ultralytics.com/images/bus.jpg")