# Total 5361 images
from roboflow import Roboflow
rf = Roboflow(api_key="wAFpXIOS5YhwXaos60c3")
project = rf.workspace("ahmad-essam-gvgib").project("smooking-detection")
version = project.version(4)
dataset = version.download("yolov8")


# Total 1200 images
# from roboflow import Roboflow
# rf = Roboflow(api_key="oYgiUPVwi9JazourafPX")
# project = rf.workspace("kiyun").project("smoking-detection-yzewv")
# version = project.version(1)
# dataset = version.download("yolov8")
                