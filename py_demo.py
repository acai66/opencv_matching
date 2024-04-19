import sys
import ctypes
import numpy as np
import cv2

# 定义MatchResult结构体
class MatchResult(ctypes.Structure):
    _fields_ = [
        ('leftTopX', ctypes.c_double),
        ('leftTopY', ctypes.c_double),
        ('leftBottomX', ctypes.c_double),
        ('leftBottomY', ctypes.c_double),
        ('rightTopX', ctypes.c_double),
        ('rightTopY', ctypes.c_double),
        ('rightBottomX', ctypes.c_double),
        ('rightBottomY', ctypes.c_double),
        ('centerX', ctypes.c_double),
        ('centerY', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('score', ctypes.c_double)
    ]

# 定义Matcher类
class Matcher:
    def __init__(self, dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea):
        self.lib = ctypes.CDLL(dll_path)
        self.lib.matcher.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.matcher.restype = ctypes.c_void_p
        self.lib.setTemplate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.match.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(MatchResult), ctypes.c_int]
        
        if maxCount <= 0:
            raise ValueError("maxCount must be greater than 0")
        self.maxCount = maxCount
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        self.angle = angle
        self.minArea = minArea
        
        self.matcher = self.lib.matcher(maxCount, scoreThreshold, iouThreshold, angle, minArea)

        self.results = (MatchResult * self.maxCount)()
    
    def set_template(self, image):
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.setTemplate(self.matcher, data, width, height, channels)
    
    def match(self, image):
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid image shape")
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.match(self.matcher, data, width, height, channels, self.results, self.maxCount)

# 示例调用
maxCount = 1
scoreThreshold = 0.5
iouThreshold = 0.4
angle = 0
minArea = 256

dll_path = './templatematching_ctype.dll' # 模板匹配库路径

# 创建Matcher对象
matcher = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
if matcher is None:
    print("Create Matcher failed")
    sys.exit(111)

# 读取模板图像
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Read image failed")
    sys.exit(111)

# 设置模板
matcher.set_template(image)

# 读取摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Open camera failed")
    sys.exit(112)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Read camera failed")
        break

    # 匹配
    matches_count = matcher.match(frame)

    if matches_count < 0:
        print("Match failed!")
    
    assert matches_count <= matcher.maxCount, "matches_count must be less than or equal to maxCount"

    # 显示结果
    for i in range(min(matches_count, matcher.maxCount)):
        result = matcher.results[i]
        if result.score > 0:
            cv2.polylines(frame, [np.array([[result.leftTopX, result.leftTopY], [result.leftBottomX, result.leftBottomY], [result.rightBottomX, result.rightBottomY], [result.rightTopX, result.rightTopY]], np.int32)], True, (0, 255, 0), 1)
            cv2.putText(frame, str(result.score), (int(result.leftTopX), int(result.leftTopY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
