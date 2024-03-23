import cv2
import numpy as np

# 读取图像
image = cv2.imread('4/1.jpg')

# 创建原始掩码和临时掩码
mask = np.zeros(image.shape, dtype=np.uint8)
temp_mask = np.zeros(image.shape, dtype=np.uint8)

# 设置绘图参数
drawing = False
prev_x, prev_y = -1, -1
brush_size = 3  # 笔刷大小
mask_color = (255, 255, 255)  # 掩码颜色（白色）

# 回调函数用于绘制掩码和线条
def draw(event, x, y, flags, param):
    global drawing, prev_x, prev_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(temp_mask, (prev_x, prev_y), (x, y), 255, brush_size)
            prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# 创建窗口并绑定鼠标事件回调函数
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw)

while True:
    # 合并掩码和临时掩码以显示线条在掩码上层
    combined_mask = cv2.add(mask, temp_mask)
    cv2.imshow('Image', cv2.add(image, combined_mask))
    cv2.imwrite('temp_mask.jpg', temp_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按ESC键退出
        break
    elif key == ord('c'):  # 清空临时掩码
        temp_mask = np.zeros(image.shape, dtype=np.uint8)
    elif key == ord('s'):  # 保存临时掩码
        cv2.imwrite('temp_mask.jpg', temp_mask)

cv2.destroyAllWindows()
