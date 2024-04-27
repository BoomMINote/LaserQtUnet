import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPixmap, QImage

unet = Unet()
def steger(img):
  # Load the image and convert to grayscale
  srcImage = img
  m, n = srcImage.shape[0], srcImage.shape[1]

  thresh = cv2.threshold(srcImage, 0, 255, cv2.THRESH_OTSU)[0] / 255 # 0.45
  srcImage = srcImage.astype(np.float64)

  # Apply Gaussian filter
  sigma = 3
  dstImage = gaussian_filter(srcImage, sigma) # 2048, 2448
  
  # Sobel derivatives
  dx = cv2.Sobel(dstImage, cv2.CV_64F, 1, 0, ksize=3)
  dy = cv2.Sobel(dstImage, cv2.CV_64F, 0, 1, ksize=3)

  # Second order derivatives
  dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
  dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
  dxy = cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize=3)
  
  hessian = np.zeros((2,2))
  points = np.zeros((m*n,2))
  
  for i in range(m):
    for j in range(n):
      if(srcImage[i,j]/255 > thresh):
        hessian[0,0] = dxx[i,j]
        hessian[0,1] = dxy[i,j]
        hessian[1,0] = dxy[i,j]
        hessian[1,1] = dyy[i,j]
        eigenval, eignevec = np.linalg.eig(hessian)
        if(eigenval[0] >= eigenval[1]):
          nx, ny = eignevec[:,0]
          fmax_dist = eigenval[0]
        else:
          nx, ny = eignevec[:,1]
          fmax_dist = eigenval[1]
        t = -(nx*dx[i,j] + ny*dy[i,j]) / (nx*nx*dxx[i,j]+2*nx*ny*dxy[i,j]+ny*ny*dyy[i,j])
        if abs(t*nx) <= 0.5 and abs(t*ny) <= 0.5:
          points[(i - 1) * m + j][0] = j + t * ny
          points[(i - 1) * m + j][1] = i + t * nx
  # Find indices where the first column of points is equal to 0
  index_1 = np.where(points[:, 0] == 0)[0]
  # Remove rows from points array where the first column is 0
  points = np.delete(points, index_1, axis=0)
  # Find indices where the second column of points is equal to 0
  index_2 = np.where(points[:, 1] == 0)[0]
  # Remove rows from points array where the second column is 0
  points = np.delete(points, index_2, axis=0)
  # Plot the points on the image
  plt.figure()
  plt.axis('off')
  plt.imshow(srcImage, cmap='gray')
  plt.plot(points[:, 0], points[:, 1], 'g.', markersize=0.03)
  plt.savefig('line-out.png', bbox_inches='tight', pad_inches=0,dpi=300)
  # plt.savefig('line-out.png')
  img = cv2.imread("line-out.png")
  return img

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UNet-CBAM-Steger激光条纹提取系统")
        self.setGeometry(300, 300, 700, 900) # start x, start y, width, height

        self.original_image_label = QLabel("原始图像")
        self.processed_image_label = QLabel("处理后图像")

        self.original_image_button = QPushButton("选择图像")
        self.original_image_button.clicked.connect(self.select_image)

        self.process_image_button = QPushButton("处理图像")
        self.process_image_button.clicked.connect(self.process_image)

        layout = QVBoxLayout()
        layout.addWidget(self.original_image_label)
        layout.addWidget(self.original_image_button)
        layout.addWidget(self.process_image_button)
        layout.addWidget(self.processed_image_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.original_image = None
        self.processed_image = None

    def select_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.img = Image.open(file_path)
            print(file_path)
            pixmap = QPixmap(file_path)
            self.original_image_label.setPixmap(pixmap.scaledToWidth(700))

    def process_image(self):
        # print(self.img.shape)
        if self.img is not None:
            count           = True
            name_classes    = ["background","laser"]
            img = self.img
            pred = unet.detect_image(img, count=count, name_classes=name_classes,mix_type=1)  
            pred = np.array(pred)
            print(pred.shape)
            processed_image = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            processed_image = steger(processed_image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            height, width, channel = processed_image.shape
            q_img = QImage(processed_image.data, width, height, processed_image.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.processed_image_label.setPixmap(pixmap.scaledToWidth(700))
        else:
            print("Please select an image first.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())






# if __name__ == "__main__":
#   image_path = "./0015.png"  # Replace with your image path
#   steger(image_path)


