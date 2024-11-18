import cv2
import numpy as np
import argparse
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

file = 'input'
os.walk(file)

for dirname, _, filenames in os.walk(file):
  for i in filenames:
    try:
        strr = ''
        nameFile = i
        image_pathhhhhh = dirname+'/'+i
        def extract_largest_image(image_path):
            # Đọc ảnh từ file
            image = cv2.imread(image_path)
            # Chuyển đổi ảnh sang ảnh xám
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Áp dụng threshold để tạo ảnh nhị phân
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            # Tìm contours trong ảnh nhị phân
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Tìm contour có diện tích lớn nhất
            largest_contour = max(contours, key=cv2.contourArea)
            # Lấy bounding box của contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Cắt ảnh từ bounding box
            largest_image = image[y:y+h, x:x+w]

            return largest_image

        # Đọc ảnh từ đường dẫn
        image = extract_largest_image(image_pathhhhhh)

        # Lấy chiều dài và chiều rộng của ảnh
        height, width, _ = image.shape

        # Tính toán vị trí để cắt ảnh
        start_width = int(width * 0.18)
        end_width = int(width * 0.43)
        start_height = int(height * 0.64)
        end_height = int(height * 0.7)

        # Cắt ảnh theo vị trí tính toán
        cropped_image = image[start_height:end_height, start_width:end_width, :]

        # Hàm để lưu mỗi contour thành một ảnh riêng biệt
        def save_contours_as_images(image, output_folder):
            result = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

            # Chuyển đổi ảnh sang ảnh đen trắng để dễ xử lý
            gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # Phát hiện số trong ảnh bằng phương pháp thresholding
            _, threshold_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY_INV)

            # Tìm contours trong ảnh
            contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sắp xếp contours theo diện tích giảm dần
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Giữ lại không quá num_contours contours
            contours = contours[:8]

            # Sắp xếp lại contours theo thứ tự trong ảnh ban đầu
            contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
            
            # Tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Lưu mỗi contour thành ảnh riêng biệt
            for i, contour in enumerate(contours):
                # Tạo một ảnh trắng để vẽ contour
                contour_image = np.zeros_like(threshold_image)
                cv2.drawContours(contour_image, [contour], -1, 255, 2)

                # Lấy bounding box của contour
                x, y, w, h = cv2.boundingRect(contour)

                # Cắt ảnh theo bounding box
                digit_image = image[y:y+h, x:x+w]
                digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
                _, digit_image = cv2.threshold(digit_image, 128, 255, cv2.THRESH_BINARY)

                cv2.imwrite(os.path.join(output_folder, f'digit_{i}.png'), digit_image)

        # Gọi hàm để lưu mỗi contour thành ảnh riêng biệt
        save_contours_as_images(cropped_image, 'output_folder')

        labels = [
            '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
            'H','I','J','K'
            ]

        def find_mssv(original_image):
            global strr
            # Define constants
            TARGET_WIDTH = 128
            TARGET_HEIGHT = 128
            MODEL_PATH = './trained_model'

            # Preprocessing the image
            image = cv2.resize(original_image, (TARGET_WIDTH, TARGET_HEIGHT))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Load the trained convolutional neural network
            model = load_model(MODEL_PATH, compile=False)

            # Classify the input image then find the index of the class with the *largest* probability
            prob = model.predict(image)[0]
            idx = np.argsort(prob)[-1]
            strr += labels[idx]
            
        for i in range(0, 8):
            original_image = cv2.imread(f'output_folder/digit_{i}.png')
            find_mssv(original_image)

        # Hàm để vẽ hình vuông quanh số
        def draw_square_around_number(image):
            result = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
            # Chuyển đổi ảnh sang ảnh đen trắng để dễ xử lý
            gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # Phát hiện số trong ảnh bằng phương pháp thresholding
            _, threshold_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY_INV)
            # Tìm contours trong ảnh
            contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Duyệt qua contours và vẽ hình vuông quanh số
            for contour in contours:
                # Lấy bounding box của contour
                x, y, w, h = cv2.boundingRect(contour)
                # Vẽ hình vuông quanh số
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        # Gọi hàm để vẽ hình vuông quanh số trên ảnh đã cắt
        draw_square_around_number(cropped_image)

        cv2.putText(image, strr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2 )

        def create_directory_if_not_exists(path):
            if not os.path.exists(path):
                os.makedirs(path)

        # Path to the output directory
        output_directory_path = "output"

        # Use the function to check and create the directory if needed
        create_directory_if_not_exists(output_directory_path)

        # Overlay the drawn squares onto the original image
        cv2.imwrite('output/' + nameFile, image)
    except:
      continue