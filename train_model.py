import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import os

# --- ตั้งค่า Path ของ Dataset เครื่องหมาย ---
DATASET_PATH = 'dataset' # โฟลเดอร์ Kaggle ของคุณ

# --- 1. โหลด MNIST (เอาเฉพาะตัวเลข 0-9) ---
print("1. กำลังโหลด MNIST (สุดยอดตัวเลข)...")
(x_digits, y_digits), _ = tf.keras.datasets.mnist.load_data()

# ปรับ Shape ให้เป็น (N, 28, 28, 1) และค่าสี 0-1
x_digits = x_digits.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# y_digits ไม่ต้องแก้ เพราะเป็น 0-9 อยู่แล้ว

print(f"   -> ได้ตัวเลขมา {len(x_digits)} รูป")

# --- 2. โหลดเครื่องหมายจาก Kaggle (เฉพาะ + - * /) ---
print("2. กำลังโหลดเครื่องหมายจาก Kaggle...")

# เอาแค่เครื่องหมาย (ไม่ต้องเอาเลข 0-9 ของ Kaggle)
symbol_map = {
    'add': 10, 
    'sub': 11, 
    'mul': 12, 
    'div': 13
}

x_symbols = []
y_symbols = []

for folder_name, label_id in symbol_map.items():
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ ไม่เจอโฟลเดอร์ {folder_name}")
        continue
        
    for f in os.listdir(folder_path):
        try:
            path = os.path.join(folder_path, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # ปรับขนาดเป็น 28x28 ให้เท่า MNIST
            img = cv2.resize(img, (28, 28))
            
            # กลับสี (ถ้าพื้นขาว ให้เป็นพื้นดำ)
            if np.mean(img) > 127: img = cv2.bitwise_not(img)
            
            x_symbols.append(img_to_array(img))
            y_symbols.append(label_id)
        except: pass

x_symbols = np.array(x_symbols).astype('float32') / 255.0
y_symbols = np.array(y_symbols)

print(f"   -> ได้เครื่องหมายมา {len(x_symbols)} รูป (ยังน้อยอยู่)")

# --- 3. ปั๊มยอดเครื่องหมาย (Balancing) ---
# MNIST มี 60,000 รูป แต่เครื่องหมายมีแค่ 2,000
# เราต้องก๊อปปี้เครื่องหมายซ้ำๆ ให้เยอะขึ้น (AI จะได้ไม่ลำเอียงตอบแต่ตัวเลข)
print("3. กำลังปั๊มยอดเครื่องหมายให้สูสีกัน...")

# ก๊อปปี้เครื่องหมายซ้ำ 15 รอบ (2,000 x 15 = 30,000 รูป)
x_symbols = np.tile(x_symbols, (15, 1, 1, 1))
y_symbols = np.tile(y_symbols, (15,))

print(f"   -> ตอนนี้มีเครื่องหมาย {len(x_symbols)} รูปแล้ว!")

# --- 4. รวมร่าง (Hybrid Dataset) ---
x_train = np.concatenate((x_digits, x_symbols), axis=0)
y_train = np.concatenate((y_digits, y_symbols), axis=0)

# สลับข้อมูลให้มั่ว (Shuffle)
idx = np.arange(len(x_train))
np.random.shuffle(idx)
x_train, y_train = x_train[idx], y_train[idx]

y_train = utils.to_categorical(y_train, 14)

# --- 5. สร้างและเทรนโมเดล ---
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(14, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 เริ่มเทรนโมเดลลูกผสม (Hybrid)...")
# ใช้ Data Augmentation ช่วยบิดรูปเครื่องหมายที่ก๊อปมา ให้หน้าตาไม่ซ้ำกัน
datagen = ImageDataGenerator(
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range=0.1
)

model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=15, verbose=1)

model.save('math_model_hybrid.h5')
print("🎉 สำเร็จ! ได้ไฟล์ 'math_model_hybrid.h5' ที่แม่นที่สุดในโลกมาแล้ว!")