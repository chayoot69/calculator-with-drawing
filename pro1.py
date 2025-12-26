import cv2
import mediapipe as mp

import tensorflow as tf
import numpy as np

import math
import time 

# ---  โหลดโมเดล AI ---เครื่องคิดเลขด้วยการวาดภาพ
try:
    model = tf.keras.models.load_model('finaldog.h5') 
    print("โหลดโมเดลสำเร็จ พร้อมคำนวณ!")
except Exception as e:
    print(f"โหลดโมเดลไม่ได้: {e}")
    model = None

# 1. ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. ตั้งค่าหน้าจอ
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # หรือ 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # หรือ 1080
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0
drawColor = (255, 0, 255) 

tipIds = [4, 8, 12, 16, 20]
last_prediction_time = 0

# ตัวแปร Global สำหรับเก็บข้อความ
final_result_text = "" 
raw_text = ""  # <--- ประกาศไว้ตรงนี้กัน Error
sw=True
active_button_name = ""
active_button_timer = 0 

# แปลงรหัสตัวเลข
class_names = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '/'
}

def findAngle(img, p1, p2, p3, lmList, draw=True):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0: angle += 360
    if angle > 180: angle = 360 - angle
    return angle

def resize_icon(img, size=60):
    if img is None: return None
    # ใช้ INTER_AREA ภาพจะคมชัดกว่าเวลาย่อ
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def preprocess_for_ai(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    (h, w) = img.shape
    if h > w:
        factor = 20.0 / h
        h_new = 20
        w_new = int(w * factor)
    else:
        factor = 20.0 / w
        w_new = 20
        h_new = int(h * factor)
    img_resized = cv2.resize(img, (w_new, h_new))
    img_final = np.zeros((28, 28), dtype=np.uint8)
    pad_x = (28 - w_new) // 2
    pad_y = (28 - h_new) // 2
    img_final[pad_y:pad_y+h_new, pad_x:pad_x+w_new] = img_resized
    img_final = img_final.reshape(1, 28, 28, 1)
    img_final = img_final.astype('float32') / 255.0
    return img_final

def fingersUp(lmList, img, myHandType):
    fingers = []
    if myHandType == "Right":
        if lmList[4][1] > lmList[3][1]: fingers.append(1)
        else: fingers.append(0)
    else: 
        if lmList[4][1] < lmList[3][1]: fingers.append(1)
        else: fingers.append(0)
    for id in range(8, 21, 4):
        angle = findAngle(img, 0, id-2, id, lmList, draw=False)
        if angle > 150: fingers.append(1)
        else: fingers.append(0)
    return fingers

def get_segmented_rois(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            raw_boxes.append([x, y, w, h])
            
    raw_boxes = sorted(raw_boxes, key=lambda b: b[0])
    
    merged_boxes = []
    while len(raw_boxes) > 0:
        curr = raw_boxes.pop(0) 
        cx_curr = curr[0] + (curr[2] // 2)
        
        while len(raw_boxes) > 0:
            next_box = raw_boxes[0]
            cx_next = next_box[0] + (next_box[2] // 2)
            
            if abs(cx_curr - cx_next) < 20:
                x_min = min(curr[0], next_box[0])
                y_min = min(curr[1], next_box[1])
                x_max = max(curr[0]+curr[2], next_box[0]+next_box[2])
                y_max = max(curr[1]+curr[3], next_box[1]+next_box[3])
                curr = [x_min, y_min, x_max - x_min, y_max - y_min]
                raw_boxes.pop(0)
            else:
                break
        merged_boxes.append(curr)
    
    cropped_images = []
    for (x, y, w, h) in merged_boxes:
        padding = 20
        y_min, y_max = max(0, y-padding), min(gray.shape[0], y+h+padding)
        x_min, x_max = max(0, x-padding), min(gray.shape[1], x+w+padding)
        img_crop = gray[y_min:y_max, x_min:x_max]
        cropped_images.append(img_crop)
        
    return cropped_images
# --- ฟังก์ชันพิเศษสำหรับแปะ PNG พื้นใส (Overlay Transparent) ---
def overlay_transparent(background, overlay, x, y):
    # ถ้าไม่มีรูป หรือรูปไม่ใช่ PNG 4 ช่องสี (BGRA) ให้คืนค่าเดิม
    if overlay is None or overlay.shape[2] < 4: return background
    
    bg_h, bg_w, _ = background.shape
    fg_h, fg_w, _ = overlay.shape

    # ป้องกันกรณียวางรูปเลยขอบจอ
    if x < 0: x = 0
    if y < 0: y = 0
    if x + fg_w > bg_w: fg_w = bg_w - x
    if y + fg_h > bg_h: fg_h = bg_h - y
    # ตัดส่วนที่เกินออก
    overlay_cropped = overlay[0:fg_h, 0:fg_w]

    # แยกช่อง Alpha (ความโปร่งใส) ออกมา (ค่า 0-1)
    alpha_mask = overlay_cropped[:, :, 3] / 255.0
    # สร้าง Alpha แบบกลับด้านสำหรับพื้นหลัง
    alpha_inv = 1.0 - alpha_mask

    # พื้นที่บนวิดีโอที่เราจะเอารูปไปแปะ
    roi = background[y:y+fg_h, x:x+fg_w]

    # คำนวณการผสมสี: (สีไอคอน * ความทึบ) + (สีพื้นหลัง * ความใส)
    for c in range(0, 3): # ทำทีละช่องสี B, G, R
        roi[:, :, c] = (alpha_mask * overlay_cropped[:, :, c] +
                        alpha_inv * roi[:, :, c])
        
    # เอาผลลัพธ์ที่ผสมเสร็จแล้ว แปะกลับลงไปในภาพหลัก
    background[y:y+fg_h, x:x+fg_w] = roi
    
    return background

# เพิ่มตัวแปร active_btn="" ในวงเล็บ
def draw_header_ui(img, mode): 
    h, w, c = img.shape
    header_h = 100            
    btn_w = w // 3  # 3 ปุ่มบน (POINTER, DRAW, CLEAR)

    # Background ส่วนหัว
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    
    # --- ปุ่มด้านบน (Top Bar) ---
    buttons = [
        ("POINTER", "Selection", (255, 255, 255), icon_pointer),
        ("DRAW",    "Drawing",   (255, 255, 255),     icon_draw),
        ("CLEAR",   "Clear",     (255, 255, 255),     icon_clear)
    ]

    for i, (text, check_mode, active_color, icon_img) in enumerate(buttons):
        x_start = i * btn_w
        x_end = (i + 1) * btn_w
        center_x = x_start + (btn_w // 2)
        is_active = (check_mode in mode)
        
        if i > 0: cv2.line(img, (x_start, 15), (x_start, header_h-15), (50, 50, 50), 2)

        if is_active:
            cv2.rectangle(img, (x_start+5, 5), (x_end-5, header_h-5), (50, 50, 50), cv2.FILLED)
            cv2.rectangle(img, (x_start+5, 5), (x_end-5, header_h-5), active_color, 3)
            text_color = active_color 
        else:
            text_color = (100, 100, 100)
            
        if icon_img is not None:
            icon_w = icon_img.shape[1]
            img = overlay_transparent(img, icon_img, center_x - (icon_w // 2), 10)
            
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(img, text, (center_x - (text_size[0] // 2), header_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # ตัดส่วน Side Bar ทิ้งไปแล้ว เพราะเราวาดใน Loop แทน
    return img


# โหลดรูป PNG (สำคัญ: ต้องใช้ IMREAD_UNCHANGED เพื่อเก็บความใสไว้)
icon_pointer_raw = cv2.imread('icon_pointer.png', cv2.IMREAD_UNCHANGED)
icon_draw_raw    = cv2.imread('icon_draw.png',    cv2.IMREAD_UNCHANGED)
icon_calc_raw    = cv2.imread('icon_calc.png',    cv2.IMREAD_UNCHANGED)
icon_clear_raw   = cv2.imread('icon_clear.png',   cv2.IMREAD_UNCHANGED)

# ตรวจสอบว่าโหลดได้ไหม และย่อขนาด
icon_pointer = resize_icon(icon_pointer_raw)
icon_draw    = resize_icon(icon_draw_raw)
icon_calc    = resize_icon(icon_calc_raw)
icon_clear   = resize_icon(icon_clear_raw)
 


#ลูปหลัก
while sw:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    # รีเซ็ตโหมดเริ่มต้น
    current_mode = "Idle" 

    if result.multi_hand_landmarks:
        for hand_lms, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            #mp.solutions.drawing_utils.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)#กระดูก
            myHandType = hand_info.classification[0].label
            lmList = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                fingers = fingersUp(lmList, img, myHandType)

                # --- 1. ลบกระดาน (4 นิ้ว) ---
                if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                    current_mode = "Clear"
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                    final_result_text = ""
                    raw_text = ""
                    print("Clear Canvas")

            
                # --- . หยุดวาด ---
                elif fingers[1] and fingers[0]==0: 
                    xp, yp = 0, 0
                    current_mode = "Selection"
                    cv2.rectangle(img, (x1-25, y1-25), (x1+25, y1+25), (255, 0, 255), 3)

                    # เตรียมพิกัดเช็คปุ่ม
                    img_h, img_w, _ = img.shape
                    center_y = img_h // 2
                    btn_left_edge = img_w - 180
                    
                    # ถ้าจิ้มฝั่งขวา
                    if x1 > btn_left_edge:
                        # >>> ปุ่ม CALC <<<
                        if (center_y - 70) < y1 < (center_y - 10):
                            # สั่ง Trigger เอฟเฟกต์ (โดยไม่หยุดโปรแกรม)
                            active_button_name = "CALC"
                            active_button_timer = 10  # แสดงเอฟเฟกต์ 10 เฟรม (~0.3 วิ)

                            # คำนวณ (ทำแค่ครั้งเดียวในรอบนั้นๆ)
                            if time.time() - last_prediction_time > 1:
                                print("🟢 CALC PRESSED!")
                                rois = get_segmented_rois(imgCanvas)
                                if len(rois) > 0:
                                    segment_results = [] 
                                    for i, roi_img in enumerate(rois):
                                        if model is not None:
                                            roi_ai = preprocess_for_ai(roi_img)
                                            prediction = model.predict(roi_ai)
                                            result_index = np.argmax(prediction)
                                            symbol = class_names[result_index]
                                            segment_results.append(symbol)

                                    raw_text = "Raw: " + " ".join(segment_results)
                                    equation = "".join(segment_results)
                                    equation = equation.replace('x', '*') 
                                    
                                    try:
                                        ans = eval(equation) 
                                        if isinstance(ans, float) and ans.is_integer(): ans = int(ans)
                                        final_result_text = f"{equation} = {ans}"
                                        print("Solved: " + final_result_text)
                                    except Exception as e:
                                        final_result_text = "Error"
                                    
                                    last_prediction_time = time.time()

                        # >>> ปุ่ม EXIT <<<
                        elif (center_y + 10) < y1 < (center_y + 70):
                            # เอฟเฟกต์
                            active_button_name = "EXIT"
                            active_button_timer = 10
                            cv2.rectangle(img, (btn_left_edge, center_y + 10), (img_w, center_y + 70), (0, 0, 255), cv2.FILLED)
                            
                            # 2. เขียนตัวหนังสือทับ (เดี๋ยวปุ่มโล่ง)
                            cv2.putText(img, "EXIT", (btn_left_edge + 40, center_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                            # 3. สั่งอัปเดตหน้าจอทันที! (ถ้าไม่มีบรรทัดนี้ จอก็จะไม่เปลี่ยนสี)
                            # เช็คชื่อหน้าต่างใน cv2.imshow ด้านล่างสุดให้ตรงกันนะ (ปกติชื่อ "Image")
                            cv2.imshow("Air Calculator", img)
                            print("🔴 EXIT PRESSED!")
                                                        
                            # ส่งสัญญาณปิด (จะปิดจริงตอนท้าย Loop)
                            sw = False

                # --- 4. วาด ---
                elif fingers[1]:
                    if xp == 0 and yp == 0: xp, yp = x1, y1
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, 20)
                    xp, yp = x1, y1
                    current_mode = "Drawing"

    # --- ส่วนแสดงผล UI (สำคัญ! ต้องวางไว้ท้ายสุด) ---
    
            

    # 1. รวมภาพ Canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # พิกัด (1050, 150) คือตำแหน่ง x, y ลองแก้ตัวเลขดูถ้ายยังไม่ตรงใจ
    
    # 2. วาด Header ด้านบน (ฟังก์ชันเดิมของคุณ)
    img = draw_header_ui(img, current_mode)

    # -------------------------------------------------------------
    # จัดการสีปุ่ม Side Bar (CALC & EXIT)
    # -------------------------------------------------------------
    
    # --- ตั้งค่าสีเริ่มต้น (สถานะปกติ: พื้นดำ มีกรอบสี) ---
    calc_bg = (0, 0, 0)      # พื้นดำ
    calc_border = (0, 255, 0) # กรอบเขียว
    calc_text = (255, 255, 255) # ตัวหนังสือขาว

    exit_bg = (0, 0, 0)      # พื้นดำ
    exit_border = (0, 0, 255) # กรอบแดง
    exit_text = (255, 255, 255) # ตัวหนังสือขาว

    # --- เช็คว่ามีการกดปุ่มไหม? (ถ้ามี ให้เปลี่ยนสี) ---
    if active_button_timer > 0:
        active_button_timer -= 1  # นับถอยหลัง
        
        if active_button_name == "CALC":
            calc_bg = (0, 255, 0)     # เปลี่ยนพื้นเป็นเขียว
            calc_text = (0, 0, 0)     # เปลี่ยนตัวหนังสือเป็นดำ (ให้อ่านง่ายบนพื้นเขียว)
            
        elif active_button_name == "EXIT":
            exit_bg = (0, 0, 255)     # เปลี่ยนพื้นเป็นแดง
            exit_text = (255, 255, 255) 

    # -------------------------------------------------------------
    # ลงมือวาดจริง (วาดครั้งเดียวทับลงไปเลย)
    # -------------------------------------------------------------
    img_h, img_w, _ = img.shape
    center_y = img_h // 2
    btn_left_edge = img_w - 180
    cv2.putText(img, "or press 'q'", (btn_left_edge + 25, center_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "multiply 'x'", (btn_left_edge + 25, center_y - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # --- เริ่มส่วนวาดเครื่องหมายหาร ---
    div_cx = btn_left_edge + 132   # ขยับซ้ายขวาแก้เลขนี้
    div_cy = center_y - 135       # ขยับขึ้นลงแก้เลขนี้
    div_color = (0, 0, 255)       # สีแดง

    # วาดขีดกลาง (-)
    cv2.line(img, (div_cx - 10, div_cy), (div_cx + 10, div_cy), div_color, 2)
    # วาดจุดบน (.)
    cv2.circle(img, (div_cx, div_cy - 8), 2, div_color, -1)
    # วาดจุดล่าง (.)
    cv2.circle(img, (div_cx, div_cy + 8), 2, div_color, -1)
    cv2.putText(img, "divide", (btn_left_edge + 25, center_y - 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, div_color, 2)
    # --- จบส่วนวาดเครื่องหมายหาร ---
    # >>> วาดปุ่ม CALC <<<5
    # 1. พื้นหลัง (Filled)
    cv2.rectangle(img, (btn_left_edge, center_y - 70), (img_w, center_y - 10), calc_bg, cv2.FILLED)
    # 2. เส้นกรอบ (Border) - หนา 3px
    cv2.rectangle(img, (btn_left_edge, center_y - 70), (img_w, center_y - 10), calc_border, 1)
    # 3. ตัวหนังสือ
    cv2.putText(img, "CALC", (btn_left_edge + 25, center_y - 25), cv2.FONT_HERSHEY_PLAIN, 2, calc_text, 3)

    # >>> วาดปุ่ม EXIT <<<
    # 1. พื้นหลัง (Filled)
    cv2.rectangle(img, (btn_left_edge, center_y + 10), (img_w, center_y + 70), exit_bg, cv2.FILLED)
    # 2. เส้นกรอบ (Border) - หนา 3px
    cv2.rectangle(img, (btn_left_edge, center_y + 10), (img_w, center_y + 70), exit_border, 1)
    # 3. ตัวหนังสือ
    cv2.putText(img, "EXIT", (btn_left_edge + 25, center_y + 55), cv2.FONT_HERSHEY_PLAIN, 2, exit_text, 3)
    
    
    # ================ตรงกลางด้านล่าง =======================

    if final_result_text != "":
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0  # ขนาดตัวหนังสือใหญ่สะใจ
        thickness = 3
        color_text = (0, 255, 0) # สีเขียวนีออน (B, G, R)
        color_bg = (30, 30, 30)  # สีพื้นหลังเทาเข้ม
        
        
        (text_w, text_h), baseline = cv2.getTextSize(final_result_text, font, scale, thickness)
        
        # หาจุดกึ่งกลางจอ
        h, w, c = img.shape
        center_x = w // 2
        
        # คำนวณตำแหน่งวางข้อความ (ให้ข้อความอยู่กลางจอ แต่อยู่ด้านล่าง)
        text_x = center_x - (text_w // 2)
        text_y = h - 40  # ถอยจากขอบล่างขึ้นมา 150 px (เผื่อที่ให้ปุ่ม Exit หรือเผื่อตกขอบ)
        
        # 3. วาดกล่องพื้นหลัง (เผื่อขอบ Padding รอบตัวหนังสือ 20px)
        pad = 20
        # กล่องพื้นหลังทึบ
        cv2.rectangle(img, 
                      (text_x - pad, text_y - text_h - pad), 
                      (text_x + text_w + pad, text_y + pad), 
                      color_bg, cv2.FILLED)
        
        # วาดเส้นขอบกล่อง (สีเขียวเหมือนตัวหนังสือ) ให้ดูสวยงาม
        cv2.rectangle(img, 
                      (text_x - pad, text_y - text_h - pad), 
                      (text_x + text_w + pad, text_y + pad), 
                      color_text, 2)
        
        # 4. วาดตัวหนังสือ
        cv2.putText(img, final_result_text, (text_x, text_y), font, scale, color_text, thickness)
   
    if sw == False:
        cv2.waitKey(500) 
        break
    cv2.imshow("Air Calculator", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

    # --- . คำนวณ (3 นิ้ว) ---
'''elif fingers[1] and fingers[2] and fingers[3]:
                    xp, yp = 0, 0
                    current_mode = "Calculate"
                    if time.time() - last_prediction_time > 1:
                        rois = get_segmented_rois(imgCanvas)
                        
                        if len(rois) > 0:
                            segment_results = [] 
                            for i, roi_img in enumerate(rois):
                                

                                if model is not None:
                                    roi_ai = preprocess_for_ai(roi_img)
                                    prediction = model.predict(roi_ai)
                                    result_index = np.argmax(prediction)
                                    symbol = class_names[result_index]
                                    segment_results.append(symbol)

                            raw_text = "Raw: " + " ".join(segment_results)
                            print(raw_text)

                            equation = "".join(segment_results)
                            
                            # 2. (Optional) ตัวช่วยแก้บั๊กสำหรับสมการยาวๆ
                            # เปลี่ยน 'x' เป็น '*' เพื่อให้ eval เข้าใจ (ถ้าใน class_names คุณใช้ 'x')
                            equation = equation.replace('x', '*') 
                            
                            # 3. ส่งเข้า eval ให้คิดเลขทันที
                            try:
                                # eval จะคำนวณตามหลักคณิตศาสตร์ (คูณหารก่อนบวกลบ) ให้เอง
                                ans = eval(equation)
                                
                                # ปัดทศนิยมถ้าเป็นจำนวนเต็ม
                                if isinstance(ans, float) and ans.is_integer():
                                    ans = int(ans)
                                    
                                final_result_text = f"{equation} = {ans}"
                                print("Solved: " + final_result_text)
                                
                            except SyntaxError:
                                print("สมการผิดรูปแบบ")
                                final_result_text = "Syntax Error"
                            except ZeroDivisionError:
                                final_result_text = "Div by 0"
                            except Exception as e:
                                print(f"Error: {e}")
                                final_result_text = "Error"
                            
                            last_prediction_time = time.time()
                   ''' 