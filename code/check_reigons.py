import cv2

frame = cv2.imread("data/sample_frame.png")

# Draw each ROI as a colored rectangle
cv2.rectangle(frame, (1460, 40),  (1650+250, 40+220),  (0, 0, 255),   2)  # kill feed — perfect
cv2.rectangle(frame, (660,  60), (660+650,  10+200), (0, 255, 255), 2)  # point % — taller
cv2.rectangle(frame, (1390, 895), (1390+500, 895+140), (255, 0, 255), 2)  # ult — perfect
cv2.rectangle(frame, (700,  1000), (500+700,  850+60),  (0, 255, 0),   2)

cv2.imwrite("data/sample_frame_rois.png", frame)
print("Saved sample_frame_rois.png")

