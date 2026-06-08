from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"E:\Waste_Detection\runs\detect\train-6\weights\best.pt")

# Image path
image_path = "2.png"

# Run prediction
results = model(image_path)

# Read original image
image = cv2.imread(image_path)

# Class mapping
BIODEGRADABLE_CLASSES = [
    "BIODEGRADABLE",
    "CARDBOARD",
    "PAPER"
]

NON_BIODEGRADABLE_CLASSES = [
    "GLASS",
    "METAL",
    "PLASTIC"
]

detected_categories = set()

# Process detections
for box in results[0].boxes:

    cls_id = int(box.cls[0])
    conf = float(box.conf[0])

    waste_type = model.names[cls_id].upper()

    # Bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Decide category and color
    if waste_type in BIODEGRADABLE_CLASSES:
        category = "BIODEGRADABLE"
        color = (0, 255, 0)      # Green
    else:
        category = "NON-BIODEGRADABLE"
        color = (0, 0, 255)      # Red

    detected_categories.add(category)

    # Draw colored bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    # Draw label
    label = f"{waste_type} {conf:.2f}"

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    print(f"Detected: {waste_type}")
    print(f"Confidence: {conf:.2f}")
    print(f"Category: {category}")

# Show overall result only once
if len(detected_categories) == 1:
    final_category = list(detected_categories)[0]
else:
    final_category = "MIXED WASTE"

cv2.putText(
    image,
    f"Category: {final_category}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 0),
    3
)

# Save output
output_path = "output.jpg"
cv2.imwrite(output_path, image)

# Display image
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\nFinal Category: {final_category}")
print(f"Output saved as: {output_path}")