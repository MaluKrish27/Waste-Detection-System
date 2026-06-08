from django.shortcuts import render
from ultralytics import YOLO
from django.conf import settings
import cv2
import os

# Load YOLO model
model = YOLO(
    r"E:\Waste_Detection\runs\detect\train-6\weights\best.pt"
)

# Category mapping
BIO_MAP = {
    "PAPER": "BIODEGRADABLE",
    "CARDBOARD": "BIODEGRADABLE",
    "BIODEGRADABLE": "BIODEGRADABLE",

    "PLASTIC": "NON-BIODEGRADABLE",
    "GLASS": "NON-BIODEGRADABLE",
    "METAL": "NON-BIODEGRADABLE",
    "NON_BIODEGRADABLE": "NON-BIODEGRADABLE"
}


def predict_waste(request):

    result = None
    biodegradable_count = 0
    non_biodegradable_count = 0

    if request.method == "POST":

        image_file = request.FILES.get("image")

        if image_file:

            os.makedirs(
                settings.MEDIA_ROOT,
                exist_ok=True
            )

            upload_path = os.path.join(
                settings.MEDIA_ROOT,
                image_file.name
            )

            with open(upload_path, "wb+") as destination:

                for chunk in image_file.chunks():
                    destination.write(chunk)

            results = model.predict(
                upload_path,
                conf=0.30,
                iou=0.45
            )

            image = cv2.imread(upload_path)

            # Keep only highest confidence detection per class
            best_detections = {}

            for box in results[0].boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                waste_type = (
                    model.names[cls_id]
                    .upper()
                    .strip()
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                if (
                    waste_type not in best_detections
                    or confidence >
                    best_detections[waste_type]["confidence"]
                ):

                    best_detections[waste_type] = {
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2)
                    }

            detections = []

            for waste_type, data in best_detections.items():

                confidence = data["confidence"]

                x1, y1, x2, y2 = data["box"]

                category = BIO_MAP.get(
                    waste_type,
                    "NON-BIODEGRADABLE"
                )

                if category == "BIODEGRADABLE":

                    color = (0, 255, 0)
                    biodegradable_count += 1

                else:

                    color = (0, 0, 255)
                    non_biodegradable_count += 1

                # Draw box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                label = (
                    f"{waste_type} "
                    f"{confidence * 100:.1f}%"
                )

                (w, h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                cv2.rectangle(
                    image,
                    (x1, max(y1 - 35, 0)),
                    (x1 + w + 10, y1),
                    color,
                    -1
                )

                cv2.putText(
                    image,
                    label,
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                detections.append({
                    "waste_type": waste_type,
                    "confidence": round(
                        confidence * 100,
                        2
                    ),
                    "category": category
                })

            output_name = (
                "output_" +
                image_file.name
            )

            output_path = os.path.join(
                settings.MEDIA_ROOT,
                output_name
            )

            cv2.imwrite(
                output_path,
                image
            )

            result = {
                "detections": detections,
                "image_url":
                settings.MEDIA_URL +
                output_name
            }

    return render(
        request,
        "predict.html",
        {
            "result": result,
            "biodegradable_count": biodegradable_count,
            "non_biodegradable_count": non_biodegradable_count
        }
    )