from ultralytics import YOLO
import cv2
import json

if __name__ == "__main__":
    detection_model = YOLO('yolo11m.pt')

    classification_model = YOLO("runs/classify/train37/weights/best.pt")

    cap = cv2.VideoCapture("private_data/carrot_example.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # only pass through objects in the person class
        results = detection_model(frame, classes=[0,1])

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                
                cropped_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                classification_results = classification_model.predict(cropped_img)
                probability = classification_results[0].probs.top1conf

                # only label if probability is higher than threshold
                if probability > 0.85:
                    index = classification_results[0].probs.top1
                    
                    label = f"Class: {classification_results[0].names[index]} ({classification_results[0].probs.top1conf:.2f})"
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        # break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

