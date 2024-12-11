YOLOv8 Animal Detection

This project uses YOLOv8, a state-of-the-art deep learning model for object detection, to detect animals in a video file. It draws bounding boxes around detected objects and labels them with the class name and confidence score. The annotated video is saved as an output file.
Requirements

    Python 3.x
    opencv-python
    ultralytics (for YOLOv8)

You can install the required dependencies using pip:

pip install opencv-python ultralytics

Files

    main.py: The main script for processing the input video and performing object detection.
    animals.mp4: The input video containing animals to detect.
    output/annotated_output.mp4: The output video with annotated bounding boxes and labels.
    yolov8n.pt: The pretrained YOLOv8 model weights (downloaded automatically via the ultralytics library).

How It Works

    Input Video: The script reads the video file animals.mp4 frame by frame.
    Object Detection: YOLOv8 is used to detect animals (or other objects) in each frame.
    Bounding Boxes and Labels: The script draws bounding boxes around detected objects and labels them with the object class name and confidence score.
    Output Video: The annotated video is saved as output/annotated_output.mp4.

How to Run

    Download the yolov8n.pt pretrained model by running the script (this will happen automatically when using the Ultralytics library).
    Place your input video file (e.g., animals.mp4) in the project directory.
    Run the script:

python main.py

    The annotated video will be saved in the output folder as annotated_output.mp4.

Customization

    Confidence Threshold: You can adjust the confidence threshold for object detection. Currently, the script uses 0.5 as the threshold (only objects with confidence above 50% are detected).
    Display Size: The script resizes the video to 640x480 for faster display. You can change the resolution by modifying the resize_width and resize_height variables in the code.
    Font Size and Box Thickness: The bounding boxes and labels are drawn with a font_scale of 1.5 and thickness of 4. You can modify these values to make the boxes and text bigger or smaller.

Example

After running the script, you will get an output like this:

    A video where animals (like cats) are detected.
    Bounding boxes and labels (e.g., "cat 0.92") appear over the detected objects.

Troubleshooting

    Model not found: If you encounter issues with downloading the model, ensure you have internet access. You can manually download the model from the Ultralytics YOLOv8 GitHub if necessary.
    Incorrect class names: YOLOv8 uses the COCO dataset class labels, so it might recognize a wide range of objects. If the script detects unwanted objects, you can filter the class IDs based on your needs.

License

This project is open-source and available under the MIT License.
