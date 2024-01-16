# Chameleon - AI Drawing Tool
Welcome to Chameleon, an AI drawing tool based on TensorRT and SDXL, featuring style fusion and fast inference. The UI component of this project is built on the Carefree-drawborad framework, and we would like to express our gratitude for providing a simple and intuitive interaction experience.


## Features

1. Style Fusion: Chameleon can naturally blend objects with different styles, creating unique artistic compositions.
2. Fast Inference: Accelerated with TensorRT, Chameleon provides faster inference, allowing you to instantly preview generated effects.
3. Additional Features:
   - Text-to-Image: Transform text into captivating images.
   - Image-to-Image: Generate images based on existing visuals.
   - Inpainting: Restore or complete images seamlessly.
   - ControlNet: Advanced control over generated content.
   - HighresFix: Enhance image resolution for finer details.
   - DemoFusion: Generate image at 4×, 16×, and even higher resolutions without any fine-tuning or prohibitive memory demands.
   - Segment-Anything: Segment and manipulate various elements in images.

## Getting Start

### 1. Prepare Environment
Ensure you have the required dependencies installed. You can do this by running:


### 2. Export TensorRT Engine
Export the TensorRT engine to optimize the inference process. Run the following command:


### 3. Start Backend API
Launch the backend API to handle the core functionality. Execute the following command:


### 4. Start Frontend UI
Initiate the frontend UI for a user-friendly experience. Run the following command:


Open http://localhost:8000 in your browser to access the Chameleon application.


## Contribution
If you have any suggestions or discover any bugs, feel free to raise them in the Issues section.

## License
This project is licensed under the MIT License.





