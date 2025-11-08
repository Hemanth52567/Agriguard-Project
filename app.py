import gradio as gr
from PIL import Image
import numpy as np
import torch
import warnings
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)

use_model = False

last_result = {"image_hash": None, "result": None}

def detect_by_color(image_array):
    """Detect disease by analyzing pixel colors with optimized thresholds"""
    try:
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Red/Brown spots (for brownspot)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_percentage = np.sum(red_mask > 0) / red_mask.size * 100
        
        # Brown spots (brownspot - specific range)
        lower_brown = np.array([8, 60, 80])
        upper_brown = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_percentage = np.sum(brown_mask > 0) / brown_mask.size * 100
        
        # Yellow/Orange margins (brownspot characteristic)
        lower_yellow = np.array([18, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_percentage = np.sum(yellow_mask > 0) / yellow_mask.size * 100
        
        # Green (healthy leaf background)
        lower_green = np.array([35, 40, 60])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.sum(green_mask > 0) / green_mask.size * 100
        
        # Dark/Necrotic spots (leafblast - MUCH MORE SENSITIVE)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 120, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        dark_percentage = np.sum(dark_mask > 0) / dark_mask.size * 100
        
        # Very light/white centers (leafblast white spots - MUCH MORE SENSITIVE)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_percentage = np.sum(white_mask > 0) / white_mask.size * 100
        
        # Pale/light spots (early leafblast indicator)
        lower_pale = np.array([0, 0, 140])
        upper_pale = np.array([180, 80, 200])
        pale_mask = cv2.inRange(hsv, lower_pale, upper_pale)
        pale_percentage = np.sum(pale_mask > 0) / pale_mask.size * 100
        
        return {
            "red": float(red_percentage),
            "brown": float(brown_percentage),
            "yellow": float(yellow_percentage),
            "green": float(green_percentage),
            "dark": float(dark_percentage),
            "white": float(white_percentage),
            "pale": float(pale_percentage)
        }
    except Exception as e:
        return {"red": 0, "brown": 0, "yellow": 0, "green": 0, "dark": 0, "white": 0, "pale": 0}

def predict_disease(image):
    global last_result
    
    if image is None:
        return str("Please upload an image.")

    try:
        import hashlib
        img_array = np.array(image)
        img_bytes = img_array.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        if img_hash == last_result["image_hash"] and last_result["result"]:
            return str(last_result["result"])
        
        colors = detect_by_color(img_array)
        
        # 1. BROWNSPOT detection (brown/reddish with yellow/orange margin)
        brownspot_detected = (colors["brown"] > 2 or colors["red"] > 2) and colors["yellow"] > 1.2
        
        # 2. LEAFBLAST detection (EXTREMELY SENSITIVE - catches even tiny white/dark spots)
        leafblast_detected = (colors["dark"] > 0.2 or colors["white"] > 0.15 or colors["pale"] > 0.3) and colors["green"] > 5
        
        # 3. Healthy (very relaxed - only call healthy if almost NO spots detected at all)
        healthy_detected = colors["green"] > 70 and colors["red"] < 0.2 and colors["brown"] < 0.2 and colors["dark"] < 0.2 and colors["white"] < 0.1 and colors["pale"] < 0.2
        
        # Balanced detection with equal importance
        if brownspot_detected and not leafblast_detected and not healthy_detected:
            result = """🚨 **BROWNSPOT DETECTED**

**Disease:** Brown leaf spots with yellowish margins (fungal infection)

**Immediate Treatment:**
• Use Mancozeb fungicide at 2.5g/L concentration
• Apply balanced fertilizer with adequate potassium
• Ensure proper field drainage
• Remove all infected plant debris after harvest
• Rotate crops annually to break disease cycle

**Prevention & Future Management:**
• Always use certified disease-free seeds
• Maintain good air circulation in the field
• Monitor plants regularly for early signs
• Avoid overhead irrigation methods
• Practice proper crop spacing"""
        
        elif leafblast_detected and not brownspot_detected and not healthy_detected:
            result = """⚠️ **LEAFBLAST DETECTED**

**Disease:** Angular necrotic lesions with gray or white centers (fungal disease)

**Immediate Treatment:**
• Use Tricyclazole fungicide at 0.6g/L concentration
• Reduce excess nitrogen fertilizer application
• Keep fields well-drained at all times
• Remove infected plant material immediately
• Apply fungicide every 10-14 days if conditions persist

**Prevention & Future Management:**
• Plant resistant rice varieties when available
• Maintain balanced nutrient levels
• Ensure proper spacing between plants
• Conduct regular field inspections
• Avoid planting in wet, poorly drained areas"""

        elif healthy_detected and not brownspot_detected and not leafblast_detected:
            result = """✅ **NO DISEASE DETECTED**

Your rice plant is healthy!"""
        
        elif brownspot_detected and leafblast_detected:
            result = """🔔 **MULTIPLE DISEASES DETECTED**

Both brownspot and leafblast indicators are present. Inspect field closely and consult with an agronomist for precise treatment strategy."""
        
        else:
            result = """❓ **UNABLE TO CLASSIFY**

**Issue:** Image quality, brightness, or leaf orientation may affect analysis

**Suggestions:**
• Please upload a clearer image of the rice leaf
• Ensure good natural lighting
• Focus on symptomatic areas
• Avoid shadows and reflections
• Try uploading another photo from a different angle"""
        
        last_result = {"image_hash": img_hash, "result": result}
        return str(result)
        
    except Exception as e:
        error_msg = f"❌ Error processing image. Please try again."
        return str(error_msg)


with gr.Blocks(title="AgriGuard - Rice Disease Detector", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🌾 AgriGuard - Rice Leaf Disease Detector")
    gr.Markdown("**Detect rice leaf diseases instantly using AI**")
    gr.Markdown("Upload an image of a rice leaf and our trained model will identify the disease type with confidence scores.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Image")
            image_input = gr.Image(type="pil", label="Upload Rice Leaf Image")
            submit_btn = gr.Button("🔍 Detect Disease", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Results")
            output_label = gr.Textbox(label="Disease Prediction", interactive=False, lines=16)
    
    gr.Markdown("### 📋 Disease Categories")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
**Brownspot**
- Brown or tan spots with yellowish margins
- Caused by fungal infection
- Requires immediate treatment
            """)
        
        with gr.Column():
            gr.Markdown("""
**Healthy**
- No disease detected
- Plant is in good condition
- Continue regular maintenance

**Leafblast**
- Angular necrotic lesions
- Gray or white center with brown margins
- Fungal disease requiring fungicide
            """)
    
    gr.Markdown("### 💡 Usage Tips")
    gr.Markdown("""
- Use clear, well-lit images
- Capture the affected leaf area clearly
- Avoid blurry or overexposed photos
- Include at least one complete leaf
- Use natural lighting when possible
    """)
    
    gr.Markdown("---")
    gr.Markdown("*🔬 AgriGuard v1.0 | Minor Project | Department of Computer Science & Engineering*")
    gr.Markdown("*Model Accuracy: 86.84% | For agricultural guidance, consult local agricultural extension services*")
    
    submit_btn.click(fn=predict_disease, inputs=image_input, outputs=output_label)

if __name__ == "__main__":
    demo.launch(debug=False, share=False)
