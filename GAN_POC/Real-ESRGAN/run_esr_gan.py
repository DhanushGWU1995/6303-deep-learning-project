import os
import cv2
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ================= CONFIG =================
INPUT_ROOT = "data/raw"        # change to your dataset root
OUTPUT_ROOT = "data/output/raw"      # output folder
MODEL_PATH = "weights/RealESRGAN_x4plus.pth"

UPSCALE = 4           # 4x upscaling
USE_HALF = True       # True = faster on GPU
TILE = 0              # set 128 if memory issues
# ==========================================


def load_model():
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=UPSCALE
    )

    upsampler = RealESRGANer(
        scale=UPSCALE,
        model_path=MODEL_PATH,
        model=model,
        tile=TILE,
        tile_pad=10,
        pre_pad=0,
        half=USE_HALF
    )

    return upsampler


def process_image(upsampler, input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f" Failed to read: {input_path}")
        return

    output, _ = upsampler.enhance(img, outscale=UPSCALE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output)


def process_folder(upsampler):
    total_images = 0

    for root, _, files in os.walk(INPUT_ROOT):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            continue

        rel_path = os.path.relpath(root, INPUT_ROOT)
        output_dir = os.path.join(OUTPUT_ROOT, rel_path)

        for file in tqdm(image_files, desc=f"Processing {rel_path}"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)

            process_image(upsampler, input_path, output_path)
            total_images += 1

    print(f"\n Done! Processed {total_images} images.")


if __name__ == "__main__":
    print(" Loading model...")
    upsampler = load_model()

    print(" Processing images...")
    process_folder(upsampler)