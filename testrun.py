import os
import cv2
import torch
from mmseg.apis import inference_segmentor, init_segmentor

def process_image(image_path, save_dir):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found. Please check the path and try again.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = inference_segmentor(model, image_rgb)
    seg_img = model.show_result(image_rgb, result, palette=model.PALETTE)
    seg_img_bgr = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)

    output_image_path = os.path.join(save_dir, "segmented_image.jpg")
    cv2.imwrite(output_image_path, seg_img_bgr)
    print(f"Segmented image saved to {output_image_path}")
    cv2.imshow('Segmentation Result', seg_img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video not found. Please check the path and try again.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = inference_segmentor(model, frame_rgb)
        seg_img = model.show_result(frame_rgb, result, palette=model.PALETTE)
        seg_img_bgr = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Segmentation Result', seg_img_bgr)
        out.write(seg_img_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 初始化模型
config_file = 'configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py'
checkpoint_file = r"D:\load\HRDA\HRDA\work_dirs\gtaHR2csHR_hrda_246ef\latest.pth"
model = init_segmentor(config_file, checkpoint_file, device='cuda' if torch.cuda.is_available() else 'cpu')

# 定义保存图像的目录
save_dir = r"C:\Users\24459\Desktop\saved_images"
output_video_path = r"C:\Users\24459\Desktop\result1.mp4"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 用户传入一张驾驶场景图像
image_path = r"C:\Users\24459\Pictures\Screenshots\test.png"
video_path = r"C:\Users\24459\Desktop"

if image_path is not None:
    process_image(image_path, save_dir)
else:
    print("no image input")

if video_path is not None:
    process_video(video_path, output_video_path)
else:
    print("no video input")

