import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def main(args):

    HEIGHT = 352
    WIDTH = 352
    video = cv2.VideoCapture(f'{args.input}')
    if (video.isOpened() == False):
        print("Could not read input video")
        exit()
    if args.background == 'b':
        background = np.zeros([HEIGHT, WIDTH, 3])
    elif args.background == 'w':
        background = np.ones([HEIGHT, WIDTH, 3])
        background = background * 255
    else:
        background = Image.open(f"{args.background}")
        background = background.resize([HEIGHT, WIDTH])
        background = np.array(background)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    prompts = args.prompts_list
    prompts = [string.replace('+', ' ') for string in prompts]
    newVideo = cv2.VideoWriter(f'{args.output}', fourcc, args.fps, [HEIGHT,WIDTH])

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    n_prompts = len(prompts)

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize([HEIGHT, WIDTH])
            inputs = processor(text=prompts, images=[frame] * n_prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            preds = outputs.logits
            if n_prompts > 1:
                preds = preds.unsqueeze(1)
                for i in range(n_prompts):
                    plt.imsave(f"tmp{i}.png", torch.sigmoid(preds[i][0]))
            else:
                plt.imsave(f"tmp0.png", torch.sigmoid(preds))
            gray_images = [cv2.cvtColor(cv2.imread(f"tmp{i}.png"), cv2.COLOR_BGR2GRAY) for i in range(n_prompts)]
            binary_images = [cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)[1] for gray_image in gray_images]
            bw_image = sum(binary_images)
            bw_image[np.where(bw_image > 1)] = 1
            old_frame = np.array(frame)
            n_channels = old_frame.shape[2]
            new_frame = np.zeros([HEIGHT, WIDTH, n_channels])
            for channel in range(n_channels):
                new_frame[:,:,channel] =  np.multiply(old_frame[:,:,channel], bw_image)
            new_frame[np.where(new_frame == 0)] = background[np.where(new_frame == 0)] # fills background
            new_frame = new_frame.astype(np.uint8)
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
            newVideo.write(new_frame)
        else:
            video.release()
            newVideo.release()
            for i in range(n_prompts):
                os.remove(f"tmp{i}.png")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Segments desired object from a video")
    parser.add_argument('input', help="path to input video")
    parser.add_argument('output', help="path to output video")
    parser.add_argument('-p', '--prompts-list', nargs='+', help="what to extract from the video")
    parser.add_argument('-b', '--background', default='b', help="black (b), white (w) or custom image background")
    parser.add_argument('-f', "--fps", type=int, default=20, help="fps on the output video")

    args = parser.parse_args()

    main(args)
