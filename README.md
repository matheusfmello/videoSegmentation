# videoSegmentation

## Project Overview

This project is a simple application of the CLIPSeg model availavle in Hugging Face. The goal is to apply image segmentation in order to extract elements
from a video. You can see more details on this article: https://medium.com/@matheusferreira_88940/video-segmentation-with-transformers-9d5b76d4bbc9


## Dependencies

You can create an environment for this project by running the following lines in your terminal.

```
conda create env --name <myEnv>
conda activate <myEnv>
pip install -r requirements.txt
```

## Usage

The segmentation is made by calling videoSegmentation.py and its arguments. You can see a generic usage example below

python videoSegmentation.py input.mp4 output.mp4 -p dog pink+balloon -b background.jpg -f 30

**-input.mp4:** video input path

**-output.mp4:** video output path

**-p dog pink+balloon:** elements to be extracted, '+' stands for spacebar in multiple words queries

**-b background.jpg:** background image path

-f 30: frames per second to be written
