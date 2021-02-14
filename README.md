# Setup
The code requires the installation of tesseract-ocr in your environment. <br>
First and foremost run the command 
```
pip install -r requirements.txt
```
After this install tesseract-ocr at your pc by following the steps mentioned in [this](https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/) article. A one line installation for ubuntu users is 
```
sudo apt-get install tesseract-ocr
```
which can then be validated by running the command `tesseract -v`.

# Running the code
```
usage: openv_text_detector.py [-h] -east EAST [-v VIDEO] [-c MIN_CONFIDENCE] [-w WIDTH] [-e HEIGHT] [-o OUTPUT] [-t TESSERACT]

optional arguments:
  -h, --help            show this help message and exit
  -east EAST, --east EAST
                        path to input EAST text detector
  -v VIDEO, --video VIDEO
                        path to optional input video file
  -c MIN_CONFIDENCE, --min-confidence MIN_CONFIDENCE
                        minimum probability required to inspect a region
  -w WIDTH, --width WIDTH
                        resized image width (should be multiple of 32)
  -e HEIGHT, --height HEIGHT
                        resized image height (should be multiple of 32)
  -o OUTPUT, --output OUTPUT
                        Path to the output file
  -t TESSERACT, --tesseract TESSERACT
                        The absolute location of the file tesseract.exe, wherever you have installed it
```
You have to neccessarily provide three arguments
1. --east is the path of the file `frozen_east_text_detection.pb`
2. --video is the path to the video file (if not provided it will use the webcam input)
3. --tesseract is the absolute location of the file tesseract.exe, wherever you have installed it

# Data
Currenly I have provided 2 video files 
1. `video1.mp4` which is a lecture in which the entire screen consists of text. The output of my code for this video can be seen in `sample_output/op1.txt`
2. `video2.mp4` which contains the lyrics of a song and hence only a localized part of the screen consists of text. The output of my code for this video can be seen in `sample_output/op2.txt`. Reasons for using this video is that the lyrics remain centered in the video and the background is plain across the text, otherwise I will have to use a different preprocessing for this.

# Currently implemented algorithm

1. I first use opencv's EAST text detector for detecting where the text is present (this is done once in 60 frames (assuming that the video is 60fps and that the video content does not change within a second)). 
2. This gives the location of all the text present in the image as bounding boxes.
3. Then I accumulate all the rectangles into one large rectangle (because of assumption mentioned below) and extract that part of the image. 
4. Then I use pytesseract to get what is present in this extracted image after performing some basic image preprocessing.

Although the output files currently carry repeated text across frames, this can be removed by doing a hamming distance check and removing the next frames text if the hamming score is too small. (currently I have removed the next frame's text if they are exactly similar to previous frame's text, but tesseract's output does not remain the same everytime due to which repeated text is occuring in the output)

# Assumptions taken

The text is localized to one part of the video (it won't be the case that one part of the text is on the top left corner and other is at the bottom right). Reason :-  My code will work for this case too, but the extracted text will most probably contain the whole image and it will be difficult for tesseract to correctly identify the text. If I do not coalesce the small bounding boxes on the texts to one big bounding box, almost each word will come  as a separate box which will make joining them together to form a meaningful sentence much more difficult.


# Improvements that can be done

1. I take the assumption that the text should be localized. This can be easily removed by not coalescing all the small rectangles but this gives rise to another problem which is the correct order of the text. But what can be done is instead to coalesce two rectangles if they are close enough otherwise keep them separate. This can be done by placing a condition that you merge two rectangles only if the distance between their center is less than the max of the sum of their lengths or breadths.
2. We can take the help of NLP (language models) to better predict the text. But this change will have to be made in OCR reader, which in our case is the tesseract-ocr, or we can correct the output of the tesseract using NLP.
