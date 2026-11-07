
## dataset
Before training you will need a dataset to actually train on.
The code expects a folder 'dataset' with subfolders like 'person1' wich need to contain .jpg files of that person
for example:
dataset
|
|-- 'person1'
|      |
|      |-- 'person1.jpg'
|      |-- 'person1_2.jpg'
|      |-- 'person1_3.jpg'
|
|-- 'person2'
       |
       |-- 'person2.jpg'
       |-- 'person2_2.jpg'
       |-- 'person2_3.jpg'

You get the idea.

## .env file
The .env file needs to contain the following information for usage of certain functions:

- RTSP_HOST=
- RTSP_PORT=
- RTSP_USERNAME=
- RTSP_PASSWORD=
- RTSP_STREAM=

- TESTMEDIA="PATH/TO/MEDIA/FILE"




## Dataset and Privacy

This project does not include or distribute any facial image datasets or trained models.

Users are responsible for providing their own datasets and ensuring they have the necessary rights and permissions to use them.

All dataset processing, model training and recognition are performed locally. No facial images, training data or recognition footage are uploaded to an external server by this software.

Users are responsible for complying with applicable privacy, data protection and copyright laws when using this software.

## Third-Party Components

This project uses the OpenCV Haar Cascade Classifier for face detection.

The following file is included in this repository:

- `haarcascade_frontalface_default.xml`

The Haar Cascade file is distributed as part of the OpenCV project and is subject to its applicable license terms.

OpenCV License: https://opencv.org/license/

All applicable copyright notices and license conditions are retained in the original file.

OpenCV is a third-party project and is not affiliated with or endorsing this software.

## Disclaimer

This software is provided "as is", without warranty of any kind, 
express or implied.

The developer is not responsible for any misuse of this software 
or for any damages resulting from its use.

Users are solely responsible for ensuring that their use of this 
software complies with all applicable laws and regulations.