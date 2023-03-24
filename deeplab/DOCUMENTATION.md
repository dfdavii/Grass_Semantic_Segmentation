**GrassNet: Real-Time Grass Segmentation in Android using the Deeplavv3 Model Architecture**

**DOCUMENTATION**

To get started with the project, we need to install and download some libraries. These libraries are used extensively to have a positive experience running all the files in the project. Throughout the project, the UBUNTU 20.04 LTS is used, which is a linux-based operating system.

**List of Tools and Libraries to Install:**

1. MINICONDA3 to get conda, the package manager. Download the 64-bit version of miniconda3 for linux or the pkg version for mac computers.
1. VISUAL STUDIO CODE at <https://code.visualstudio.com/>

After downloading, make sure to get the python extension (IntelliSense) from microsoft

3. Labelme: annotation tool

15

**TABLE OF CONTENTS**

Image Scraping ……………………………………………………………………………. 3

Image Annotation: Labelme ……………………………………………………………… 3 Data Processing …………………………………………………………………………… 8

Folder Organization ……………………………………………………………………….. 9

Image Resizing …………………………………………………………………………….     10

Mask Preprocessing ………………………………………………………………………      10

Creating the tfrecord folder ……………………………………………………………….      11

Training and Evaluating the model ………………………………………………………      12

Visualizing the Results ……………………………………………………………………      14

Exporting as Frozen Graph ………………………………………………………………      14

Exporting as TFLITE File …………………………………………………………………      14 Android Application ……………………………………………………………………………… 15

Adding the model to Android ……………………………………………………………..      15

**Image Scraping**

There are plenty of options to generate grass images from google or other websites. The best option is from Bing.com. The google downloader lets you download a specific amount but there is an indefinite amount using the bing downloader. To install the library, on your terminal type the following, providing you have **python and the python manager pip** already installed in your computer:

pip install bing-image-downloader

Create a python file and copy the following in the file:

from bing\_image\_downloader import downloader![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.001.png)

downloader.download("cats on grass", limit=60, output\_dir='/home/dfdavii/Documents/image\_downloader',

adult\_filter\_off=True, force\_replace=False, timeout=60)

The first parameter ‘cats on grass’  ’is the name of the folder in the directory, output\_dir. The folder should contain images of cats on grass. You can be creative with the content of each image. The default limit is 100 images. All images obtained should be in the .jpg format. Using the python file, thousands of images containing any object on grass. The goal is to generate images containing grass and another object. The other objects would be the background class in addition to the grass class. All the images should be added to one folder to create a dataset of grass images. The images are of different sizes. Later each image is to be resized to 256x256 pixels.

**Image Annotation: Labelme**

We have a dataset of grass images saved in a folder. Image segmentation requires the ground truth of each individual image, which will become the label. To generate the ground truth or mask, the annotation tool **Labelme** is used. With miniconda3 already installed, open the terminal on the linux system, create a virtual environment name labelme:

conda create --name=labelme python=3.7

Then, activate the environment. After activating, python 3.7 should be installed in the environment.

conda activate labelme

Now the terminal should look as follows:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.002.png)

It is time to install labelme. Type the following code in the terminal:

pip install labelme

To open the library, type the following in the terminal

labelme

A new window should automatically pop up resembling the image below. For the sake of organization, make a new folder named ‘grass\_json’in the same directory as the downloaded images from Bing.com. We want to output the ground truths or masks in its own folder. Going back to the labelme window:

a)Choose ‘Open Dir’ to navigate to the folder where the grass

images reside. Open the folder

b)Now, all the jpg files should be available in the file list on

the labelme window. The first image should be displayed on the screen.

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.003.jpeg)

c)To navigate to the grass\_json folder created earlier. Go to

file and choose ‘Change Output Dir’.

d)Now on the labelme window, choose ‘Create Polygons’. You can

now draw polygons around the grass object found in the image. After drawing polygons around the grass, the last point of the polygons should be joined to the first point, then a small window pops up, prompting you to write the name of the label. Write ‘grass’, then OK. See the following image:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.004.jpeg)

e)The labelme software should create a label in the label list

tab. It should also pick a color for the grass label, which

will probably be red. This color represents the RGB of grass. f)Save the annotation by clicking ‘Save’. You can go to File and

click on ‘Save Automatically’. If the first image is 1.jpg, the

annotation is saved as 1.json.

g)Now, press ‘Next Image’ to draw out the next grass object.

After finishing with all the images in the folder, close the labelme window.

We are only drawing the grass objects in all the images, so there should only be one label. There are videos on Youtube that go through all the steps.

Now there should be the dataset folder with all the jpg images and output folder with all the json files.

You are provided with the python file ‘**labelme2voc.py**’. Move this python file to the same directory as the labelme output folder which contains all the json files. Create a text file named ‘classes.txt’, also in the same folder as the python file, the labelme output folder. Write ‘background’ on the first line, and ‘grass’ on the next line, as shown below:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.005.png)

On the terminal, navigate to the folder where all the aforementioned files reside, make sure you are still in the labelme virtual environment, type the following in the terminal:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.006.png)

The ‘**labelme2voc.py**’ is the python file that will generate the masks. The ‘zz’ folder is the output folder where all the json files reside. The ‘zz2’ folder is the name of the folder where all the new masks reside. After running the above command in the terminal, navigate to the current directory where you will find the folder ‘zz2’. The folders should be named differently based on the name you gave them. The ‘zz2’ folder should contain the following folders inside.

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.007.png)We need the JPEGImages and SegmentationClassPNG folders. Notice that all the JPEImages files are in jpg format and the SegmentationClassPNG files are in png format.

**DATA PROCESSING**

Up to this point, we have the images and their masks. We need to resize both sets of images and generate tfrecords for training. The tfrecord format is used to store binary records. It stores the sequences of byte-strings. If the dataset contains a lot of images, it expedites the training process.

With this documentation you are provided with a folder named ‘**models**’. The folder is a shorter version of Tensorflow’s deep learning models. It has been modified to train only on our custom grass dataset.

To work with this folder, go back to Visual Studio Code

a)Make sure Visual Studio Code (VSCODE) )can run python files b)Go to the terminal tab on VSCODE and choose New Terminal. c)Create a new virtual environment with conda on the terminal:

‘conda create tf’

d)Activate the new environment: ‘ conda activate tf’

e)Install the following on the new virtual environment:

pip install tensorflow-gpu==1.15.3

pip install tf\_slim

pip install numpy

pip install Pillow

pip install PrettyTable

pip install matplotlib

f)Download and unzip the model folder from the grassNet github

account

g)Go to the terminal on VSCODE, on the tf virtual environment and

export the model folder:

export PYTHONPATH=/home/dfdavii/Downloads/models/research:/home/dfdavii/Dow nloads/models/research/deeplab:/home/dfdavii/Downloads/models/resear ch/slim:{PYTHONPATH}![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.008.png)

h)Make sure to change this directory to the one where your models

folder is downloaded. In this example, the models folder resides in the Downloads directory.

**Folder Organization**

Create a new directory in the ./models/research/deeplab/datasets directory. This new directory is named ‘Custom\_dataset’. In this new folder, Make new files (-) and folders (+) as below:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.009.png)After organizing the **Custom\_Dataset** folder, move all the jpg images to the JPEGImages folder, and all the masks from the SegmentationClassPNG into the SegmentationClass folder.

The ImageSets/Segmentation folder consists of the train.tx, trainval.txt and val.txt. We need to add just the names of the images in those text files. The trainval.txt should contain the names of all the images. On your computer, go to the files folder, move all the images to trainval.txt. On VSCODE, open the trainval.txt, remove the extension of all the files by highlighting the .jpg extension. You should right-click on the highlighted part, there is a command that removes all the .jpg extensions from all the files. Save the trainval.txt. Do the same for the train.txt and the val.txt files. The train.txt contains the names of the training images. It should be less than the trainval.txt. If your trainval.txt contains 500 names, the train.txt should contain around the first 450 names. The val.txt contains the last 50 names.

**Resizing the images and their masks**

We need to resize all images and all masks with the same width and height. For this project, we choose the 256x256 image size.

a)Go to the dataset folder in ‘./models/research/deeplab’

directory, choose the resize\_images.py, change to the directory where your images are located, and run the file. Do the same for the mask images.

**Mask Preprocessing**

The files in our masks folder are in RGB format; the background class is encoded with the [0,0,0] and the grass class in [0, 128, 0]. We need to change the background class where the pixel values are 0, and the grass class with 1. In the Custom\_Dataset folder, run the file ‘convert\_rgb\_to\_index.py’. Change the ‘label\_dir’ and ‘new\_label\_dir’ according to the location of your masks. Running the file creates a SegmentationRaw folder in the Custom\_Dataset directory. Now, we are ready to create a tf\_record folder.

**Creating the tfrecord folder**

We have a ‘PACAL\_VOC’ dataset. In the datasets folder, the python file ‘data\_generator’ should look as follow:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.010.png)

For this example, we have 336 training images, based on the train.txt files, a total of 361 images, which should match the number of names in the trainval.txt files, and 25 test images, which match the number of names in the val.txt files. We are augmenting the training images to 1500. We have 2 classes, the background and the grass classes. Now, we are ready to generate the tfrecord folder. Navigate to the ‘build\_voc2012\_data.py’ in the dataset folder, change the directories accordingly.

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.011.jpeg)

Run the python file. That should create the tfrecord files. Navigate to the tfrecord folder to make sure the files are present.

**Training and Evaluating the model**

Included in the zip folder is a pretrained model. It is located in the datasets folder under the name ‘deeplabv3\_mnv2\_pascal\_train\_aug’. This folder contains 3 files. To begin training the deeplabv3 model, navigate to the ‘train.py’ file in the deeplab folder. Change the directories according to the location of your files. The ‘tf\_initial\_checkpoint’ and the ‘train\_logdir’ directories should be changed. The flags.DEFINE\_integer('training\_number\_of\_steps', 600,

'The number of steps used for training')![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.012.png)

is set to 600 steps. You can change the number of steps depending on your

dataset. Run the ‘train.py’ python file (It will not take too long to run, even with using CPU). After running, make sure several files are present in the checkpoint folder in the Custom\_Dataset folder. In the file

‘eval.py’, change both the ‘eval\_logdir’ and ‘checkpoint\_dir’ to the location of the evaluation folder and the checkpoint folder on your system.

flags.DEFINE\_string('eval\_logdir', ![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.013.png)'/home/dfdavii/Downloads/models/research/deeplab/datasets/Custom\_Dataset/E valuation', 'Where to write the event logs.')

flags.DEFINE\_string('checkpoint\_dir', '/home/dfdavii/Downloads/models/research/deeplab/datasets/Custom\_Dataset/c heckpoint', 'Directory of model checkpoints.')

’You can run the train.py and eval.py file simultaneously: Run the train.py file, wait 30 seconds or so, run the eval.py file. You can also run the train.py file first, then run the eval.py. The eval.py file should return the accuracy of the model after running.

During or after training, the losses can be visualized by using the tensorboard command inside the virtual environment. When the library tensorflow is installed, it also includes the visualization tool tensorboard to visualize the loss, the regularization loss and other metrics. This can be accomplished as the following examples shows:

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.014.png)

In this example, we are in the tf2 virtual environment, you need to change the location to where the model checkpoint resides. The command should display an http location to which you should navigate on your browser. It may look similar to ‘http://DaveFL:6006/’.

**Visualizing the Results**

The file ‘vis.py’, located in the datasets folder, is used to visualize the predicted masks. Change the vis\_logdir, checkpoint\_dir and the dataset\_dir according to your own locations. After running the file, the predicted masks should be in the vis\_logdir folder.

**Exporting as Frozen Graph**

We now have a tensorflow model in the checkpoint folder. We may need to use it in another API. Since we want to use it in an android application, we need to ‘freeze’ it. To do so, modify the ‘export\_model.py’ file located in the deeplab folder, then run the file.

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.015.png)

**Exporting as TFLITE File**

The end goal of the project is to make an Android application with the deeplab architecture. The model is now encoded in a frozen graph. It needs to be formatted as tensorflow lite file, a tflite. In the

deeplab folder, the ‘export\_tflite’ python file does just that.

![](Aspose.Words.d7e18762-9b7c-4533-87e6-e5c046a39103.016.png)In the above example, change the file paths according to your directories. The ‘MODEL\_VARIANT’ remains unchanged since our pretrained model is ‘mobilenet\_v2’. The tflite model should be saved in the ‘OUT\_PATH\_TFLITE’. Be sure it is in the correct directory.

**Android Application**

We need to create an android application to utilize the tflite model in order to have a real time segmentation of grass objects. To do so,

a)Go to [~~https://developer.android.com/studio~~](https://developer.android.com/studio)

b)Click on Download Android Studio. That will download the newest

version based on your operating system.

c)Go to your download folder and click on the file. Follow the

instructions on the screen to install it.

There should be an android application which accompanies this project. The file should be cloned or downloaded from github and unzipped to a known location. Go to Android Studio, and open the unzipped folder.

**Adding the model to Android**

When the android application is open in Android Studio: a)Right-click on the app folder, choose New, then click on

Directory

b)Type ‘assets’. There will be a list of directories. Choose

src\main\assets

c)Add the tflite model into the assets folder.

d)To run the application on your android phone, follow the instructions of this video: https://youtu.be/kpTPRsPOpRs
