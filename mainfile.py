# SETTING UP THE ENVIRONMENT

#from google.colab import drive
#drive.mount('/content/drive')
%cd /content/drive/MyDrive/noise2noise-master
#!git clone https://github.com/NVlabs/noise2noise.git
#%cd noise2noise
!pip install tensorflow==1.15
!pip install -r requirements.txt

#--------------------------------------------------------------------

# Defining a directory to store all the plots and figures
images_dir = '/content/drive/MyDrive/noise2noise-master/FIGURES'

#--------------------------------------------------------------------
# Path of BSDS dataset: /content/drive/MyDrive/BSDS300
# Path of pre-trained models: /content/drive/MyDrive/Colab Notebooks/noise2noise

#--------------------------------------------------------------------
# Training using N2C_Gaussian (PRE-TRAINED)

%cd /content/drive/MyDrive/noise2noise-master

!python config.py validate --dataset-dir=/content/drive/MyDrive/BSDS300/images/test --noise=gaussian --network-snapshot=/content/drive/MyDrive/noise2noise-pretrained/network_final-gaussian-n2c.pickle

#--------------------------------------------------------------------
# Training using N2N_Gaussian (PRE-TRAINED)

%cd /content/drive/MyDrive/noise2noise-master

!python config.py validate --dataset-dir=/content/drive/MyDrive/BSDS300/images/test --noise=gaussian --network-snapshot=/content/drive/MyDrive/noise2noise-pretrained/network_final-gaussian-n2n.pickle

#--------------------------------------------------------------------
# Converting images to records file

# Convert the images in BSD300 training set into a records file
!python config.py train --help 
!python dataset_tool_tf.py --input-dir /content/drive/MyDrive/BSDS300/images/train --out=/content/drive/MyDrive/BSDS300/bsd300_train.tfrecords

# Convert the images in BSD300 testing set into a records file
!python dataset_tool_tf.py --input-dir /content/drive/MyDrive/BSDS300/images/test --out=/content/drive/MyDrive/BSDS300/bsd300_test.tfrecords

#Training Images of BSDS300: bsd300_train.tfrecords
#Testing  Images of BSDS300: bsd300_test.tfrecords

#--------------------------------------------------------------------
# Training the network on BSDS300
# Training the Network on BSDS300 Training Data
!python config.py --desc='-test' train --train-tfrecords=/content/drive/MyDrive/BSDS300/bsd300_train.tfrecords --long-train=False --noise=gaussian

#--------------------------------------------------------------------
# Training the Network on BSDS300 Training Data [Noise2Noise]
# '--long-train=false'
# 'train_config.iteration_count = 10000'
# 'train_config.eval_interval = 1000'
# run time:=
# PSNR: for std.dev=25
!python config.py --desc='-test' train --train-tfrecords=/content/drive/MyDrive/BSDS300/bsd300_train.tfrecords --long-train=false --noise=gaussian

#--------------------------------------------------------------------
# Training the Network on BSDS300 Training Data [Noise2Clean]
# '--long-train=false'
# 'train_config.iteration_count = 10000'
# 'train_config.eval_interval = 1000'
# run time:=10mins15s
# PSNR:30.01 for std.dev=25
!python config.py --desc='-test' train --noise2noise=false --train-tfrecords=/content/drive/MyDrive/BSDS300/bsd300_train.tfrecords --long-train=false --noise=gaussian

#--------------------------------------------------------------------------------
# Comments:





















