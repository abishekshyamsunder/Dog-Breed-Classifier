# Dog classification Application

### Installation Requirements

This project was implemented on a machine with GPU support. Thus the requiremets for such a machine has been provided.  
`pip install opencv-python==3.2.0.6`
`pip install h5py==2.6.0`
`pip install matplotlib==2.0.0`
`pip install numpy==1.12.0`
`pip install scipy==0.18.1`
`pip install tqdm==4.11.2`
`pip install keras==2.0.2`
`pip install scikit-learn==0.18.1`
`pip install pillow==4.0.0`
`pip install ipykernel==4.6.1`
`pip install tensorflow-gpu==1.0.0`

### Project Motivation
1. This project aims to identify the breed of a dog given an input image.  
2. It seeks to use OpenCV to identify if an input image has a human face in it as well as use a CNN to classify images of dogs into their respective breeds.  
  - This can either use a Network built from scratch or make use of Transfer learning.  

### File Descriptions
- dog_app.ipynb: This file is the work horse of the whole application, which contains code for implementing the classifier
- images folder: It contains the images used to finally test the working of our algorithm/model
- Data folder: This folder is empty in the directory. (please remove the please_delete.txt files). The data for this is large, and is thus hosted on [Google Drive](https://drive.google.com/file/d/1E8oEj0-TAJB6w0DHDLKZegJcQCUY48TG/view?usp=sharing). Download it from here and replace in the folder accordingly
- bottleneck_features folder: This folder is empty in the directory. (please remove the please_delete.txt files). The data for this is in .npz format and is large. This is hosted on [Google Drive as well](http:google.com). Download and put it in the folder accordingly
- extract_bottleneck_features.py: This file contains the code necessary to extract features for inputting to model

### Working interacting with the project
Before running the cells in the jupyter notebook, perform the following steps ('Replacing the paths to variables in the notebook')
- Find: `train_files, train_targets = load_dataset('../../../data/dog_images/train')`.  
  Replace: `train_files, train_targets = load_dataset('data/dog_images/train')`.  
- Find: `valid_files, valid_targets = load_dataset('../../../data/dog_images/valid')` 
  Replace: `valid_files, valid_targets = load_dataset('data/dog_images/valid')`   
- Find: `valid_files, valid_targets = load_dataset('../../../data/dog_images/test')` 
  Replace: `valid_files, valid_targets = load_dataset('data/dog_images/test')`
- Find: `bottleneck_features = np.load('../../../data/bottleneck_features/DogInceptionV3Data.npz')`.  
  Replace: `valid_files, valid_targets = load_dataset('bottleneck_features/DogInceptionV3Data.npz')`.  
 
Then run the command: `jupyter notebook` and open the .ipynb file to run the cells

### Project summary
An Algorithm was setup to identify if an image had a human or a dog in it. Then it would also identify the breed of the dog.  
When detecting humans, the face-detector misidentified dogs as human faces 11% of the time.  
When detecting dogs, the dog-detector misidentified humans as dogs faces 2% of the time.  
The CNN model obtained after transfer learning had an accuracy of 73%
