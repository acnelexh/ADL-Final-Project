# Final Project for Hemibrain Node Classification

This project is dedicated to the classification of nodes in the Hemibrain, a detailed map of the neural circuitry in the Drosophila (fruit fly) brain. 
The following instructions will guide you through the process of setting up and running the experiments included in this project.

## Table of Contents

1. [Installation](#installation)
2. [Training](#training)
3. [Experiments](#experiments)
4. [Contributors](#contributors)

## Installation

Before you can train the models, you need to clone the repository and install the required dependencies.

```
git clone https://github.com/acnelexh/ADL-Final-Project.git
cd ADL-Final-Project
pip install -r requirements.txt
```


## Training

To train the models, you can run the `train.py` script located in the root directory. You will need to specify the configuration file for the model as a command-line argument.

```
python train.sh --model SimpleGNN ....
```
Or alternatively, you can use the provided .sh file within bin directory
```
./bin/train.sh
```

## Experiments

All the configurations for the experiments are located in the `./bin/train.sh` file. To run an experiment, you can use the following command:

```
sh ./bin/train.sh
```

Please make sure that you have the necessary permissions to run the script. If you do not have the necessary permissions, you can add them using the following command:

```
chmod +x ./bin/train.sh
```

Feel free to modify the configurations in the `train.sh` file to suit your needs. The more detailed desciprtion of each parameters can be found in train.py file.

## Contributors

This project is managed by Zongyu Chen (zc2657@columbia.edu) and Benjamin Wu (bw2761@columbia.edu).
Please feel free to contribute to the project by creating a pull request.
