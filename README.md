# Project: try to Fool AI

## Description 

Image a fully AI driven car insurance company which handles claims via a neural network based image classifier which detects if the uploaded image contain a damage.

Task: Please train a classifier based on a data set of damaged and non-damaged cars.

Damaged car images can be obtained here:

<https://www.kaggle.com/lplenka/coco-car-damage-detection-dataset> (Links to an external site.)

Non-damaged cars images shall be obtained from the internet to generate a balanced dataset


After a NN based classifier has been successfully trained with reasonable performance, using the ART library <https://github.com/Trusted-AI/adversarial-robustness-toolbox> (Links to an external site.) an attack needs to be used to add adversarial noise to the non-damaged cars to be misclassified as damaged cars.

Datasets in repo for the neural network
	- Undamaged car images
	- Damaged car images
