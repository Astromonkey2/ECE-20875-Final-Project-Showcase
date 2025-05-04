#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images=images,labels=labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)
  idxs = [np.where(labels == n)[0][0] for n in number_list]
  images_nparray = images[idxs]
  labels_nparray = labels[idxs]
  return images_nparray, labels_nparray
  #return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
  n = len(images)
  fig, axes = plt.subplots(1,n, figsize =(2*n, 2))
  if n == 1:
    axes = [axes]
  for ax, img, lbl in zip(axes, images, labels):
    ax.imshow(img, cmap = 'gray')
    ax.set_title(str(lbl))
  plt.tight_layout()
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers)
#Part 2
print_numbers(class_number_images , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshaped)

def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  Accuracy = np.mean(results == actual_values)
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
all_flat = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
all_preds = model_1.predict(all_flat)
print("GaussianNP predictions for [0-9]:", all_preds)
print_numbers(allnumbers_images,all_preds)




#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
knn_results = model_2.predict(X_test_reshaped)
print("KNN overall accuracy:", OverallAccuracy(knn_results, y_test))
knn_preds = model_2.predict(all_flat)
print("KNN predictions for [0-9]:", knn_preds)
print_numbers(allnumbers_images, knn_preds)



#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
mlp_results = model_3.predict(X_test_reshaped)
print("MLP overall accuracy:", OverallAccuracy(mlp_results, y_test))
mlp_preds = model_3.predict(all_flat)
print("MLP predictions for [0-9]:", mlp_preds)
print_numbers(allnumbers_images, mlp_preds)



#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
X_train_p_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
for name,model in [("GaussianNB", GaussianNB()), ("knn", KNeighborsClassifier(n_neighbors = 10)), ("MLP",MLPClassifier(random_state =0, max_iter=300))]:
  model.fit(X_train_p_reshaped, y_train)
  preds = model.predict(X_test_reshaped)
  print(f"{name} accuracy on poisoned data:", OverallAccuracy(preds, y_test))

for name,model in [("GaussianNB", GaussianNB()), ("knn", KNeighborsClassifier(n_neighbors = 10)), ("MLP",MLPClassifier(random_state =0, max_iter=300))]:
  model.fit(X_train_p_reshaped, y_train)
  poisoned_preds = model.predict(all_flat)
  print(f"{name} poisoned predictions for [0-9]:", poisoned_preds)
  print_numbers(allnumbers_images, poisoned_preds)


#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64
X_train_p_flat = X_train_poison.reshape(X_train_poison.shape[0], -1)
kpca = KernelPCA(
    n_components= None,       # keep no nonlinear components
    kernel='rbf',          # The Radial Basis Function kernel is well-suited for image data
    gamma=1e-4,            # A smaller gamma makes the RBF kernel more flexible
    alpha=5e-3,            # Regularization parameter to prevent overfitting
    fit_inverse_transform=True, # Enables projecting data back to original space 
    random_state=42 )

X_train_denoised = kpca.inverse_transform( kpca.fit_transform(X_train_p_flat))


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
for name,model in [("GaussianNB", GaussianNB()), ("knn", KNeighborsClassifier(n_neighbors = 10)), ("MLP",MLPClassifier(random_state =0, max_iter=300))]:
  model.fit(X_train_denoised, y_train)
  preds = model.predict(X_test_reshaped)
  print(f"{name} accuracy on denoised data:", OverallAccuracy(preds, y_test))

for name,model in [("GaussianNB", GaussianNB()), ("knn", KNeighborsClassifier(n_neighbors = 10)), ("MLP",MLPClassifier(random_state =0, max_iter=300))]:
  model.fit(X_train_denoised, y_train)
  denoise_preds = model.predict(all_flat)
  print(f"{name} denoised predictions for [0-9]: ", denoise_preds)
  print_numbers(allnumbers_images, denoise_preds)

# This visualization snippet was added to clearly compare the original ("Clean"),
# the attacked ("Poisoned"), and the KernelPCA-denoised images side by side.
# It uses the official Matplotlib API
import matplotlib.pyplot as plt

# reshape denoised data back to 8×8 images
X_den_images = X_train_denoised.reshape(-1, 8, 8)

# number of samples to display
n_display = 5

# create a 3×N grid of subplots
fig, axes = plt.subplots(3, n_display, figsize=(2 * n_display, 6))

for i in range(n_display):
    # Clean image
    axes[0, i].imshow(X_train[i], cmap='gray')
    axes[0, i].set_title('Clean')
    axes[0, i].axis('off')
    
    # Poisoned image
    axes[1, i].imshow(X_train_poison[i], cmap='gray')
    axes[1, i].set_title('Poisoned')
    axes[1, i].axis('off')
    
    # Denoised image
    axes[2, i].imshow(X_den_images[i], cmap='gray')
    axes[2, i].set_title('Denoised')
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()

