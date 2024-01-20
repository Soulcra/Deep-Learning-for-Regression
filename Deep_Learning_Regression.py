4. import os
5. import torch
6. import torch.nn as nn
7. import torch.optim as optim
8. import torchvision.transforms as transforms
9. import torchvision.models as models
10. from torch.utils.data import DataLoader, Dataset
11. import pandas as pd
12. from PIL import Image
13.
14. # Define the paths to your training and testing datasets
15. train_data_dir = 'extracted_dataset/DS_IDRID/Train'
16. test_data_dir = 'extracted_dataset/DS_IDRID/Test'
17.
18. # Function to clean the label by removing '-'
19. def clean_label(label):20. return label.replace('-', '')
21.
22. class CustomRegressionDataset(Dataset):
23. def __init__(self, data_dir, transform=None):
24. self.data_dir = data_dir
25. self.transform = transform
26. self.image_paths = [os.path.join(data_dir, filename)
for filename in os.listdir(data_dir) if filename.endswith('.jpg')]
27.
28. def __len__(self):
29. return len(self.image_paths)
30.
31. def __getitem__(self, index):
32. image_path = self.image_paths[index]
33. image = Image.open(image_path)
34.
35. # Extract the label from the filename and clean it
36. label = clean_label(image_path.split('_')[-
1].split('.')[0])
37. label = float(label)
38.
39. if self.transform:
40. image = self.transform(image)
41.
42. return image, label
43.
44. # Define data transformations and augmentations
45. transform = transforms.Compose([
46. transforms.Resize((224, 224)),
47. transforms.RandomHorizontalFlip(),
48. transforms.RandomVerticalFlip(),
49. transforms.RandomRotation(15),
50. transforms.ColorJitter(brightness=0.2, contrast=0.2,
saturation=0.2, hue=0.2),
51. transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
52. transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
53. transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
54. transforms.RandomGrayscale(p=0.2),
55. transforms.ToTensor(),
56. transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
57. ])
58.
59. # Create data loaders for training and testing60. train_dataset =
CustomRegressionDataset(data_dir=train_data_dir,
transform=transform)
61. test_dataset = CustomRegressionDataset(data_dir=test_data_dir,
transform=transform)
62.
63. # Define hyperparameters to experiment with
64. experiments = [
65. {'batch_size': 64, 'learning_rate': 0.001, 'num_epochs':
30},
66. ]
67.
68. results = [] # Store results for different experiments
69.
70. # Define the acceptable error margin (e.g., within 0.5 units
for regression)
71. threshold = 0.5
72.
73. for experiment in experiments:
74. batch_size = experiment['batch_size']
75. learning_rate = experiment['learning_rate']
76. num_epochs = experiment['num_epochs']
77.
78. train_loader = DataLoader(train_dataset,
batch_size=batch_size, shuffle=True)
79. test_loader = DataLoader(test_dataset,
batch_size=batch_size, shuffle=False)
80.
81. model = models.resnet18(pretrained=True)
82. num_ftrs = model.fc.in_features
83. model.fc = nn.Linear(num_ftrs, 1) # Regression model with
a single output neuron
84.
85. criterion = nn.MSELoss() # Mean Squared Error loss for
regression
86. optimizer = optim.SGD(model.parameters(),
lr=learning_rate, momentum=0.9)
87.
88. device = torch.device("cuda" if torch.cuda.is_available()
else "cpu")
89. model.to(device)
90.
91. for epoch in range(num_epochs):
92. model.train()
93. for inputs, labels in train_loader:94. inputs = inputs.to(device)
95. labels = labels.to(device).float() # Convert labels
to float
96. optimizer.zero_grad()
97. outputs = model(inputs)
98. loss = criterion(outputs, labels)
99. loss.backward()
100. optimizer.step()
101. within_threshold = 0 # Counter for predictions within the
acceptable error margin
102.
103. if len(test_loader.dataset) == 0:
104. print("Testing dataset is empty.")
105. else:
106. model.eval()
107.
108. with torch.no_grad():
109. for inputs, labels in test_loader:
110. inputs = inputs.to(device)
111. labels = labels.to(device).float() # Convert
labels to float
112. outputs = model(inputs)
113. errors = torch.abs(outputs - labels)
114. within_threshold += (errors <=
threshold).sum().item()
115.
116. accuracy = (within_threshold / len(test_loader.dataset)) *
100.0
117.
118. # Display results for this experiment
119. print("Experiment Results:")
120. print(f"Batch Size: {batch_size}, Learning Rate:
{learning_rate}, Num Epochs: {num_epochs}")
121. print(f"Accuracy within {threshold} units: {accuracy:.2f}%")