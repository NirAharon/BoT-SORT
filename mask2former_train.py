from datasets import Dataset, DatasetDict, Image
import os
import cv2
import numpy as np
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoFeatureExtractor, SegformerForSemanticSegmentation,  Mask2FormerImageProcessor
image_paths_train = []
label_paths_train = []
image_paths_valid = []
label_paths_valid = []
checkpoint_dir = '/home/tony/Desktop/segmentation'
train_images_path = '/ssd_data/niranjan/coco_persons/train'
train_annotation_path = '/home/tony/Desktop/segmentation/train_annotation'

valid_images_path = '/ssd_data/niranjan/coco_persons/valid'
valid_annotation_path = '/home/tony/Desktop/segmentation/valid_annotation'

for img in os.listdir(train_images_path):
    file_name = os.path.join(train_images_path,img)
    if os.path.isfile(file_name) and any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        image_paths_train.append(file_name)
for img in os.listdir(train_annotation_path):
    file_name = os.path.join(train_annotation_path,img)
    label_paths_train.append(file_name)

for img in os.listdir(valid_images_path):
    file_name = os.path.join(valid_images_path,img)
    if os.path.isfile(file_name) and any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        image_paths_valid.append(file_name)
for img in os.listdir(valid_annotation_path):
    file_name = os.path.join(valid_annotation_path,img)
    label_paths_valid.append(file_name)


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

train_dataset = create_dataset(image_paths_train, label_paths_train)
validation_dataset = create_dataset(image_paths_valid, label_paths_valid)
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
  }
)
import json
# simple example
id2label = {0:'background',1: 'person'}
with open('id2label.json', 'w') as fp:
    json.dump(id2label, fp)



example = dataset['train'][0]
image = example['image']
# image_array = np.array(image)

# # # Display the image using cv2.imshow()
# cv2.imshow("name", image_array)
# cv2.waitKey(10000)
seg = np.array(example['label'])
# get green channel
instance_seg = seg[:, :, 1]
print(np.unique(instance_seg))

instance_seg = np.array(example["label"])[:,:,1] # green channel encodes instances
class_id_map = np.array(example["label"])[:,:,0] # red channel encodes semantic category
class_labels = np.unique(class_id_map)
print(class_labels)
# create mapping between instance IDs and semantic category IDs
inst2class = {}
for label in class_labels:
    instance_ids = np.unique(instance_seg[class_id_map == label])
    inst2class.update({i: label for i in instance_ids})
print(inst2class)



#processor = MaskFormerImageProcessor(ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)
processor = Mask2FormerImageProcessor(ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)



# import albumentations as A

# ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
# ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# transform = A.Compose([
#     A.Resize(width=512, height=512),
#     A.Normalize(mean=ADE_MEAN, std=ADE_STD),
# ])



# transformed = transform(image=np.array(image), mask=instance_seg)
# pixel_values = np.moveaxis(transformed["image"], -1, 0)
# instance_seg_transformed = transformed["mask"]
# print(pixel_values.shape)
# print(instance_seg_transformed.shape)
     


# print(np.unique(instance_seg_transformed))

# inputs = processor([pixel_values], [instance_seg_transformed], instance_id_to_semantic_id=inst2class, return_tensors="pt")

import torch

# for k,v in inputs.items():
#   if isinstance(v, torch.Tensor):
#     print(k,v.shape)
#   else:
#     print(k,[x.shape for x in v])

# assert not torch.allclose(inputs["mask_labels"][0][0], inputs["mask_labels"][0][1])

# print(inputs["class_labels"])
# print("Label:", id2label[inputs["class_labels"][0][1].item()])
# visual_mask = (inputs["mask_labels"][0][1].numpy() * 255).astype(np.uint8)
# cv2.imshow("name",visual_mask)
# cv2.waitKey(0)


import numpy as np
from torch.utils.data import Dataset

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, processor, transform=None):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB"))

        instance_seg = np.array(self.dataset[idx]["label"])[:,:,1]
        class_id_map = np.array(self.dataset[idx]["label"])[:,:,0]
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
          inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        return inputs



import albumentations as A

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# note that you can include more fancy data augmentation methods here
# train_transform = A.Compose([
#     A.Resize(width=512, height=512),
#     A.Normalize(mean=ADE_MEAN, std=ADE_STD),
# ])
transform_params = {"crop_size": 512, "blur_limit": 3, "sigma_limit": (0.2, 1.0), "fog_coef_upper": 0.5}
train_transform = A.OneOrOther(
        first=A.Compose(
            [
                A.LongestMaxSize(max_size=transform_params["crop_size"], always_apply=True),
                # A.RandomCrop(height=transform_params["crop_size"],
                #              width=transform_params["crop_size"],
                #              p=1),
                A.ToGray(p=0.5),
                A.HorizontalFlip(),
                A.OneOf([
                    A.ColorJitter(),
                    A.RGBShift(),
                ]),
                A.OneOf([
                    A.GaussianBlur(blur_limit=transform_params["blur_limit"],
                                   sigma_limit=transform_params["sigma_limit"]),
                    A.RandomFog(fog_coef_upper=transform_params["fog_coef_upper"]),
                    A.MedianBlur(blur_limit=transform_params["blur_limit"]),
                    A.Blur(blur_limit=transform_params["blur_limit"]),
                ]),
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomBrightnessContrast(),
                ]),
                A.SomeOf([
                    A.GaussNoise(),
                    A.RandomRain(blur_value=transform_params["blur_limit"], p=0.25),
                    A.Emboss(),
                    A.Sharpen(),
                    A.Equalize(),
                    A.HueSaturationValue(),
                ], n=3),
                #A.Lambda(lambda_function, p=1),
                A.Normalize(mean=ADE_MEAN, std=ADE_STD, p=1),
            ],
            p=1,
            #additional_targets=additional_targets
        ),
        second=A.Compose([
            A.LongestMaxSize(max_size=transform_params["crop_size"], always_apply=True),
            # A.RandomCrop(height=transform_params["crop_size"],
            #              width=transform_params["crop_size"],
            #              p=1),
            #A.Lambda(lambda_function, p=1),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD, p=1),
        ],
            p=1,
            #additional_targets=additional_targets
        ),
    )

train_dataset = ImageSegmentationDataset(dataset["train"], processor=processor, transform=train_transform)
val_dataset = ImageSegmentationDataset(dataset["validation"], processor=processor, transform=train_transform)



from torch.utils.data import DataLoader 

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)



from PIL import Image

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
#                                                           id2label=id2label,
#                                                           ignore_mismatched_sizes=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)


# train_batch = next(iter(train_dataloader))
# val_batch = next(iter(val_dataloader))


import torch
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)



best_val_loss = 100
for epoch in range(10):
    print("Epoch:", epoch)
    model.train()
    running_loss = 0.0
    num_samples = 0
    for idx, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        # if idx % 100 == 0:
        #     print("Loss:", running_loss/num_samples)

        # Optimization
        optimizer.step()
    print("Loss:", running_loss/num_samples)
    model.eval()
    val_loss = 0.0
    num_val_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Forward pass (no backpropagation)
            val_outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            val_loss += val_outputs.loss.item()
            num_val_samples += batch["pixel_values"].size(0)

    val_loss /= num_val_samples
    print("val_Loss:", val_loss)
    # Save the model's state if validation loss is the lowest
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        

