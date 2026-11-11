# Processed datasets

Download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1kEi1WrSiOtaD2fb7fjfRnSnbMSuIhqNC?usp=drive_link).
Do not commit dataset images to this repository.

Use CelebA/{train,test}/{HR,LR/x2,LR/x4,LR/x8} and the same structure for Helen.
HR images are 128×128 RGB. Pair filenames must match exactly. The supplied
training loader selects 96×96 HR crops at ×2 and 128×128 crops at ×4/×8.

The complete historical raw-to-HR alignment code and original filename mapping
are not included. A new crop or alignment of raw CelebA is a different dataset.
Use the author's processed split to compare with the original experiments.
