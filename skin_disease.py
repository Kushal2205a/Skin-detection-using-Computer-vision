from pathlib import Path 

root_file = Path("C:/Users/Kushal/Documents/Skin Disease Detection/skin-disease-datasaet/train_set")


for cls in  sorted(root_file.iterdir()):
    print(cls.name,sum(1 for _ in cls.glob('*')))