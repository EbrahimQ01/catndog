import torch
import torch.nn as nn
import torch.optim as optim

from model import DogNCatCNN
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# تصاویر رو به یک تصویر 128 در128 تبدیل میکنه 
# به تنصور تبدیل میکنه کی یک جور شیه در یاضی هست که مثل یک ارایه چند بعدی از تصاویر هست 
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# بر اساس نام پوشه سگ و گربه تصاویر برچسب میزند و از ترنسفورمر بالا استفاده میکند
train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

# تصاویر بالا را بهصورت رندم در دسته های 32 عددی میگزارد
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)



# نمونه گیری از کدی که ساختیم
md = DogNCatCNN().to(device)


# در واقع همون طور که از معنی متغییر مشخص هست چیزی هست درواقع میخوام بر اون اساس بگیم که مدلم چقدر به جاده خاکی زده  
# اگر برنماه اشتباه کند مقدار LOSS برزرگ میشود اگر نه کوچک میشود 
#میخوایم کاری کنیم که این مقدار هی کمترشود 
criterion = nn.CrossEntropyLoss()

# این کد وزن های شبکه را تغییر میدهد تا loss کم و کم تر شود 
optimizer = optim.Adam(md.parameters(), lr=0.001)


for i in range(10):
    running_loss = 0
    for img, labels in train_loader:
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # برای اینکه مدل ما اشتباه یاد نگییرد  اشتباهات قبلی را قبل از یادگیری جدید پاک میکنیم تا روی همدیگه انباشت نشن 
        
        outputs = md(img) # تصویر را وارد مدل شبکه ای که در نمونه گیری کردیم میکند
        loss = criterion(outputs, labels) # همون خطی هست که میگه جقدر اشتباه کردی بالا توضیح دادم
        loss.backward() # وزن ها چقدر در این loss یا اشتباه نقش داشتن اما تغییر نمیکند
        optimizer.step() # این خط درواقع وزن ها رو تغییر میده
        running_loss += loss.item()
    print("loss:", i, running_loss/len(train_loader))

## نمایش خروجی
print("number of img:", len(train_dataset)) 
print("Classes", train_dataset.classes)

image, labels = next(iter(train_loader))

print(image.shape) 
print(labels)


# این کد رو مدل مارا ذخیره میکنه تا من مجبور نباشم هر باربرای استفاده یادگیری رو انجام بدم و این کامپیوتر بدتختمن با cori3 نسل 12 بدون گرافیک دهن ش سرویس نشه :)
torch.save(md.state_dict(), "puppy_n_kitty_model.pth")
 
print("done everything!!!!")
