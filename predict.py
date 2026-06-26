import torch
from model import DogNCatCNN
import cv2
from torchvision import transforms



# این خط کد میاد میگه اگر گرافیک nvidia داری میتونی از cuda استفاده کنی 
#در غیر اینصورت از cpu که فکر کنم باتوجه به وضعی که هست به درد کسی نمی خوره باتوحه به قمت ا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# مدل رو من میسازم و مدلی که ذخیره کردم رو لود میکنم
md = DogNCatCNN().to(device)
md.load_state_dict(torch.load("puppy_n_kitty_model.pth"))

#قرار داردن مدل در حالت تست
md.eval()


# اینجا چون مدل نمیتونه که عکس خام بگیرده عکس ها رو به تنسور تبدیل میکنیم
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# تصویر رو میخوانیم
img = cv2.imread("dog.3.jpg")

# رنگ بندی رو مشخص میکنه
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# اینجا از تنسوری که بالا ساختیم استفاده میکنیم و تصویر رو به فرمت که شبکه عصبی بهمه تبدیل میکنیم
tensor = transform(img)

# میخوایم که همیشه بتج به یک شکل تصویر رو داشته باشه حتا اگر یکی باشه
tensor = tensor.unsqueeze(0)
tensor = tensor.to(device)


# از هیچ  گرادینتس استفاده نکن فقط برایپیشتینی
with torch.no_grad():
    output = md(tensor)

#  خروجی ما درواقع 2 عدد هست ما بزرگ ترین رو انتخاب  میکنم که درواقع میگه از هر کدوم عکس احتمال کدم بیشتهر
# یک میگه در چه بعدی باشه
# مثلا اگر خروجی صفر باشه گربه هست اگر که یک پس سگه  
prediction = torch.argmax(output, 1).item()

print(prediction)
if prediction == 0:
    print("cat")
else:
    print("dog")
    
