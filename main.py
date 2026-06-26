import cv2

img = cv2.imread("./cat.jpg")



print("image type:",type(img)) # نمایش نوع عکس از ای ارایه numpy هست
print("resolation and chanel:",img.shape) # نشون دهنده ابعاد و نوع عکس 

# print(img) # نشون دادن مقادیر هر پیکسل

print(img[0][0]) # نشون دادن مقادیر اولین پیکسل
print(img[156][213])

# img[0][0] = [0,0,255] # تغییر اولین رنگ پیکسل به رنگ قرمز کامل


# تغییر رنگ یک خانه 100 در 100 در تصویر به قرمز کامل
for y in range(100):
    for x in range(100):
        img[y][x] = [0,0,255]

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
