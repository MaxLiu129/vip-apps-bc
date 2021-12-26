import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\123123\Desktop\vcpkg-master\ports\tesseract'

print(pytesseract.image_to_string(r'D:\github_repo\apps-bc-shuihan\algorithm\screenshot1.png'))