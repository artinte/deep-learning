import easyocr

reader = easyocr.Reader(['ch_sim', 'en'])
result = reader.readtext('docs/res/00/cover.png')
print(result)
