import cv2
from ocr import BoardOCR

ocr = BoardOCR()
img = cv2.imread("sessions/latest/words.png")
for name, im in ocr._word_list_variants(img):
    res = ocr.reader.readtext(im, detail=1, paragraph=False)
    print("===", name)
    for box, text, conf in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        print(f'  {conf:.2f} "{text}"  cx={sum(xs)/4:.0f} cy={sum(ys)/4:.0f}')

bank = ocr.read_word_bank(img)
print("BANK", bank)
