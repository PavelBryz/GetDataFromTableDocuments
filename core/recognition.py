from utilities.converter import convert_from_base64, convert_bytes_to_image
import pandas
import utilities.converter as conv
# import pytesseract


def process_json(json):
    if json["type"].upper() == "PDF":
        bytes = convert_from_base64(json["file"])
        res = convert_bytes_to_image(bytes)
        for im in res:
            do_magic(im)
    return {"results": "bla, bla"}


def process_json_test(text):
    file = pandas.DataFrame([text])
    file.to_excel("res.xlsx")
    # base64 = conv.convert_to_base64("res.xlsx")
    return "res.xlsx"

def do_magic(im):
    print("pytesseract.image_to_string(im)")

