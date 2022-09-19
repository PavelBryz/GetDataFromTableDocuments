from utilities.converter import convert_from_base64, convert_bytes_to_image
# import pytesseract

def process_json(json):
    if json["type"].upper() == "PDF":
        bytes = convert_from_base64(json["file"])
        res = convert_bytes_to_image(bytes)
        for im in res:
            do_magic(im)
    return {"results": "bla, bla"}


def do_magic(im):
    print("pytesseract.image_to_string(im)")

