from classes.cellimage import CellImage
from classes.contour import ContourOfCell, ContourOfTable, ContourOfLine
from classes.data import Data
from classes.lineimage import LineImage
from classes.pageImage import PageImage, OPERATION_TYPE_TABLE, OPERATION_TYPE_TEXT
from classes.tableimage import TableImage
from classes.text_recognizer import Recognizer, TYPE_ERODE, TYPE_THRESHOLD
from utilities.converter import pdf_to_png

from werkzeug.datastructures import FileStorage

from classes.image import Image
from classes.pageImage import PageImage
from utilities.converter import convert_from_base64, convert_bytes_to_image, pdf_to_png
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


def process_file(file: FileStorage):
    data = Data()
    if file.mimetype == 'application/pdf':
        filename = file.filename.split('\\')[-1]
        file.save(filename)
        images = pdf_to_png(filename)
        for i, img in enumerate(images):
            im = PageImage(img)

            # im.display()

            im.rotate()

            im.find_counters(OPERATION_TYPE_TABLE)

            im.counters = ContourOfTable.filter_contours(im.counters)
            ContourOfTable.set_numbers(im.counters)
            im.counters = ContourOfTable.sort_contours(im.counters)

            # im.display()

            _, im_width = im.get_width_height()

            for ctr in im.counters:
                tbl = TableImage(im.crop_image(ctr))

                _, tbl_width = tbl.get_width_height()
                if tbl_width >= im_width / 2:
                    tbl.draw_lines()

                tbl.resize(2)

                tbl.find_counters()
                tbl.cells = ContourOfCell.filter_contours(tbl.cells)
                ContourOfCell.set_rows(tbl.cells)
                ContourOfCell.set_columns(tbl.cells)
                tbl.cells = ContourOfCell.sort_contours(tbl.cells)

                # tbl.display()

                for cl in tbl.cells:
                    cell = CellImage(tbl.crop_image(cl))

                    rec = Recognizer(cell.image)

                    cl.text_processor.add_with_filter(rec.processing_image(3, 180, TYPE_THRESHOLD))
                    cl.text_processor.add_with_filter(rec.processing_image(3, 150, TYPE_THRESHOLD))
                    cl.text_processor.add_with_filter(rec.processing_image(2, 145, TYPE_THRESHOLD))
                    cl.text_processor.add_with_filter(rec.processing_image(2, 220, TYPE_THRESHOLD))
                    cl.text = cl.text_processor.get_result()
                    print(f"file={file.filename}|{i}|{cl}")
                    data.add_cell(cl, ctr, i, file.filename)

                    # cell.display()

            im.erase_tables()

            im.find_counters(OPERATION_TYPE_TEXT)
            ContourOfLine.set_numbers(im.counters)
            im.counters = ContourOfLine.sort_contours(im.counters)

            for ln in im.counters:
                line = LineImage(im.crop_image(ln))
                # line.resize(2)

                rec = Recognizer(line.image)

                ln.text_processor.add_with_filter(rec.processing_image(3, 180, TYPE_THRESHOLD))
                ln.text_processor.add_with_filter(rec.processing_image(3, 150, TYPE_THRESHOLD))
                ln.text_processor.add_with_filter(rec.processing_image(2, 145, TYPE_THRESHOLD))
                ln.text_processor.add_with_filter(rec.processing_image(2, 220, TYPE_THRESHOLD))
                ln.text = ln.text_processor.get_result()
                print(f"file={file.filename}|{i}|{ln}")
                data.add_line(ln, i, file.filename)

                # line.display()

    data.sort()
    data.df.to_excel("res.xlsx")

    return "res.xlsx"

def do_magic(im):
    print("pytesseract.image_to_string(im)")

