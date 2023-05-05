import shutil
import pathlib
from imgann import Convertor

XML_DIR = "/home/badboy-002/github/senior_project/pretrain_dataset/annotations"
IMG_DIR = "/home/badboy-002/github/senior_project/pretrain_dataset/images"
JSON_DIR = "/home/badboy-002/github/senior_project/pretrain_dataset/pretrain_coco.json"

Convertor.voc2coco(IMG_DIR, XML_DIR, JSON_DIR)


"""
Merges PASCAL VOC databases into one master one.

Usage: $ python merge.py /output/dir /voc1 /voc2 /voc2 /voc3
"""

import os
import shutil
import xml.etree.ElementTree

def merge(datasets, output):
    try:
        os.makedirs(output)
        os.makedirs("{0}/Annotations".format(output))
        os.makedirs("{0}/ImageSets/Main".format(output))
        os.makedirs("{0}/JPEGImages".format(output))
    except OSError:
        pass

    print ("Output: {0}").format(output)

    for n, dataset in enumerate(datasets):
        print ("Process Dataset: {0}").format(dataset)

        m = "a" if n > 0 else "w"

        # transcribe imagesets
        for imset in os.listdir("{0}/ImageSets/Main".format(dataset)):
            print ("Process ImageSet: {0}").format(imset)
            path = "{0}/ImageSets/Main/{1}".format(dataset, imset)
            out = open("{0}/ImageSets/Main/{1}".format(output, imset), m)
            for line in open(path):
                line = line.split(" ")
                if len(line) == 2:
                    line = "{0}_{1} {2}".format(n, line[0], line[1])
                else:
                    line = "{0}_{1}\n".format(n, line[0])
                out.write(line)

        # transcraibe JPEGs
        for im in os.listdir("{0}/JPEGImages".format(dataset)):
            print ("Processing JPEG: {0}").format(im)
            path = "{0}/JPEGImages/{1}".format(dataset, im)
            target = "{0}/JPEGImages/{1}_{2}".format(output, n, im)
            shutil.copyfile(path, target)

        # transcribe + merge annotations
        for anno in os.listdir("{0}/Annotations".format(dataset)):
            print ("Processing Annotation: {0}").format(anno)
            path = "{0}/Annotations/{1}".format(dataset, anno)
            target = "{0}/Annotations/{1}_{2}".format(output, n, anno)
            shutil.copy(path, target)

            tree = xml.etree.ElementTree.parse(target)
            root = tree.getroot()
            root.find("filename").text = "{0}_{1}".format(n, root.find("filename").text)
            open(target,"w").write(xml.etree.ElementTree.tostring(root))


if __name__ == "__main__":
    import sys
    merge(sys.argv[2:], sys.argv[1])