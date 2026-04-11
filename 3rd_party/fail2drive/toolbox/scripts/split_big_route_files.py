from lxml import etree
import os
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--xml-file", type=str, required=True, help="The big xml file to split")
parser.add_argument("--out-folder", type=str, required=True, help="The folder to save the split files")
args = parser.parse_args()

tree = etree.parse(args.xml_file)
routes = tree.getroot()
id = 0
for route in routes:
    new_tree = etree.ElementTree(etree.Element("routes"))
    root = new_tree.getroot()
    root.append(route)

    filename = f"{pathlib.Path(args.xml_file).name.replace('.xml', '')}_{id:02d}.xml"
    with open(os.path.join(args.out_folder, filename), "wb") as f:
        test = etree.tostring(new_tree, xml_declaration=True, encoding="utf-8", pretty_print=True)
        f.write(test)
    new_tree.write(os.path.join(args.out_folder, filename))
    id += 1
