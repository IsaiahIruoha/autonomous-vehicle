import os
import xml.etree.ElementTree as ET

def fix_cvat_pascal(xml_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 1) Fix or add <depth>
            size_tag = root.find("size")
            if size_tag is not None:
                depth_tag = size_tag.find("depth")
                # If <depth> is missing or blank, set it to "3"
                if depth_tag is None or (depth_tag.text is None) or not depth_tag.text.strip():
                    if depth_tag is None:
                        depth_tag = ET.Element("depth")
                        size_tag.append(depth_tag)
                    depth_tag.text = "3"  # Assume 3-channel RGB

            # 2) Ensure each <object> has <pose>
            for obj in root.findall("object"):
                pose_tag = obj.find("pose")
                if pose_tag is None:
                    pose = ET.Element("pose")
                    pose.text = "Unspecified"
                    # Insert the <pose> element right after <name> if possible
                    name_index = None
                    for i, child in enumerate(obj):
                        if child.tag == "name":
                            name_index = i
                            break
                    if name_index is not None:
                        obj.insert(name_index + 1, pose)
                    else:
                        obj.append(pose)

                # 3) Remove CVAT <attributes> if present
                attributes_tag = obj.find("attributes")
                if attributes_tag is not None:
                    obj.remove(attributes_tag)

            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"Fixed {xml_file}")

if __name__ == "__main__":
    # Point this to where your .xml files are
    annotations_folder = "/Users/isaiah/Desktop/dataset/annotations"
    fix_cvat_pascal(annotations_folder)
