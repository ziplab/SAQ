import os


def link_imagenet(src_data_path, dst_data_path, sample_path):
    classname_list = []
    with open(sample_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            classname_list.append(line)
    print(classname_list)

    for dataset in ["train", "val"]:
        src_path = os.path.join(src_data_path, dataset)
        dst_path = os.path.join(dst_data_path, "{}100".format(dataset))

        for classname in classname_list:
            src_dir = os.path.join(src_path, classname)
            dst_dir = os.path.join(dst_path, classname)

            print("Link {} to {}".format(src_dir, dst_dir))

            os.symlink(src_dir, dst_dir)


if __name__ == "__main__":
    src_data_path = "path_of_imagenet"
    dst_data_path = "path_of_imagenet"
    sample_path = "./sampled_classes.txt"
    link_imagenet(src_data_path, dst_data_path, sample_path)
