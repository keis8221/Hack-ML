import re

def get_path():
    file_name = '/home/tomo/Documents/fit/output_1json.txt'
    path_join = []
    with open(file_name) as f:
        for n, line in enumerate(f):
            if n != 0:
                path_lst = []
                splited = line.split()
                s_file = "/home/tomo/Documents/insta_data/origin/"
                img_file, info_file = "image/", "info/"
                pattern = r"\d+\.jpg"
                image_split = re.findall(pattern, splited[2])
                image_file_join, info_file_join = s_file+img_file+splited[0]+"-"+image_split[0], s_file+info_file+splited[1]
                path_lst.append(splited[0])
                path_lst.append(image_file_join)
                path_lst.append(info_file_join)
                path_join.append(path_lst)
    return path_join

get_path()