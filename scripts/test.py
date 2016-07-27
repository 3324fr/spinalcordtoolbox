

dirs_name = []
folder_path = '/Users/cavan/data/essais_t2s/'
for root, dirs, files in os.walk(folder_path):
    for dir in dirs:
        if dir.find("t2s") == -1 and dir.find("t2") == -1 and dir.find("mt") == -1 and dir.find("dmri") == -1:
            dirs_name.append(dir)

list_data = []
for iter, dir in enumerate(dirs_name):
    for root, dirs, files in os.walk(folder_path + '/' + dir):
        for file in files:
            if file.endswith('t2s.nii.gz') and file.find('_gmseg') == -1 and file.find('_seg') == -1:
                pos_t2 = file.find('t2s')
                subject_name, end_name = file[0:pos_t2 + 3], file[pos_t2 + 3:]
                file_seg = subject_name + '_seg_manual' + end_name
                file = file
                list_data.append(folder_path + '/' + file)
                list_data.append(folder_path + '/' + file_seg)
                im = Image(file)
                im.setFileName('/Users/cavan/src/' + )
    return list_data, dirs_name