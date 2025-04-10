import json
import pandas as pd
import cv2
import os

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
            '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            '[','\\',']','^','_','`',
            'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            '{','|','}','~']

# ICDAR2015
# TOTAL_FILES = 500
# f = open('ic15_test.json')

# TotalText
# TOTAL_FILES = 300
# with open('test.json', 'r') as f:
#     data = json.load(f)
#     data.keys()
    
#TOTAL_FILES = 1255
#with open('train.json', 'r') as f:
 #   data = json.load(f)
  #  data.keys()

#SynText150k
TOTAL_FILES = 94723
with open('train.json', 'r') as f:
    data = json.load(f)
    data.keys()


TXT_FORMAT = True
JSON_FORMAT = False

for image_id in range(0, TOTAL_FILES, 1):
    print("image_id={0}".format(image_id))
    instance_id = -1
    
    total_anno = []
    output_folder = 'output_folder'
    if not os.path.exists(output_folder):
       os.makedirs(output_folder)
    for index, item in enumerate(data['annotations']):
        inst_anno = {}
        image_number_str  = "{:07d}".format(image_id)
        if item['image_id'] == image_id:
            instance_id = instance_id + 1
            
            # decoce recs
            #recs = []
            single_word = ""
            for char_index in range(len(item['rec'])):
                if item['rec'][char_index] == 96:
                    decode_char = "."
                else:
                    decode_char = CTLABELS[item['rec'][char_index]]
                #recs.append(decode_char)
                single_word = single_word + decode_char
                
            if TXT_FORMAT:
                image_id_str = "image_id: " + image_number_str            
                bbox_str = "bbox: {0}".format(item['bbox'])            
                bezier_pts_str = "bezier_pts: {0}".format(item['bezier_pts'])

                #rec_str = "rec: {0}".format(recs)
                rec_str = "rec: {0}".format(single_word)
                instance_id_str = "text_instance: {0}".format(instance_id)
                
                file_path = os.path.join(output_folder, image_number_str + ".txt")

                f = open(file_path, "a")
                f.write(instance_id_str + "\n")
                f.write(bbox_str + "\n")
                f.write(bezier_pts_str + "\n")
                f.write(rec_str + "\n")
                f.close()

            if JSON_FORMAT:
                inst_anno["image_id"] = item['image_id']
                inst_anno["bbox"] = item['bbox']
                inst_anno["bezier_pts"] = item['bezier_pts']
                print("item['bezier_pts']={0}".format(item['bezier_pts']))
                inst_anno["rec"] = item['rec']
                #inst_anno["rec_decode"] = recs
                inst_anno["rec_decode"] = single_word
                total_anno.append(inst_anno)
                print("append total_anno")
    if JSON_FORMAT:
        with open("{0}.json".format(image_number_str), 'w') as f:
            json.dump(total_anno, f, indent=2)
            print("write total_anno")

