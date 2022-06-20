from distutils.log import debug
from fileinput import filename
from cv2 import FileStorage_NAME_EXPECTED
from flask import Flask ,request,jsonify, send_file
from pyparsing import original_text_for
import werkzeug
import os
import shutil
import cv2
import pandas as pd
from flask_jsonpify import jsonpify
from flask import send_file


def predict(filenames):

    output_dir = 'c:/Users/sondos/StudioProjects/Medicalv2/test'
    input_dir = "C:/Users/sondos/StudioProjects/Medicalv2/input/"
    yolo_dir = 'C:/Users/sondos/StudioProjects/Medicalv2/yolov5'
    mobileNet_path = 'C:/Users/sondos/StudioProjects/Medicalv2/mobileNetModel.h5'
    logistic_path = 'C:/Users/sondos/StudioProjects/Medicalv2/logisticModel.sav'

    original_image = cv2.imread(input_dir + filenames[0])
    o_height, o_width, _ = original_image.shape
    o_size = max(o_height, o_width) if max(o_height, o_width) > 416 else 416

    # some attr
    wbc_count = 0
    types_count = [0, 0, 0, 0]

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)

    # remove old yolo prediction
    if os.path.isdir(yolo_dir + r'/runs/detect'):
        shutil.rmtree(yolo_dir + r'/runs/detect')

    os.system(r'python "' + yolo_dir + '/detect.py" --weights "' + yolo_dir
              + r'/runs/train/yolov5s_results/weights/best.pt" --img ' + str(o_size) + ' --conf 0.62  --source "' +
              input_dir + '" --save-txt')

    for filename in filenames:

        # get yolo prediction and loop through them
        if os.path.exists(yolo_dir + r'/runs/detect/exp/labels/'+os.path.splitext(filename)[0]+'.txt'):

            f = open(yolo_dir + r'/runs/detect/exp/labels/' + os.path.splitext(filename)[0]+'.txt', 'r')
            for line in f:
                line = line.split()
                types_count[int(line[0])] += 1
                # count WBC to name the tmp images
                wbc_count += 1
                print('WBC #', wbc_count)

            f.close()

    final_count = pd.DataFrame([types_count], columns=['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'])
    final_count[['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']].to_csv(output_dir + r'\final_count.csv',
                                                                               index=False)  


app=Flask(__name__)
@app.route("/upload",methods = ['GET', 'POST'])
def upload():
    if request.method=="POST":

        fileNamess = []
        # imagefile =request.files["image"]
        if os.path.isdir('/Users/sondos/StudioProjects/Medicalv2/input/'):
                shutil.rmtree('/Users/sondos/StudioProjects/Medicalv2/input/')
                os.mkdir('/Users/sondos/StudioProjects/Medicalv2/input/')
        for item in request.files.getlist('image'):
            # data = item.read()           
            filename = werkzeug.utils.secure_filename(item.filename)
            # print(filename)         
            item.save("/Users/sondos/StudioProjects/Medicalv2/input/"+filename)
            # print("save image")
            fileNamess.append(filename)
            
        predict(fileNamess)
        return send_file('test/final_count.csv',
                     mimetype='text/csv',
                     attachment_filename='final_count.csv',
                     as_attachment=True)
      
if __name__ =="__main__":
    app.run(debug=True,port=80)