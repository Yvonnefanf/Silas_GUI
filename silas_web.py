 # -*- coding: utf-8 -*
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,make_response, jsonify, send_file, abort
from markupsafe import escape
import os
from werkzeug.utils import secure_filename
import json
from flask_bootstrap import Bootstrap
import random
from flask_cors import CORS, cross_origin
import psutil
import time
import csv
import shutil

import numpy as np


from sklearn.metrics import accuracy_score

from OptExplain.silas import RFC
from OptExplain.Main_Process import MainProcess


ROUND_NUMBER = 6

from FeatureImportance.utils import load_data
#TODO
from FeatureImportance.muc import REATURE_RFC, MUC
from FeatureImportance.shapley import Shapley
from FeatureImportance.adversarial_sample import AdversarialSample

import pandas as pd
import glob

app = Flask(__name__, static_folder='static')
CORS(app)
bootstrap = Bootstrap(app)

@app.route('/')
def index():
    return redirect(url_for('SilasGUI'))


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return do_the_login()
    else:
        return show_the_login_form()

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        base_path = os.path.dirname(__file__)
        #file_path = os.path.join(base_path,"static/uploads",secure_filename(f.filename))
        file_path = base_path + "/uploads/file"
        f.save(file_path)
        return redirect(url_for('upload'))
    return render_template('upload.html')

def decode_para(para):
    res = []
    s = para.split("&")
    for x in s:
        res.append(x.split("="))
    return res


def default_para():
    para = [
        "number_of_data=未知",
        "number_of_feats=未知",
        "target=未知"
    ]
    res = ""
    for x in para:
        res = res + x + "&"
    res = res[0:len(res)-1]
    return res


def get_current_para(result_path):

    res = ""

    f = open("%s/feature-stats.json"%result_path,)
    feat_stat = json.load(f)
    f.close()
    num_data = feat_stat[-1]["stats"]["mean-count"]
    res = res + "number_of_data=%s"%num_data

    target = feat_stat[-1]["attribute-name"]
    res = res + "&target=%s"%target

    num_feats = len(feat_stat) - 1
    res = res + "&number_of_feats=%s"%num_feats

    f = open("%s/web.log"%result_path, "r")
    WL = f.readlines()
    for x in WL:
        if "Silas_model_id=" in x:
            y = x.replace("\n","")
            res = res + "&" + y
            break


    return res
 


def Silas_training(training_data_file, validation_data_file, hyper_parameters):
   
    #os.system("cd silas-temp")
    os.system("./silasPro gen-all -o silas-temp/results %s %s > silas-temp/results/gen.log"%(training_data_file, validation_data_file))

    f_in = open("silas-temp/results/settings.json", "r")
    f_out = open("silas-temp/results/settings.json_new", "w")
    for x in f_in.readlines():
        x = x.replace("\"mode\": \"classification\"", "\"mode\": \"%s\""%hyper_parameters['task-mode'])
        x = x.replace("\"type\": \"GreedyNarrow1D\"", "\"type\": \"%s\""%hyper_parameters['tree-algorithm'])
        x = x.replace("\"feature-proportion\": \"sqrt\"", "\"feature-proportion\": \"%s\""%hyper_parameters['feature-proportion'])
        x = x.replace("\"max-depth\": 64", "\"max-depth\": %s"%hyper_parameters['max-depth'])
        x = x.replace("\"desired-leaf-size\": 64", "\"desired-leaf-size\": %s"%hyper_parameters['desired-leaf-size'])
        x = x.replace("\"type\": \"ClassicForest\"", "\"type\": \"%s\""%hyper_parameters['forest-algorithm'])
        x = x.replace("\"number-of-trees\": 100", "\"number-of-trees\": %s"%hyper_parameters['number-of-trees'])
        x = x.replace("\"sampling-proportion\": 1.0", "\"sampling-proportion\": %s"%hyper_parameters['sampling-proportion'])
        x = x.replace("\"oob-proportion\": 0.05", "\"oob-proportion\": %s"%hyper_parameters['oob-proportion'])
        f_out.write(x)
    f_in.close()
    f_out.close()

    cmd = "mv silas-temp/results/settings.json_new silas-temp/results/settings.json"
    os.system(cmd)

    os.system("./silasPro learn -o silas-temp/results/model silas-temp/results/settings.json > silas-temp/results/train.log")
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    fid = int(random.random() * 10000000000)
    os.system("tar -cvf silas-temp/results-%d.tar silas-temp/results"%fid)
    f = open("silas-temp/results/web.log", "w")
    f.write("Silas_model_id=%d\n"%fid)
    f.close()
    #os.system("cd ..")
    return 0


def Silas_validation(training_data_file, hyper_parameters):
    os.system("./silasPro gen-all -o silas-temp/validation-results %s > silas-temp/results/gen.log"%(training_data_file))

    f_in = open("silas-temp/validation-results/settings.json", "r")
    f_out = open("silas-temp/validation-results/settings.json_new", "w")
    for x in f_in.readlines():
        x = x.replace("\"mode\": \"classification\"", "\"mode\": \"%s\""%hyper_parameters['task-mode'])
        x = x.replace("\"type\": \"GreedyNarrow1D\"", "\"type\": \"%s\""%hyper_parameters['tree-algorithm'])
        x = x.replace("\"feature-proportion\": \"sqrt\"", "\"feature-proportion\": \"%s\""%hyper_parameters['feature-proportion'])
        x = x.replace("\"max-depth\": 64", "\"max-depth\": %s"%hyper_parameters['max-depth'])
        x = x.replace("\"desired-leaf-size\": 64", "\"desired-leaf-size\": %s"%hyper_parameters['desired-leaf-size'])
        x = x.replace("\"type\": \"ClassicForest\"", "\"type\": \"%s\""%hyper_parameters['forest-algorithm'])
        x = x.replace("\"number-of-trees\": 100", "\"number-of-trees\": %s"%hyper_parameters['number-of-trees'])
        x = x.replace("\"sampling-proportion\": 1.0", "\"sampling-proportion\": %s"%hyper_parameters['sampling-proportion'])
        x = x.replace("\"oob-proportion\": 0.05", "\"oob-proportion\": %s"%hyper_parameters['oob-proportion'])
        f_out.write(x)
    f_in.close()
    f_out.close()

    cmd = "mv silas-temp/validation-results/settings.json_new silas-temp/validation-results/settings.json"
    os.system(cmd)

    os.system("./silasPro learn -o silas-temp/validation-results/model silas-temp/validation-results/settings.json > silas-temp/validation-results/train.log")



    # templates/Silas-Bootstrap-treeview/silas-model-to-html.py
    print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    fid = int(random.random() * 10000000000)
    os.system("tar -cvf silas-temp/validation-results-%d.tar silas-temp/validation-results"%fid)
    f = open("silas-temp/validation-results/web.log", "w")
    f.write("Silas_model_id=%d\n"%fid)
    f.close()
    #os.system("cd ..")
    return 0


def Silas_test(test_data_file):
    #os.system("cd silas-temp")
    os.system("./silasPro predict -o silas-temp/results/predictions.csv silas-temp/results/model/ silas-temp/results/test.csv")
    s = "number_of_data=1,000,000"
    return s
################################################# silas GUI global #################################################

@app.route("/SilasGUI", methods=["GET", "POST"])
def SilasGUI():
    # return render_template("SilasIndex.html")
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/SilasGUI/uploadFile", methods=["POST", "GET"])
def uploadFiles():
    if request.method == 'POST':
        f = request.files['file']
        fileName = request.form['fileName']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,"silas-temp/uploads",fileName)
        f.save(file_path) 
    return make_response(jsonify({'path': file_path}), 200)


@app.route("/SilasGUI/downloadFile", methods=["GET"])
def downloadFiles():
    safe_path = request.args.get("path")
    try:
        if os.path.exists(safe_path):
            print("exists")
            directory_path, filename = os.path.split(safe_path)
            return send_from_directory(directory_path, filename, as_attachment=True)
        else:
            return jsonify({'code': 404, 'message': 'File not found'}), 404
    except Exception as e:
            return jsonify({'code': 500, 'message': 'Internal server error', 'error': str(e)}), 500
        # return make_response(jsonify({'code':20000}), 200)
        # return redirect(url_for("SilasTrainPage"))
    
@app.route("/SilasGUI/deleteFile", methods=["GET"])
def deleteFile():
    safe_path = request.args.get("path")
    try:
        if not os.path.exists(safe_path):
            return jsonify({'code': 404, 'message': 'File not found'})
        else:
            os.remove(safe_path)
            return jsonify({'code': 20000, 'message': 'deleted'})
     
    except Exception as e:
            return jsonify({'code': 500, 'message': 'Internal server error', 'error': str(e)}), 500
        # return make_response(jsonify({'code':20000}), 200)
        # return redirect(url_for("SilasTrainPage"))

################################################# silas GUI Train Page #######################################################################################
#get train res
@app.route("/SilasGUI/trainPage/getTrainResult", methods=["POST", "GET"])
def getTrainResult():
    with open('silas-temp/results/lastest.json','r',encoding = 'utf-8') as fp:
        setting_data = json.load(fp)
    trainlog = open("silas-temp/results/train.log")
    logs = trainlog.readlines()
    strlines = []
    for line in logs:
        strlines.append(line)
    return make_response(jsonify({'code':20000,'TrainResult': strlines,'setting_data':setting_data}), 200)

#get train res of tree model
@app.route("/SilasGUI/trainPage/getTreeData", methods=["POST", "GET"])
def getTreeData():
    treeDatalist = []
    fileList = os.listdir("silas-temp/results/model/tree-store")
    for json_id in fileList:
        res_path = "silas-temp/results/model/tree-store/" + json_id
        with open(res_path,"r") as fp:
            data_j = json.load(fp)
            treeDatalist.append(data_j)
    return make_response(jsonify({'code':20000,'treeDataList': treeDatalist}), 200)


@app.route("/SilasGUI/starttrain", methods=["POST"])
def SilasstartTrain():
    if request.method == "POST":
        base_path = os.path.dirname(__file__)

        file_path_train = "silas-temp/uploads/train.csv"

        file_path_valid = "silas-temp/uploads/test.csv"

        hyper_parameters = dict([])
        print('request',request.json)
    
        for x in request.json:
            hyper_parameters[x] = request.json[x]
        with open('silas-temp/results/lastest.json', 'w', encoding='utf-8') as file_obj:
            json_hyper_parameters = json.dumps(hyper_parameters)
            file_obj.write(json_hyper_parameters)

        Silas_training(file_path_train, file_path_valid, hyper_parameters)
        # para_new = get_current_para("silas-temp/results")
        # create_Silas_temp_page(para_new)
        with open('silas-temp/results/lastest.json','r',encoding = 'utf-8') as fp:
            setting_data = json.load(fp)
        trainlog = open("silas-temp/results/train.log")
        logs = trainlog.readlines()
        strlines = []
        for line in logs:
            strlines.append(line)
        return make_response(jsonify({'code':20000, 'TrainResult': strlines,'setting_data':setting_data}), 200)
        # return redirect(url_for("SilasTrainPage"))

################################################# silas GUI Validation Page #######################################################################################
#get validation res
@app.route("/SilasGUI/testPage/getValResult", methods=["POST", "GET"])
def getValTrainResult():
    with open('silas-temp/results/val_lastest.json','r',encoding = 'utf-8') as fp:
        setting_data = json.load(fp)
    strlines = []
    if os.path.exists('silas-temp/validation-results/train.log'):
        trainlog = open("silas-temp/validation-results/train.log")
        logs = trainlog.readlines()
        for line in logs:
            strlines.append(line)
    return make_response(jsonify({'code':20000,'TrainResult': strlines,'setting_data':setting_data}), 200)

#get validation res of tree model
@app.route("/SilasGUI/testPage/getTreeData", methods=["POST", "GET"])
def getVaTreeData():
    treeDatalist = []
    tree_store_path = "silas-temp/validation-results/model/tree-store"
    if os.path.exists(tree_store_path):
        fileList = os.listdir(tree_store_path)
        for json_id in fileList:
            res_path = "silas-temp/validation-results/model/tree-store/" + json_id
            with open(res_path,"r") as fp:
                data_j = json.load(fp)
                treeDatalist.append(data_j)
    return make_response(jsonify({'code': 20000,'treeDataList': treeDatalist}), 200)

@app.route("/SilasGUI/startvalidation", methods=["POST"])
def Silasstartvalidation():
    if request.method == "POST":
        base_path = os.path.dirname(__file__)

        file_path_train = "silas-temp/uploads/train.csv"

        hyper_parameters = dict([])
        print('request',request.json)
        for x in request.json:
            hyper_parameters[x] = request.json[x]
        with open('silas-temp/results/val_lastest.json', 'w', encoding='utf-8') as file_obj:
            json_hyper_parameters = json.dumps(hyper_parameters)
            file_obj.write(json_hyper_parameters)

        Silas_validation(file_path_train, hyper_parameters)

        with open('silas-temp/results/val_lastest.json','r',encoding = 'utf-8') as fp:
    
            # load()函数将fp(一个支持.read()的文件类对象，包含一个JSON文档)反序列化为一个Python对象
            setting_data = json.load(fp)
        strlines = []
        if os.path.exists('silas-temp/validation-results/train.log'):
            trainlog = open("silas-temp/validation-results/train.log")
            logs = trainlog.readlines()
            for line in logs:
                strlines.append(line)
        return make_response(jsonify({'code':20000,'TrainResult': strlines,'setting_data':setting_data}), 200)

################################################# silas GUI Prediction Page #######################################################################################
@app.route("/SilasGUI/predictPage/prediction", methods=["POST"])
def SilasPredict():
    if request.method == 'POST':
        base_path = os.path.dirname(__file__)

        res = request.get_json()
        model_path = res['model_path']
        print("mode",model_path)
        store_path = 'silas-temp/results/prediction.csv'
        if model_path == 'silas-temp/validation-results/model':

            store_path = 'silas-temp/validation-results/prediction.csv'
    
        print("store_path",store_path)
    
        os.system("./silasPro predict -o %s %s silas-temp/uploads/unlabeled.csv"%(store_path, model_path))
        return make_response(jsonify({'code': 20000}), 200)

@app.route("/SilasGUI/predictPage/getRes", methods=["POST","GET"])
def getPredcitionRes():
    type = request.args.get("type")
    path = 'silas-temp/results/prediction.csv'
    if type == "validation":
        path = 'silas-temp/validation-results/prediction.csv'
    results = []
    f = open(path, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    header = rows[0].split(",")

    # for row in rows[1:]:
    #     item = {}
    #     ele = row.split(',')
    #     if ele != [''] and len(ele)>=1:
    #         for i in range(len(header)-1):
    #             item[header[i]] = ele[i]
    #         final_list.append(item)
    for row in rows:
        ele = row.split(',')
        strs = "\t".join(ele)
        final_list.append(strs)
    return make_response(jsonify({'code': 20000,'final_list':final_list}), 200)



################################################# silas GUI Grid Search Page #######################################################################################
# generate grid search setting
@app.route("/SilasGUI/gridSearchPage/gengridSearch", methods=["POST", "GET"])
def genGridSearch():
    os.system("./silasPro gen-gridsearch-settings -m c -o silas-temp/grid-search/gridsearch-settings.json")
    f = open("silas-temp/grid-search/gridsearch-settings.json")
    gridsearchjson = json.load(f)
    return make_response(jsonify({'code':20000, 'gridsearchjson': gridsearchjson}), 200)

def remove_folder(path):
    # 判断路径是否存在
    if os.path.exists(path):
        # 获取指定路径下的所有文件和文件夹
        files = os.listdir(path)
        for file in files:
            # 拼接文件路径
            file_path = os.path.join(path, file)
            # 判断是否是文件夹
            if os.path.isdir(file_path):
                # 递归调用函数，删除子文件夹下的所有文件和文件夹
                remove_folder(file_path)
            else:
                if file != 'gridsearch-settings.json':
                    os.remove(file_path)
        # check
        remaining_files = os.listdir(path)
        if len(remaining_files) == 0 or (len(remaining_files) == 1 and 'gridsearch-settings.json' not in remaining_files):
            #  if fold empty or not include gridsearch-settings.json delete
            try:
                os.rmdir(path)
            except OSError as e:
                print(f"Error: {e.strerror}")
 
# generate grid search setting
@app.route("/SilasGUI/gridSearchPage/gridSearch", methods=["POST", "GET"])
def GridSearch():
    # befor search remove other 
    res = request.get_json()
    settingsPath = res['jsonPath']
    jsonData = res['gridSettingJson']
    

    with open('silas-temp/grid-search/gridsearch-settings.json', 'w', encoding='utf-8') as file_obj:
            file_obj.write(jsonData)
    
    remove_folder("silas-temp/grid-search")
    os.system("./silasPro gridsearch silas-temp/grid-search/gridsearch-settings.json silas-temp/results/settings.json")
    # os.system("./silasPro gridsearch -o silas-temp/gridsearchresults %s %s > silas-temp/results/gridsearch.log"%(training_data_file, validation_data_file))
    return  make_response(jsonify({'code':20000}), 200)

@app.route("/SilasGUI/gridSearchPage/getGridSearchRes", methods=["GET"])
def getGridSearchResult():
 
    file_path = r'silas-temp/grid-search'                                     #文件夹位
    file = glob.glob(os.path.join(file_path, 'batch*', "*.csv"))                   #文件列表
    if len(file) == 0:
        return make_response({'code':20000,'data': []}, 200)
    file.sort()
    data = pd.read_csv(file[0])
    arr = []
    for i in range(len(file)):
        with open(file[i]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['path'] = file[i].replace("results.csv","")
                res_path = file[i].replace("results.csv","learner-settings.json")
                with open(res_path,"r") as fp:
                    data_j = json.load(fp)
                    row['settings'] = data_j
                arr.append(row)          
    return make_response(jsonify({'code':20000,'data': arr}), 200)



@app.route("/SilasGUI/reset", methods=["POST", "GET"])
def SilasReset():
    os.system("rm -r silas-temp")
    #os.system("cp -r silas-pro silas-temp")
    os.system("mkdir silas-temp")
    os.system("mkdir silas-temp/results")
    # create_Silas_temp_page(default_para())
    return redirect(url_for("SilasGUI"))




@app.route("/SilasGUI/getRes", methods=["GET"])
def silasRes():

    base_path = os.path.dirname(__file__)
    res_path = os.path.join(base_path, 'silas-temp/results/feature-stats.json')
    with open(res_path,"r") as fp:
        feature_stats = json.load(fp)
    
    return make_response(jsonify({'result': '8888','base':base_path,'feature-stats':feature_stats}), 200)

################################################# silas GUI Dashboard #######################################################################################
@app.route("/SilasGUI/getStastisticData",methods=["GET"])
def getStastisticData():
    finishedTrainings = 0
    explanations = 0
    gridSearchModels = 0
    finishedValidation = 0
    finishedTrainingsList = []
    explanationsList = []
    for root,dirs,files in os.walk('silas-temp/grid-search'):    #遍历统计
        for dir in dirs:
            gridSearchModels += 1 
    for root,dirs,files in os.walk('silas-temp/explanation'):    #遍历统计
        for file in files:
            explanations += 1 
            fullPath = os.path.join(root, file)  # Construct full path of the file
            fileType = "validation" if "validation" in file else "train" 
            explanationsList.append({'path': fullPath})
    for root,dirs,files in os.walk('silas-temp'):    #遍历统计
        for file in files:
            if file.endswith('.tar'):
                finishedTrainings += 1 
                fullPath = os.path.join(root, file)  # Construct full path of the file
                fileType = "validation" if "validation" in file else "train" 
                finishedTrainingsList.append({'path':fullPath,'type':fileType})  # Append the full path to the list

    # for root,dirs,files in os.walk('silas-temp'):    #遍历统计
    #     for file in files:
    #         finishedTrainings += 1 
        
    return make_response(jsonify({'code':20000, 'finishedTrainings':finishedTrainings ,"gridSearchModels":gridSearchModels,"explanations":explanations, "explanationsList":explanationsList, "finishedTrainingsList":finishedTrainingsList}), 200)

@app.route("/SilasGUI/getCpuMemo", methods=["GET"])
def getCpuMemo():
    b = psutil.cpu_percent(interval=1.0)  # cpu
    a = psutil.virtual_memory().percent  # memo
    cpunum = psutil.cpu_count(logical=False)

    nowtime = time.strftime("%H:%M:%S", time.localtime())

    physical_hard_disk = []
    for disk_partition in psutil.disk_partitions():
        o_usage = psutil.disk_usage(disk_partition.device)
        physical_hard_disk.append(
            {
                "device": disk_partition.device,
                "fstype":disk_partition.fstype,
                "opts": disk_partition.opts,
                "total": o_usage.total,
            }
        )

    return make_response(jsonify({'time': nowtime,"cpu":a,"memo":b,"cpunum":cpunum,"physical_hard_disk":physical_hard_disk}), 200)

# result = []
def path_to_dict(path):
    item = os.path.basename(path)
    value = shutil.disk_usage(path)
    d = {'name': item, "used":value.used,"free":value.free,"value":value.used, 'children': [path_to_dict(os.path.join(path, x)) for x in os.listdir(path) if
                                    os.path.isdir(os.path.join(path, x)) and os.path.basename(os.path.join(path, x))[
                                        0] not in ["~", "."]]}
    return d

@app.route("/SilasGUI/getDiskStatus",methods=["GET","POST"])
def getDiskStatus():

    req_data = request.args.get("search_dir")
    usage = path_to_dict(req_data)

    return make_response(jsonify(usage["children"]), 200)



@app.route("/SilasGUI/modeldownload/<fid>", methods=["POST", "GET"])
def SilasModelDownload(fid):
    base_path = os.path.dirname(__file__)
    model_dir = base_path + "/silas-temp/"
    model_file = "results-%s.tar"%fid
    try:
        if os.path.exists(model_dir + model_file):
            return send_from_directory(model_dir, filename=model_file, as_attachment=True)
        else:
            return "<h1>文件%s不存在</h1>"%(model_dir + model_file)
    except:
        return "<h1>文件%s无法下载</h1>"%(model_dir + model_file)


@app.route("/SilasGUI/extensionPage/explanation", methods=["POST"])
def optexplain():
    # get parameters from front-end
    res = request.get_json()
    # print("res",res)
    model_path = res['modelPath']
    # test_file = res['testFilePath']
    # pf = res['predictionFilePath']
    test_file = 'silas-temp/uploads/explanation_test.csv'
    pf = 'silas-temp/uploads/explanation_prediction.csv'

    # set the hyperparameters
    conjunction = False
    maxsat_on = False
    size_filter = True

    generation = 20
    scale = 20
    acc_weight = 0.5
    # read the test data
    with open(os.path.join(model_path, 'metadata.json')) as f:
        metadata = json.load(f)
    test_data = pd.read_csv(test_file)
    columns = list(test_data.columns)
    if os.path.exists(os.path.join(model_path, 'settings.json')):
        with open(os.path.join(model_path, 'settings.json')) as f:
            settings = json.load(f)
        label_column = columns.index(settings['output-feature'])
    else:
        label_column = len(columns) - 1
    test_data = test_data.values.tolist()

    ## adjust data type
    for i, f in enumerate(metadata['attributes']):
        if f['type'] == 'nominal' and isinstance(test_data[0][i], (float, int)) and i != label_column:
            for j, sample in enumerate(test_data):
                test_data[j][i] = str(round(float(sample[i]), ROUND_NUMBER))
    X_test = [sample[:label_column] + sample[label_column + 1:] for sample in test_data]

    # create random forest from Silas
    print('RF...', end='\r')
    clf = RFC(model_path, pred_file=pf, label_column=label_column)
    y_test = []
    for sample in test_data:
        y = sample[label_column]        

        # Zhe modified the below on 30/03/2023. Original code:
        # if isinstance(y, int) or (isinstance(y, float) and y.is_integer()):
        #     y = str(int(y))

        if isinstance(y, bool):
            y = str(y)
        else:
            try:
                y = float(y)
            except ValueError:
                pass

        if isinstance(y, int) or (isinstance(y, float) and y.is_integer() and str(int(y)) in clf.classes_):
            y = str(int(y))   
        elif str(y) in clf.classes_:
            y = str(y)    

        y_test.append(clf.classes_.index(y))  # int labels
    print('RF acc:', accuracy_score(y_test, clf.predict()))

    # output
    base_name = os.path.basename(model_path)
    file_num = 1
    if not os.path.exists('explanation'):
        os.makedirs('explanation')
    while os.path.exists(f'explanation/{base_name}_{file_num}.txt') is True:
        file_num += 1
    file = open(f'explanation/{base_name}_{file_num}.txt', 'w')
    cur_file = 'explanation/{}_{}.txt'.format(base_name,file_num )
    file.write('generation = {}\tscale = {}\tacc_weight = {}\tmaxsat = {}\ttailor = {}\n\n'.
               format(generation, scale, acc_weight, maxsat_on, size_filter))
    print('explain...')
    m = MainProcess(clf, X_test, y_test, file, generation=generation, scale=scale, acc_weight=acc_weight,
                    conjunction=conjunction, maxsat_on=maxsat_on, tailor=size_filter, fitness_func='Opt')
    best_param, posList = m.pso()
    m.explain(best_param, auc_plot=False)
    # best_param, posList = m.pso()
    explainRes, performance = m.explain(best_param, auc_plot=False)
    file.close()
    return make_response(jsonify({'code':20000, 'explainRes':explainRes,"posList":posList,"performance":performance, "cur_file": cur_file}), 200)

@app.route("/SilasGUI/featurePage/featureImportance", methods=["POST", "GET"])
def featureImportance():
    res = request.get_json()
    imtype = res["type"]
    model_path = res["model_path"]
   
    test_file = "silas-temp/uploads/featureImportance_test.csv"
    X_test, y_test, label_column = load_data(model_path, test_file)
    rfc = REATURE_RFC(model_path, X_test, y_test, label_column)
    muc = MUC(rfc)
    numberOfsubsets = None

    featureImportanceList = []
    adversarialObj = {}
    if imtype == 'feature':
        print('========== M-Shapley ==========')
        shapley = Shapley(muc, verbose=True)
        if not os.path.exists('FeatureImportance/muc_save'):
            os.mkdir('FeatureImportance/muc_save')
        save_file = f'FeatureImportance/muc_save/{model_path.split("/")[-1]}.json'
        shapley.compute_muc(X_test, y_test, save_file)
        res = dict()
        for c in range(rfc.n_classes_):
            res[c] = shapley.value(c, numberOfsubsets)
            print(f'class "{rfc.classes_[c]}":')
            subList = []
            for f in res[c]:
                print(f'\t"{rfc.features_[f]}": {res[c][f]:.6f}')
                subList.append({"name": rfc.features_[f],"value":res[c][f]})
            featureImportanceList.append({"class":rfc.classes_[c],"details":subList,"sublenght":len(subList)})

    # Adversarial sample
    if imtype == 'adversarial':
        test_file = "silas-temp/uploads/adversial_test.csv"
        print('========== Adversarial Sample ==========')
        print('WARNING: nominal features should be one-hot encoded.')
        x, y = X_test[0], y_test[0]


        print('Generating for x:  ', x)
        print('Original class:    ', muc.predict(x))
        adv = AdversarialSample(muc, verbose=True)
        opt_sample = adv.opt_adv_sample(
            x, y, num_samples=10000, num_itr=100
        )

        adversarialObj['x'] = x

        print('Opt sample:        ', opt_sample)
        print('Distance:          ', adv.distance(x, opt_sample))
        print('Adv sample class:  ', muc.predict(opt_sample))
        adversarialObj = {"x":x,"orgClass":str(muc.predict(x)),"opt_sample":opt_sample.tolist(),"Distance":str(adv.distance(x, opt_sample)),"advSamClass":muc.predict(opt_sample).tolist()}

    
    return make_response(jsonify({'code':20000,'featureImportance':featureImportanceList,"adversarialObj":adversarialObj}), 200)


@app.route('/user/<name>')
def user(name):
  return render_template('treeview.html', name=name)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000


if __name__ == '__main__':
    app.run()

