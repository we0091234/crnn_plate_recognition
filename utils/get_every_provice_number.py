import os
import shutil
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

if __name__=="__main__":
    palteStr=r"京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民"
    file_path =r"/mnt/Gu/trainData/plate/new_git_train/CCPD_CRPD_OTHER/"
    file_list=[]
    pro_dict={}
    for province in palteStr:
        pro_dict[province]=0
    # print(pro_dict)
    save_folder="save_train"
    allFilePath(file_path,file_list)
    index=0
    error=0
    for file in file_list:
        index+=1
        try:
            plate_name=os.path.basename(file).split("_")[0]
            pro_dict.get(plate_name[-1],0)
            if plate_name[-1] in ["警","港","澳","学","挂","领"]:
                pro_dict[plate_name[-1]]+=1
            else:
                pro_dict[plate_name[0]]+=1
            
        except:
            error+=1
            # print(plate_name,"error")
    a = sorted(pro_dict.items(), key=lambda x: x[1],reverse=True)
    for key in a:
        print(key)