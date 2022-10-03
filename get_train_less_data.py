import os
import shutil
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


if __name__ == "__main__":
    r_path = r"/mnt/Gu/trainData/plate/TrainOri"
    save_dir = r"/mnt/Gu/trainData/plate/git_train/extra"
    name_str="学使领警"
    name_dict={}
    save_path = r""
    for i_str in name_str:
        name_dict[i_str] =0
    file_list=[]
    index = 0
    allFilePath(r_path,file_list)
    for  pic_path in file_list:
        index+=1
        pic_name=os.path.basename(pic_path)
        plate_name=pic_name.split("_")[0]
        if plate_name[-1] in name_str:
            name_dict[plate_name[-1]]+=1
            # save_folder =os.path.join(save_dir,plate_name[-1])
            # if not os.path.exists(save_folder):
            #     os.mkdir(save_folder)
            # if name_dict[plate_name[-1]]<=100:
            pic_name1=plate_name+"_"+str(index)+".jpg"
            new_pic_path = os.path.join(save_dir,pic_name1)
            shutil.copy(pic_path,new_pic_path)
                
        elif plate_name[0] in name_str:
            name_dict[plate_name[0]]+=1
            # save_folder =os.path.join(save_dir,plate_name[0])
            # if not os.path.exists(save_folder):
            #     os.mkdir(save_folder)
            # if name_dict[plate_name[0]]<=100:
            pic_name1=plate_name+"_"+str(index)+".jpg"
            new_pic_path = os.path.join(save_dir,pic_name1)
            shutil.copy(pic_path,new_pic_path)

    
    print(name_dict)
    