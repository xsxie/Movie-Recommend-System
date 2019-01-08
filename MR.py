# -*- coding: utf-8 -*-

"""
	Movie Recommend System
	Author: Xie xueshuo
	Date: 2019.01.03
"""

import re
import os
import sys
import time
import numpy as np
import scipy.io as sio

#定义读取和保存文件路径
#file_path = os.path.dirname(os.path.realpath(sys.executable))+'\\' #获取可执行程序的当前路径，用于编译为可执行文件时使用
file_path = os.getcwd()+'\\' #获取文件所在的当前路径，运行.py文件时使用
savepath = file_path + 'data\\'#定义文件保存的路径

#声明全局变量
minnum=1e-4   #定义极小常量，防止除零错误
userid_vector_name='userid_vector'
item_score_matrix_name='item_score_matrix'
itemid_matrix_name='itemid_matrix'

'''
    处理train.txt和test.txt文件，生成用户user和item的矩阵，并保存到文件
    设定threshold=100，将用户评分item的个数少于阈值以及评分方差为0的进行过滤，选取有效的用户集合
'''
def process_file_to_matrix(filename='train',filter=('item','std'),threshold=100,process_test_file=False,testfilname='process_test_file'):

    savename=savepath+'threshold_'+str(threshold)+'_filter'+str(filter)+'_'+filename+'.mat'
    testname=savepath+testfilname+'.mat'
    test_item_length=6
    
    if  process_test_file and os.path.isfile(testname) and (os.path.isfile(savename)):
        print('A matrix file has  saved : %s' % (savename))
        print('A matrix file has  saved : %s' % (testname))
        
        train = sio.loadmat(savename)
        test = sio.loadmat(testname)
        
        return [train,test]
    elif (not process_test_file) and (os.path.isfile(savename)):
        print('A matrix file has  saved : %s' % (savename))
        train = sio.loadmat(savename)
        return train
    file = open(file_path + filename+'.txt')
    train = (file.read())
    b = train.split('\n')
    # user对item打分的矩阵，每一行维打分的item
    user_item_rate_vector = []
    # user对item打分的id，每一行为打过分的item的id，对应于上面的打分矩阵
    user_item_rate_id_vector = []
    # userid向量，对应于上面的行号
    userid_vector = []
    #存user对应打分的均值和标准差X*2
    user_mean_variance=[]

    test_user_id_vector=[]
    test_item_id_matrix=[]
    test_item_score_matrix=[]
    print('Processing...')
    
    time_show = 100000
    lens = len(b)
    total_user=0
    for i in range(lens):
        if not (re.match('^[0-9]*\|[0-9]*$', b[i]) == None):
            res = b[i].split('|')
            num = int(res[1])
            user_id = int(res[0])
            user_rate = []
            user_rate_id = []

            test_user_id_vector.append(user_id)
            test_item_score_vector=[]
            test_item_id_vector=[]
            total_user+=1
            for j in range(num):
                if i%time_show==0:
                   print("Processed user_id:%d,\t has %d item score,\t percentage: %.2f"%(user_id,num,100*(i/lens)))
                i += 1
                pair = b[i].split()
                item_id=int(pair[0])
                item_score=float(pair[1])
                if j < test_item_length:
                    test_item_score_vector.append(item_score)
                    test_item_id_vector.append(item_id)
                user_rate.append(item_score)
                user_rate_id.append(item_id)

            #处理test.txt
            test_item_id_matrix.append(test_item_id_vector)
            test_item_score_matrix.append(test_item_score_vector)
            #处理train.txt
            rate_vector = np.asarray(user_rate, np.float32)
            rate_vector_mean=rate_vector.mean()
            rate_vector_variance=rate_vector.std()
            user_mean_variance.append([rate_vector_mean,rate_vector_variance])
            if (filter[0] == 'item' and num < threshold) or (filter[1]=='std' and rate_vector_variance==0):#如果标准差为0，则过滤此用户,num<thre过滤
                continue
            rate_vector = (rate_vector - rate_vector_mean) / (rate_vector_variance + minnum)
            user_item_rate_vector.append(rate_vector)
            user_item_rate_id_vector.append(np.asarray(user_rate_id, np.int32))
            userid_vector.append(user_id)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    user_item_rate_vector = np.asarray(user_item_rate_vector)
    user_item_rate_id_vector = np.asarray(user_item_rate_id_vector)
    userid_vector = np.asarray(userid_vector)
    user_mean_variance = np.asarray(user_mean_variance,np.float32)
    savefile={'user_rate_item_score':user_item_rate_vector,"user_rate_item_id":user_item_rate_id_vector,"userid_vector":np.reshape(userid_vector,[1,len(userid_vector)]),'user_mean_variance':user_mean_variance}
    sio.savemat(savename, savefile)
    print('Saved train_result to file %s' % (savename))
    print('Total user:\t%d,Valid user:\t%d,Saved user:\t%d' % (total_user,total_user-len(userid_vector),len(userid_vector)))

    if process_test_file:
        test_user_id_vector=np.array(test_user_id_vector,np.int32)
        test_item_id_matrix=np.array(test_item_id_matrix,np.int32)
        test_item_score_matrix=np.array(test_item_score_matrix,np.float32)
        test_file={userid_vector_name:np.reshape(test_user_id_vector,[1,len(test_user_id_vector)]),itemid_matrix_name:test_item_id_matrix,item_score_matrix_name:test_item_score_matrix}
        sio.savemat(testname, test_file)
        print('Saved test_result to file %s' % (testname))
        return [savefile,test_file]
    return savefile

'''
    计算用户之间的相似性
    循环计算每一个用户之间的相似度，获取相似得分以及最相似用户列表
    取前k个最相似的，并保存用户相似度矩阵
'''
def calculate_similarity(similarknumber=100):

    savename = savepath + 'k_' + str(similarknumber) + '_usersimilar.mat'
    if os.path.isfile(savename) :
        print('A matrix file has  saved : %s' % (savename))
        file = sio.loadmat(savename)
        return file

    train=process_file_to_matrix(filter=('item','std'))
    user_item_rate_vector=train['user_rate_item_score'][0]
    user_item_rate_id_vector = train['user_rate_item_id'][0]
    userid_vector = train['userid_vector'][0]
    user_length = len(userid_vector)

    k_similar=np.zeros([user_length,similarknumber],np.float32)
    k_user_id = np.zeros([user_length, similarknumber],np.int32)

    showtime=2
    for i in range(user_length):
        if (i+1)%showtime==0:
            time_end = time.time()
            print('Processed  user_id:%d(percentage: %.2f%%(%d/%d)), Spend time:%.4fs'%(userid_vector[i],100*((i+1)/user_length),(i+1),user_length,(time_end-time_start)))
        time_start = time.time()
        index=i
        user_iid=userid_vector[index]
        use_ri=user_item_rate_id_vector[index][0]
        user_irate = user_item_rate_vector[index][0]

        if user_irate.std() == 0:
            continue
        useriset=set(use_ri)

        for j in range(i+1,user_length):
            jndex=j
            userj = user_item_rate_id_vector[jndex][0]
            userjrate = user_item_rate_vector[jndex][0]
            if userjrate.std()==0:
                continue
            userjid = userid_vector[jndex]
            userjset = set(userj)
            cross = useriset & userjset
            similiar=0
            if len(cross)<0:
                continue
            for item in cross:
                indexi=np.where(use_ri==item)[0]
                indexj = np.where(userj == item)[0]
                similiar+=user_irate[indexi]*userjrate[indexj]

            if k_similar[index,similarknumber-1]<similiar:
                k_similar[index, similarknumber - 1]=similiar
                k_user_id[index, similarknumber - 1]=userjid
                sortid=np.argsort(-k_similar[index])
                k_user_id[index]=k_user_id[index,sortid]
                k_similar[index] = k_similar[index, sortid]

            if k_similar[jndex,similarknumber-1]<similiar:
                k_similar[jndex, similarknumber - 1]=similiar
                k_user_id[jndex, similarknumber - 1]=user_iid
                sortid=np.argsort(-k_similar[jndex])
                k_user_id[jndex]=k_similar[jndex,sortid]

    savedict={'userid_vector':np.reshape(userid_vector,[1,len(userid_vector)]),'user_k_similar_score':k_similar,'user_k_similar_user':k_user_id}
    sio.savemat(savename,savedict)
    print('Saved k-similarity to file: %s' % (savename))
    return savedict

'''
    处理test.txt，获取测试文件的score
    user对item打分的id，每一行为打过分item的id，对应于上面的打分矩阵
    userid向量，对应于上面的行号
'''
def process_test_file(filename='test',hasscore=False):

    savename=savepath+filename+'.mat'
    if (os.path.isfile(savename)):
        print('A matrix file has  saved : %s' % (savename))
        train = sio.loadmat(savename)
        return train

    file = open(savepath+'train_test_score.txt')
    test = (file.read())
    b = test.split('\n')
    
    user_item_rate_id_vector = []
    user_item_rate_vector=[]
    
    userid_vector = []
    print('Processing...')
    time_show = 10000
    lens = len(b)
    for i in range(lens):
        if not (re.match('^[0-9]*\|[0-9]*$', b[i]) == None):
            res = b[i].split('|')
            num = int(res[1])
            user_id = int(res[0])
            user_rate_id = []
            user_rate=[]
            
            for j in range(num):
                if i%time_show==0:
                   print("Processed user_id:%d,\t has %d item score,\t percentage: %.2f"%(user_id,num,100*(i/lens)))
                i += 1
                if hasscore:
                    itemattr=b[i].split()
                    user_rate_id.append(int(itemattr[0]))
                    user_rate.append(float(itemattr[1]))
                else:
                    user_rate_id.append(int(b[i]))
            user_item_rate_id_vector.append(np.asarray(user_rate_id, np.int32))
            if hasscore:
                user_item_rate_vector.append(np.asarray(user_rate, np.float32))
            userid_vector.append(user_id)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    user_item_rate_id_vector = np.asarray(user_item_rate_id_vector)
    userid_vector = np.asarray(userid_vector)
    if hasscore:
        user_item_rate_vector=np.asarray(user_item_rate_vector,np.float32)
        savefile = {itemid_matrix_name: user_item_rate_id_vector, userid_vector_name:np.reshape(userid_vector,[1,len(userid_vector)]),item_score_matrix_name:user_item_rate_vector}
    else:
        savefile={itemid_matrix_name:user_item_rate_id_vector,userid_vector_name:np.reshape(userid_vector,[1,len(userid_vector)])}
    sio.savemat(savename, savefile)
    print('Saved result score to file: %s' % (savename))
    return savefile

'''
    计算并预测用户得分
    如果用户已经对其打分，则就是该得分。
    如果没有用户打分但具有相似用户，则取得用户相似的且对item打分且与user相似度大于threshold的N个user的打分按相似度加权求和，
    如果没有相似用户，则计算用户打分平均值
    先查找userid是否在相似度的矩阵里，如果不在则说明其没有相似用户，则直接用其平均分代替，或者用item-item协同
    不存在相似用户,则直接返回该用户打分均值
'''
def prediction_score(filename='prediction_score',test=None):

    savename=savepath+'prediction_score.mat'
    if (os.path.isfile(savename)):
        print('A matrix file has  saved : %s' % (savename))
        train = sio.loadmat(savename)
        return train

    if test==None:
        test=process_test_file()
    
    test_user_id_vector = test[userid_vector_name][0]
    test_user_rate_item_id = test[itemid_matrix_name]
    test_user_rate_matrix=[]
    
    similarity=calculate_similarity()
    user_k_similar_score=similarity['user_k_similar_score']
    user_k_similar_user=similarity['user_k_similar_user']
    
    train=process_file_to_matrix()
    user_item_rate_vector = train['user_rate_item_score'][0]
    user_item_rate_id_vector = train['user_rate_item_id'][0]
    userid_vector = train['userid_vector'][0]
    user_mean_variance = train['user_mean_variance']

    user_length=len(test_user_id_vector)
    showtime=10
    for i in range(user_length):
        rate_vector=[]
        test_item_length=len(test_user_rate_item_id[i])
        user_id = test_user_id_vector[i]
        user_index = (np.where(userid_vector==user_id))[0]
        if len(user_index)==0:
            rate_vector=np.ones(test_item_length)*user_mean_variance[user_id,0]
            test_user_rate_matrix.append(rate_vector)
            continue
        
        k_similar_user=user_k_similar_user[user_index[0]]
        k_similar_score=user_k_similar_score[user_index[0]]
        
        for item_id in test_user_rate_item_id[i]:
            N=0
            S=[]
            W=[]
            
            for j in range(len(k_similar_user)):
                if k_similar_score[j]==0:
                    break
                userj = k_similar_user[j]
                userjndex = (np.where(userid_vector == userj))[0]
                if len(userjndex)==0:
                    print(k_similar_user,userid_vector)
                userjitemvecter=user_item_rate_id_vector[userjndex[0]][0]
                jndexitem=(np.where(userjitemvecter == item_id))[0]
                if len(jndexitem)==0:
                    continue
                scoreobj=user_item_rate_vector[userjndex[0]][0]
                score=scoreobj[jndexitem[0]]
                weight=k_similar_score[j]
                S.append(score)
                W.append(weight)

            if len(S)==0:
                rate_vector.append(0)
                continue
            s=np.array(S)
            w=np.array(W)
            calcscore=(s*w).sum()/w.sum()
            rate_vector.append(calcscore)

        rate_vector=np.array(rate_vector)
        rate_vector=rate_vector*user_mean_variance[user_id,1] +user_mean_variance[user_id,0]
        rate_vector[rate_vector<0]=0
        test_user_rate_matrix.append(rate_vector)
        if i%(showtime)==0:
            print('Calculate user:%d\'s rate,rate vector is:\n'%(user_id),rate_vector)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savefile = {userid_vector_name:np.reshape(test_user_id_vector,[1,len(test_user_id_vector)]),itemid_matrix_name:test_user_rate_item_id,item_score_matrix_name: np.array(test_user_rate_matrix)}
    sio.savemat(savename, savefile)
    print('Saved prediction score to file: %s' % (savename))
    return savefile

'''
    保存中间结果为文本文件，如果没有打分矩阵则输出测试文本
'''
def save_file_txt(userid_vector,itemid_matrix,item_score_matrix,filename):

    f=open(savepath+filename,'w')
    format='  '
    user_length=len(userid_vector)
    for i in range(user_length):
        user_id=userid_vector[i]
        item_length=len(itemid_matrix[i])
        f.write(str(user_id)+'|'+str(item_length)+'\n')
        for j in range(item_length):
            item_id=itemid_matrix[i][j]
            if item_score_matrix.any()==None:
                f.write(str(item_id)+'\n')
                continue
            item_score=item_score_matrix[i][j]
            f.write(str(item_id)+format+str(item_score)+format+'\n')
    f.close()

'''
    计算均方根误差
'''
def calculate_RMSE(mat1,mat2):

    RMSE=np.sqrt(((mat1-mat2)**2).mean())
    return RMSE
'''
def main():

    start = time.time()
    train,test=process_file_to_matrix(process_test_file=True,testfilname='train_test_matrix')
    pre=prediction_score(filename='train_test_score',test=test)
    save_file_txt(pre[userid_vector_name][0],pre[itemid_matrix_name],pre[item_score_matrix_name],'prediction_score.txt')
    save_file_txt(test[userid_vector_name][0],test[itemid_matrix_name],test[item_score_matrix_name],'train_test_score.txt')
    mat1=process_test_file(filename='train_test_score',hasscore=True)
    mat2=process_test_file(filename='prediction_score',hasscore=True)
    RMSE=calculate_RMSE(mat1[item_score_matrix_name],mat2[item_score_matrix_name])
    print('RMSE:%.4f'%(RMSE))
    spend_time=time.time()-start
    print('Total time is %.4fmin(%.4fs)'%(spend_time/60,spend_time))
'''
if __name__ == '__main__':

    #main()
    start_time = time.time()
    train_matrix,test_matrix = process_file_to_matrix(process_test_file=True, testfilname='train_test_matrix')
    prediction_score_result = prediction_score(filename='train_test_score', test=test_matrix)
    save_file_txt(prediction_score_result[userid_vector_name][0], prediction_score_result[itemid_matrix_name], prediction_score_result[item_score_matrix_name], 'prediction_score.txt')
    save_file_txt(test_matrix[userid_vector_name][0], test_matrix[itemid_matrix_name], test_matrix[item_score_matrix_name], 'train_test_score.txt')
    train_test_matrix = process_test_file(filename='train_test_score', hasscore=True)
    prediction_score_matrix = process_test_file(filename='prediction_score', hasscore=True)
    RMSE = calculate_RMSE(train_test_matrix[item_score_matrix_name], prediction_score_matrix[item_score_matrix_name])
    print('RMSE:%.4f' % (RMSE))
    spend_time = time.time() - start_time
    print('Total time is %.4fmin(%.4fs)' % (spend_time / 60, spend_time))