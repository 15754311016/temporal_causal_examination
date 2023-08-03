import numpy as np
def calc_volume(diameter):
    return 4 / 3 * np.pi * (diameter / 2) ** 3
norm_const = calc_volume(13)
def one_step_pehe(path):

    # 加载.npy文件
    data = np.load(path)
    # 打印加载的数据
    print(data.files)
    predict_outcomes = data['means']
    ground_outcomes = data['output']
    activate_entries = data['active_entries']
    treatments = data['treatments']
    num_samples, time_dim, output_dim = activate_entries.shape
    last_entries = activate_entries - np.concatenate([activate_entries[:, 1:, :], np.zeros((num_samples, 1, output_dim))], axis=1)
    last_entries.shape
    treatment_last_step = np.zeros((num_samples,4))
    index = np.array([np.nonzero(row)[0] for row in last_entries.squeeze()]).squeeze()
    last_predict_outcomes = np.zeros((num_samples,1))
    last_ground_outcomes =np.zeros((num_samples,1))
    for i in range(treatment_last_step.shape[0]): 
        treatment_last_step[i, :] = treatments[i, index[i], :]
        last_predict_outcomes[i,:] = predict_outcomes[i, index[i], :]
        last_ground_outcomes[i,:] = ground_outcomes[i, index[i], :]
    # 得到不同treatment combination的索引
    indices_1000 = np.where(np.all(treatment_last_step == [1, 0, 0, 0], axis=1))[0]
    indices_0100 = np.where(np.all(treatment_last_step == [0, 1, 0, 0], axis=1))[0]
    indices_0010 = np.where(np.all(treatment_last_step == [0, 0, 1, 0], axis=1))[0]
    indices_0001 = np.where(np.all(treatment_last_step == [0, 0, 0, 1], axis=1))[0]
    # indices_1000 = np.where(np.all(treatment_last_step == [0, 0], axis=1))[0]
    # indices_0100 = np.where(np.all(treatment_last_step == [1, 0], axis=1))[0]
    # indices_0010 = np.where(np.all(treatment_last_step == [0, 1], axis=1))[0]
    # indices_0001 = np.where(np.all(treatment_last_step == [1, 1], axis=1))[0]
    # print(indices)
    print(len(indices_0001))
    print(len(indices_0010))
    print(len(indices_0100))
    print(len(indices_1000))
    # 得到不同treatment combination下的outcome
    last_predict_outcomes_1000 = last_predict_outcomes[indices_1000, :]
    last_predict_outcomes_0100 = last_predict_outcomes[indices_0100, :]
    last_predict_outcomes_0010 = last_predict_outcomes[indices_0010, :]
    last_predict_outcomes_0001 = last_predict_outcomes[indices_0001, :]
    # ground outcomes
    last_ground_outcomes_1000 = last_ground_outcomes[indices_1000, :]
    last_ground_outcomes_0100 = last_ground_outcomes[indices_0100, :]
    last_ground_outcomes_0010 = last_ground_outcomes[indices_0010, :]
    last_ground_outcomes_0001 = last_ground_outcomes[indices_0001, :]
    # error pehe for 0100
    pehe_0100 = np.sqrt(np.mean(((last_predict_outcomes_0100-last_predict_outcomes_1000)-(last_ground_outcomes_0100-last_ground_outcomes_1000))**2))
    # error pehe for 0010
    pehe_0010 = np.sqrt(np.mean(((last_predict_outcomes_0010-last_predict_outcomes_1000)-(last_ground_outcomes_0010-last_ground_outcomes_1000))**2))
    # error pehe for 0001
    pehe_0001 = np.sqrt(np.mean(((last_predict_outcomes_0001-last_predict_outcomes_1000)-(last_ground_outcomes_0001-last_ground_outcomes_1000))**2))
    pehe = (pehe_0001+pehe_0010+pehe_0100)/3/norm_const*100
    return pehe


def one_step_pehe_rmsn(path):

    # 加载.npy文件
    data = np.load(path)
    # 打印加载的数据
    print(data.files)
    predict_outcomes = data['means']
    ground_outcomes = data['output']
    activate_entries = data['active_entries']
    treatments = data['treatments']
    num_samples, time_dim, output_dim = activate_entries.shape
    last_entries = activate_entries - np.concatenate([activate_entries[:, 1:, :], np.zeros((num_samples, 1, output_dim))], axis=1)
    last_entries.shape
    treatment_last_step = np.zeros((num_samples,2))
    index = np.array([np.nonzero(row)[0] for row in last_entries.squeeze()]).squeeze()
    last_predict_outcomes = np.zeros((num_samples,1))
    last_ground_outcomes =np.zeros((num_samples,1))
    for i in range(treatment_last_step.shape[0]): 
        treatment_last_step[i, :] = treatments[i, index[i], :]
        last_predict_outcomes[i,:] = predict_outcomes[i, index[i], :]
        last_ground_outcomes[i,:] = ground_outcomes[i, index[i], :]
    # 得到不同treatment combination的索引
    # indices_1000 = np.where(np.all(treatment_last_step == [1, 0, 0, 0], axis=1))[0]
    # indices_0100 = np.where(np.all(treatment_last_step == [0, 1, 0, 0], axis=1))[0]
    # indices_0010 = np.where(np.all(treatment_last_step == [0, 0, 1, 0], axis=1))[0]
    # indices_0001 = np.where(np.all(treatment_last_step == [0, 0, 0, 1], axis=1))[0]
    indices_1000 = np.where(np.all(treatment_last_step == [0, 0], axis=1))[0]
    indices_0100 = np.where(np.all(treatment_last_step == [1, 0], axis=1))[0]
    indices_0010 = np.where(np.all(treatment_last_step == [0, 1], axis=1))[0]
    indices_0001 = np.where(np.all(treatment_last_step == [1, 1], axis=1))[0]
    # print(indices)
    print(len(indices_0001))
    print(len(indices_0010))
    print(len(indices_0100))
    print(len(indices_1000))
    # 得到不同treatment combination下的outcome
    last_predict_outcomes_1000 = last_predict_outcomes[indices_1000, :]
    last_predict_outcomes_0100 = last_predict_outcomes[indices_0100, :]
    last_predict_outcomes_0010 = last_predict_outcomes[indices_0010, :]
    last_predict_outcomes_0001 = last_predict_outcomes[indices_0001, :]
    # ground outcomes
    last_ground_outcomes_1000 = last_ground_outcomes[indices_1000, :]
    last_ground_outcomes_0100 = last_ground_outcomes[indices_0100, :]
    last_ground_outcomes_0010 = last_ground_outcomes[indices_0010, :]
    last_ground_outcomes_0001 = last_ground_outcomes[indices_0001, :]
    # error pehe for 0100
    pehe_0100 = np.sqrt(np.mean(((last_predict_outcomes_0100-last_predict_outcomes_1000)-(last_ground_outcomes_0100-last_ground_outcomes_1000))**2))
    # error pehe for 0010
    pehe_0010 = np.sqrt(np.mean(((last_predict_outcomes_0010-last_predict_outcomes_1000)-(last_ground_outcomes_0010-last_ground_outcomes_1000))**2))
    # error pehe for 0001
    pehe_0001 = np.sqrt(np.mean(((last_predict_outcomes_0001-last_predict_outcomes_1000)-(last_ground_outcomes_0001-last_ground_outcomes_1000))**2))
    pehe = (pehe_0001+pehe_0010+pehe_0100)/3/norm_const*100
    return pehe

def treatment_timing_accuracy(path):
    data = np.load(path+'/predictions_multi_steps.npz')
    treatment_seq = np.load(path+'/test_cf_treatment_seq.npz')
    # print(treatment_seq.files)
    patient_i = treatment_seq['patient_ids_all_trajectories']
    patient_t=treatment_seq['patient_current_t']
    # for i in range(59):
        # print(np.count_nonzero(patient_t==i))
    # for i in range(1000):
    #     print(np.count_nonzero(patient_i==i))
    i_t = np.concatenate([patient_i.reshape(-1,1),patient_t.reshape(-1,1)],axis=1)
    index_per_patients = [-1]
    for i in range(i_t.shape[0]-1):
        if np.all(i_t[i] == i_t[i+1]):
            None
        else:
            index_per_patients.append(i)
    index_per_patients.append(len(patient_i)-1)
    index_per_patients=list(np.array(index_per_patients)+1)
    # print(index_per_patients)
    predict_outcomes = data['means'].squeeze()
    ground_outcomes = data['output'].squeeze()
    activate_entries = data['active_entries'].squeeze()
    treatments = data['treatments']
    # print(predict_outcomes.shape)
    # print(ground_outcomes.shape)
    # print(activate_entries.shape)
    # print(treatments.shape)
    # treatments[:10,:,:]
    timing_predict_all = []
    timing_ground_all = []
    for i in range(len(index_per_patients)-1):
        predict_outcome_seq_last = predict_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(predict_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        timing_predict=np.where(~np.all(target_treatments == [1, 0, 0, 0], axis=1))[0][0]
        timing_predict_all.append(timing_predict)

        ground_outcome_seq_last = ground_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(ground_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        timing_ground = np.where(~np.all(target_treatments == [1, 0, 0, 0], axis=1))[0][0]
        timing_ground_all.append(timing_ground)
    # print(timing_predict_all)
    # print(timing_ground_all)

    # print(np.argmin(predict_outcomes[:10,4]))
    # print(treatments[:10])
    predict_outcomes[:10]
    # print(ground_outcomes)
    timing_ground_all = np.array(timing_ground_all)
    timing_predict_all = np.array(timing_predict_all)
    acc = (timing_ground_all==timing_predict_all).sum()/len(timing_ground_all)
    print(acc)
    return acc


def treatment_timing_accuracy_rmsn(path):
    data = np.load(path+'/predictions_multi_steps.npz')
    treatment_seq = np.load(path+'/test_cf_treatment_seq.npz')
    # print(treatment_seq.files)
    patient_i = treatment_seq['patient_ids_all_trajectories']
    patient_t=treatment_seq['patient_current_t']
    # for i in range(59):
        # print(np.count_nonzero(patient_t==i))
    # for i in range(1000):
    #     print(np.count_nonzero(patient_i==i))
    i_t = np.concatenate([patient_i.reshape(-1,1),patient_t.reshape(-1,1)],axis=1)
    index_per_patients = [-1]
    for i in range(i_t.shape[0]-1):
        if np.all(i_t[i] == i_t[i+1]):
            None
        else:
            index_per_patients.append(i)
    index_per_patients.append(len(patient_i)-1)
    index_per_patients=list(np.array(index_per_patients)+1)
    # print(index_per_patients)
    predict_outcomes = data['means'].squeeze()
    ground_outcomes = data['output'].squeeze()
    activate_entries = data['active_entries'].squeeze()
    treatments = data['treatments']
    # print(predict_outcomes.shape)
    # print(ground_outcomes.shape)
    # print(activate_entries.shape)
    # print(treatments.shape)
    # treatments[:10,:,:]
    timing_predict_all = []
    timing_ground_all = []
    for i in range(len(index_per_patients)-1):
        predict_outcome_seq_last = predict_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(predict_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        timing_predict=np.where(~np.all(target_treatments == [0, 0], axis=1))[0][0]
        timing_predict_all.append(timing_predict)

        ground_outcome_seq_last = ground_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(ground_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        timing_ground = np.where(~np.all(target_treatments == [0, 0], axis=1))[0][0]
        timing_ground_all.append(timing_ground)
    # print(timing_predict_all)
    # print(timing_ground_all)

    # print(np.argmin(predict_outcomes[:10,4]))
    # print(treatments[:10])
    predict_outcomes[:10]
    # print(ground_outcomes)
    timing_ground_all = np.array(timing_ground_all)
    timing_predict_all = np.array(timing_predict_all)
    acc = (timing_ground_all==timing_predict_all).sum()/len(timing_ground_all)
    print(acc)
    return acc

def treatment_type_accuracy(path):
    data = np.load(path+'/predictions_multi_steps.npz')
    treatment_seq = np.load(path+'/test_cf_treatment_seq.npz')
    # print(treatment_seq.files)
    patient_i = treatment_seq['patient_ids_all_trajectories']
    patient_t=treatment_seq['patient_current_t']
    # for i in range(59):
        # print(np.count_nonzero(patient_t==i))
    # for i in range(1000):
    #     print(np.count_nonzero(patient_i==i))
    i_t = np.concatenate([patient_i.reshape(-1,1),patient_t.reshape(-1,1)],axis=1)
    index_per_patients = [-1]
    for i in range(i_t.shape[0]-1):
        if np.all(i_t[i] == i_t[i+1]):
            None
        else:
            index_per_patients.append(i)
    index_per_patients.append(len(patient_i)-1)
    index_per_patients=list(np.array(index_per_patients)+1)
    # print(index_per_patients)
    predict_outcomes = data['means'].squeeze()
    ground_outcomes = data['output'].squeeze()
    activate_entries = data['active_entries'].squeeze()
    treatments = data['treatments']
    # print(predict_outcomes.shape)
    # print(ground_outcomes.shape)
    # print(activate_entries.shape)
    # print(treatments.shape)
    # treatments[:10,:,:]
    type_predict_all = []
    type_ground_all = []
    for i in range(len(index_per_patients)-1):
        predict_outcome_seq_last = predict_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(predict_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        # if i==1:
        #     print(target_treatments)
        type_predict_idx=np.where(~np.all(target_treatments == [1, 0, 0, 0], axis=1))[0][0]
        if np.all(target_treatments[type_predict_idx]==[0,1,0,0]):
            type_predict_all.append(0)
        elif np.all(target_treatments[type_predict_idx]==[0,0,1,0]):
            type_predict_all.append(1)


        ground_outcome_seq_last = ground_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(ground_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        # if i==1:
        #     print(target_treatments)
        type_ground_idx = np.where(~np.all(target_treatments == [1, 0, 0, 0], axis=1))[0][0]
        if np.all(target_treatments[type_ground_idx]==[0,1,0,0]):
            type_ground_all.append(0)
        else:
            type_ground_all.append(1)
    # print(type_predict_all)
    # print(type_ground_all)
    type_ground_all = np.array(type_ground_all)
    type_predict_all = np.array(type_predict_all)
    acc = (type_ground_all==type_predict_all).sum()/len(type_ground_all)
    print(acc)
    return acc

def treatment_type_accuracy_rmsn(path):
    data = np.load(path+'/predictions_multi_steps.npz')
    treatment_seq = np.load(path+'/test_cf_treatment_seq.npz')
    # print(treatment_seq.files)
    patient_i = treatment_seq['patient_ids_all_trajectories']
    patient_t=treatment_seq['patient_current_t']
    # for i in range(59):
        # print(np.count_nonzero(patient_t==i))
    # for i in range(1000):
    #     print(np.count_nonzero(patient_i==i))
    i_t = np.concatenate([patient_i.reshape(-1,1),patient_t.reshape(-1,1)],axis=1)
    index_per_patients = [-1]
    for i in range(i_t.shape[0]-1):
        if np.all(i_t[i] == i_t[i+1]):
            None
        else:
            index_per_patients.append(i)
    index_per_patients.append(len(patient_i)-1)
    index_per_patients=list(np.array(index_per_patients)+1)
    # print(index_per_patients)
    predict_outcomes = data['means'].squeeze()
    ground_outcomes = data['output'].squeeze()
    activate_entries = data['active_entries'].squeeze()
    treatments = data['treatments']
    # print(predict_outcomes.shape)
    # print(ground_outcomes.shape)
    # print(activate_entries.shape)
    # print(treatments.shape)
    # treatments[:10,:,:]
    type_predict_all = []
    type_ground_all = []
    for i in range(len(index_per_patients)-1):
        predict_outcome_seq_last = predict_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(predict_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        # if i==1:
        #     print(target_treatments)
        type_predict_idx=np.where(~np.all(target_treatments == [ 0, 0], axis=1))[0][0]
        if np.all(target_treatments[type_predict_idx]==[0,1]):
            type_predict_all.append(0)
        elif np.all(target_treatments[type_predict_idx]==[1,0]):
            type_predict_all.append(1)


        ground_outcome_seq_last = ground_outcomes[index_per_patients[i]:index_per_patients[i+1],:][:,4]
        idx_seq = np.argmin(ground_outcome_seq_last)
        target_treatments = treatments[index_per_patients[i]+idx_seq,:,:]
        # if i==1:
        #     print(target_treatments)
        type_ground_idx = np.where(~np.all(target_treatments == [0,0], axis=1))[0][0]
        if np.all(target_treatments[type_ground_idx]==[0,1]):
            type_ground_all.append(0)
        else:
            type_ground_all.append(1)
    # print(type_predict_all)
    # print(type_ground_all)
    type_ground_all = np.array(type_ground_all)
    type_predict_all = np.array(type_predict_all)
    acc = (type_ground_all==type_predict_all).sum()/len(type_ground_all)
    print(acc)
    return acc

def one_step_rmse():
    ct_coff_0 = np.array([0.8121, 0.7507, 0.7086, 0.8000, 0.8358])
    ct_coff_1 = np.array([0.7910, 0.7404, 0.7416, 0.8743, 0.8919])
    ct_coff_2 = np.array([0.8065, 0.8347, 0.8049, 0.8822, 0.9750])
    ct_coff_3 = np.array([0.9462, 1.0169, 0.9223, 1.0152, 1.2651])
    ct_mean = np.concatenate([ct_coff_0,ct_coff_1,ct_coff_2,ct_coff_3]).reshape(4,5).mean(axis=1)
    ct_std = np.concatenate([ct_coff_0,ct_coff_1,ct_coff_2,ct_coff_3]).reshape(4,5).std(axis=1)

    crn_coff_0 = np.array([0.7820, 0.7412, 0.7217, 0.7870, 0.8387])
    crn_coff_1 = np.array([0.7697, 0.7804, 0.7507, 0.7797, 0.8580])
    crn_coff_2 = np.array([0.8016, 1.0087, 0.7686, 0.9599, 0.9396])
    crn_coff_3 = np.array([0.9767, 1.0607, 1.0253, 0.9981, 1.2194])
    crn_mean = np.concatenate([crn_coff_0,crn_coff_1,crn_coff_2,crn_coff_3]).reshape(4,5).mean(axis=1)
    crn_std = np.concatenate([crn_coff_0,crn_coff_1,crn_coff_2,crn_coff_3]).reshape(4,5).std(axis=1)

    gnet_coff_0 = np.array([0.8197, 0.7706, 0.9135, 0.8243, 0.8303])
    gnet_coff_1 = np.array([0.8809, 0.7542, 0.9160, 0.8308, 0.9536])
    gnet_coff_2 = np.array([0.9747, 0.9230, 1.0835, 0.9926, 1.0334])
    gnet_coff_3 = np.array([1.0975, 1.1941, 1.1013, 1.3819, 1.8948])
    gnet_mean = np.concatenate([gnet_coff_0,gnet_coff_1,gnet_coff_2,gnet_coff_3]).reshape(4,5).mean(axis=1)
    gnet_std = np.concatenate([gnet_coff_0,gnet_coff_1,gnet_coff_2,gnet_coff_3]).reshape(4,5).std(axis=1)

    edct_coff_0 = np.array([0.7943, 0.7692, 0.6953, 0.7572, 0.8411])
    edct_coff_1 = np.array([0.7778, 0.6942, 0.7246, 0.7935, 0.8730])
    edct_coff_2 = np.array([0.7992, 0.8520, 0.7690, 0.8738, 0.9079])
    edct_coff_3 = np.array([0.9603, 1.0571, 0.9678, 1.0031, 1.2796])
    edct_mean = np.concatenate([edct_coff_0,edct_coff_1,edct_coff_2,edct_coff_3]).reshape(4,5).mean(axis=1)
    edct_std = np.concatenate([edct_coff_0,edct_coff_1,edct_coff_2,edct_coff_3]).reshape(4,5).std(axis=1)

    rmsn_coff_0 = np.array([1.1534, 0.8781, 0.8660, 0.9773, 0.9508])
    rmsn_coff_1 = np.array([1.2964, 1.1676, 0.9542, 1.0412, 1.1184])
    rmsn_coff_2 = np.array([1.1246, 1.0428, 1.0332, 1.0578, 1.3378])
    rmsn_coff_3 = np.array([1.1838, 1.2766, 1.2373, 1.2207, 1.4106])
    rmsn_mean = np.concatenate([rmsn_coff_0,rmsn_coff_1,rmsn_coff_2,rmsn_coff_3]).reshape(4,5).mean(axis=1)
    rmsn_std = np.concatenate([rmsn_coff_0,rmsn_coff_1,rmsn_coff_2,rmsn_coff_3]).reshape(4,5).std(axis=1)
    for i in range(4):
        print("mean and std:", str(edct_mean[i].round(4))+"+-"+str(edct_std[i].round(4)))
# one_step_rmse()
def multi_steps_mse():
    num = []
    for i in range(5):
        # file = open('/home/qianghuang/CausalTransformer-main/multirun/2023-07-16/19-33-35/'+str(i)+'/train_enc_dec.log','r')
        file = open('/home/qianghuang/CausalTransformer-main/multirun/2023-07-16/19-30-18/'+str(i)+'/train_multi.log','r')
        # file = open('/home/qianghuang/CausalTransformer-main/multirun/2023-07-15/12-46-14/'+str(i)+'/train_rmsn.log','r')
        # file = open('/home/qianghuang/CausalTransformer-main/multirun/2023-07-13/01-53-59/'+str(i)+'/train_gnet.log','r')
        line = file.readlines()[-1].strip()
        index = line.find("{'2-step':")
        dict_num = line[index:]
        my_dict = eval(dict_num)
        dict_values = my_dict.values()
        # print(my_dict)
        # print(list(dict_values))
        num.extend(dict_values)
    num = np.array(num).reshape(-1,5,5)
    num = np.transpose(num,axes=(0, 2, 1))
    mean_values = np.mean(num, axis=2)
    std_values = np.std(num, axis=2)
    print(mean_values)
    print(std_values)
    # gamma = [0,1,2,3,6,8,10]
    gamma = [0]
    # gamma = [6,8,10]
    for i,v in enumerate(gamma):
        print()
        for j in range(0,5):
            print("gamma "+str(v)+", step "+str(j+2)+" mean and std: "+ str(mean_values[i][j].round(4))+"+-"+str(std_values[i][j].round(4)))

# multi_steps_mse()
    
def timing_acc_mean():
    acc = []
    for i in range(0,35):
        treat_timing_acc = treatment_timing_accuracy("/home/qianghuang/CausalTransformer-main/multirun/2023-06-19/23-51-11/"+str(i))
        acc.append(treat_timing_acc)
    print("=============")
    acc = np.array(acc).reshape(-1,5)
    mean = acc.mean(1)
    std = acc.std(1)
    # gamma = [6,8,10]
    gamma = [0,1,2,3,6,8,10]
    for i,v in enumerate(gamma):
        print("gamma "+str(v)+" mean and std: "+str(mean[i].round(4))+"+-"+str(std[i].round(4)))
    # print("mean and std:", np.array([acc]).mean(), "+-", np.array([acc]).std())
# timing_acc_mean()

def type_acc_mean():
    acc = []
    for i in range(0,35):
        treat_type_acc = treatment_type_accuracy("/home/qianghuang/CausalTransformer-main/multirun/2023-06-19/23-51-11/"+str(i))
        acc.append(treat_type_acc)
    print("=============")
    acc = np.array(acc).reshape(-1,5)
    mean = acc.mean(1)
    std = acc.std(1)
    # gamma = [6,8,10]
    gamma = [0,1,2,3,6,8,10]
    for i,v in enumerate(gamma):
        print("gamma "+str(v)+" mean and std: "+str(mean[i].round(4))+"+-"+str(std[i].round(4)))
    # print("mean and std:", np.array([acc]).mean(), "+-", np.array([acc]).std())


# type_acc_mean()
def pehe_mean():
    pehe = []
    for i in range(35):
        # pehe_each = one_step_pehe("/home/qianghuang/CausalTransformer-main/multirun/2023-06-15/20-56-45/"+str(i)+"/predictions_encoder.npz")
        pehe_each = one_step_pehe("/home/qianghuang/CausalTransformer-main/multirun/2023-06-19/23-51-11/"+str(i)+"/predictions_encoder.npz")
        # pehe_each = one_step_pehe("/drive2/qianghuang/2023-06-08/00-58-16/"+str(i)+"/predictions_encoder.npz")
        print(pehe_each)
        pehe.append(pehe_each)
    print("=============")
    pehe = np.array(pehe).reshape(-1,5)
    mean = pehe.mean(1)
    std = pehe.std(1)
    # gamma = [6,8,10]
    gamma = [0,1,2,3,6,8,10]
    for i,v in enumerate(gamma):
        print("gamma "+str(v)+" mean and std: "+str(mean[i].round(4))+"+-"+str(std[i].round(4)))

# pehe_mean()

def one_step_rmse_read_log(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    target_string = "Test normalised RMSE (only counterfactual): "
    # target_string = "Test normalised RMSE (all): "
    matching_lines = []

    for index, line in enumerate(lines):
        if target_string in line:
            matching_lines.append((index, line))

    for index, line in matching_lines:
        start_index = line.index(target_string)
        end_index = start_index + len(target_string)
        # print(f"Line index: {index}, String position: {start_index}-{end_index}, Line content: {line}")
    return float(matching_lines[0][1][end_index:])

def one_step_rmse_read_log_mimic(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    # target_string = "Test normalised RMSE (only counterfactual): "
    target_string = "Test normalised RMSE (all): "
    matching_lines = []

    for index, line in enumerate(lines):
        if target_string in line:
            matching_lines.append((index, line))

    for index, line in matching_lines:
        start_index = line.index(target_string)
        end_index = start_index + len(target_string)
        # print(f"Line index: {index}, String position: {start_index}-{end_index}, Line content: {line}")
    return float(matching_lines[0][1][end_index:end_index+10])

def one_step_rmse_mean():
    one_mse_all = []
    for i in range(5):
        # print(i)
        mse = one_step_rmse_read_log_mimic('/home/qianghuang/CausalTransformer-main/multirun/2023-07-16/19-30-18/'+str(i)+'/train_multi.log')
        # mse = one_step_rmse_read_log('/home/qianghuang/CausalTransformer-main/multirun/2023-07-13/01-53-59/'+str(i)+'/train_gnet.log')
        # mse = one_step_rmse_read_log_mimic('/home/qianghuang/CausalTransformer-main/multirun/2023-07-15/12-46-14/'+str(i)+'/train_rmsn.log')
        # mse = one_step_rmse_read_log_mimic('/home/qianghuang/CausalTransformer-main/multirun/2023-07-16/19-33-35/'+str(i)+'/train_enc_dec.log')
        one_mse_all.append(mse)
    mse = np.array(one_mse_all).reshape(-1,5)
    mean = mse.mean(1)
    std = mse.std(1)
    # gamma = [0,1,2,3,6,8,10]
    # gamma = [0,1,2,3]
    # gamma = [6,8,10]
    gamma = [0]
    for i,v in enumerate(gamma):
        print("gamma "+str(v)+" mean and std: "+str(mean[i].round(4))+"+-"+str(std[i].round(4)))

# pehe_mean()
# pehe_mean()

# pehe_mean()
# timing_acc_mean()
# one_step_rmse_mean()
# multi_steps_mse()
one_step_rmse_mean()
# type_acc_mean()
    
    
