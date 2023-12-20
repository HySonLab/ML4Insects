import os
import preprocessingV1
# # Clean data filename
# path = 'C:\\Users\\My PC\\Desktop\\Data from Daniel\\8zt'
# for filename in os.listdir(path):
#     x = filename
#     split = x.split('_')
#     if split[0] == '8h' or split[0] == '8hr':
#         split.pop(0)
#     newname = ''
#     for n in range(len(split)-1):
#         newname += split[n]
#         newname += '_'
#     newname += split[len(split)-1]

#     os.rename(os.path.join(path,filename), os.path.join(path,newname))

# #Clean analysis filename
# path = 'C:\\Users\\My PC\\Desktop\\Data from Daniel\\16zt_ANA'
# for filename in os.listdir(path):
#     x = filename
#     split = x.split('_')
#     if split[0] == '8h' or split[0] == '8hr':
#         split.pop(0)
#     newname = ''
#     for n in range(len(split)-2):
#         newname += split[n]
#         newname += '_'
#     newname += split[len(split)-2]
#     newname += split[len(split)-1]

#     os.rename(os.path.join(path,filename), os.path.join(path,newname))
# #Verify that all file names are correct
# os.chdir(path+'\\16zt')
# name = os.listdir()
# os.chdir(path+'\\16zt_ANA')
# ana_name = os.listdir()
# filename = []
# for n in name:
#     split = n.split('.')
#     filename.append(split[0])
# filename = np.unique(filename)
# ananame = []
# for n in name:
#     split = n.split('.')
#     ananame.append(split[0])
# ananame = np.unique(ananame)

# for i in range(len(filename)):
#     if filename[i] not in ananame:
#         print('Error')
#         break

def generate_signal_dictionary(data = {},data_test = {},outlier_filter = False, downsampling = False, denoising = False, name_dict = None, name = '0zt'):

    train, test = name_dict[name]

    for i in range(len(train)):

        # Read data table and analysis file
        data[train[i]] = preprocessingV1.read_signal(train[i])

        # preprocessing option
        if outlier_filter == True:
            data[train[i]] = preprocessingV1.outlier_filtering(data[train[i]][0],data[train[i]][1])
        if downsampling == True:
            data[train[i]] = preprocessingV1.downsampling(data[train[i]][0],data[train[i]][1])
        if denoising == True:
            data[train[i]][0] = preprocessingV1.wavelet_denoising(data[train[i]][0],wavelet = 'sym4',n_level = 5)
            
    for i in range(len(test)):

        # Read data table and analysis file
        data_test[test[i]] = preprocessingV1.read_signal(test[i])

        # preprocessing option
        if outlier_filter == True:
            data_test[test[i]] = preprocessingV1.outlier_filtering(data_test[test[i]][0],data_test[test[i]][1])
        if downsampling == True:
            data_test[test[i]] = preprocessingV1.downsampling(data_test[test[i]][0],data_test[test[i]][1])
        if denoising == True:
            data_test[test[i]][0] = preprocessingV1.wavelet_denoising(data_test[test[i]][0],wavelet = 'sym4',n_level = 5)

    return data, data_test