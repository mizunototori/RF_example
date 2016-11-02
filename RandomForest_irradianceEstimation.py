def RandomForest_irradianceEstimation(pointname, load_files, days, test_idx, N_est=30, from_idx=0, end_idx=-1):
    # Parameters:
    #   pointname : string
    #       the name of objective point
    #
    #   load_files : list of strings
    #       the list of load file names
    #   
    #   from_idx : integer
    #       the index of begging to read
    #   
    #   days : list of integer
    #       the  list of days for comparison
    #
    #   test_idx : list of integer
    #       the indeces
    #
    #   N_est : integer
    #       the number of estimater of Random Forest Regressor
    #        

    import numpy as np
    import matplotlib.pyplot as plt
    from io import StringIO
    from sklearn import tree
    from sklearn.preprocessing import Imputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import KFold
    from datetime import datetime as dt
    # import seaborn as sns
    from IPython.core.display import Image
    

    def create_traindata(given_data, test_idx, N, from_idx, end_idx):
        data = [separate_list(d,test_idx[0], test_idx[1], from_idx, end_idx) for d in given_data] 
        data_test = [d[0] for d in data] 
        data_train = [d[1] for d in data]

        true_data = data_test[0]

        train_label = data_train[0]
        train_data = [data_train[i] for i in range(1,N)]
        test_data = [data_test[i] for i in range(1,N)]    
        return train_data, train_label, test_data, true_data

    def comp_plot(dataset, days, title='Title'):
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        fs = 20;
        N = len(dataset[0]['true_data'])/8;
        for i in range(0,len(test_idx)):        
            plt.subplot(2,2,i+1)
            plt.title(title, fontsize=fs)
            plt.ylim(0,1400)
    #        plt.xlim(0,1400)
            plt.tick_params(labelsize=fs)
            plt.hold(True);
            plt.plot(dataset[i]['true_data'],label="Ground obs. H",lw=2, color='lightgray')
            plt.plot(dataset[i]['estimated'], '--', label="Estimated H", lw=2, color='black')
            plt.grid(True)
            plt.title(title+'  2015/4/'+str(days[i]), fontsize=fs)
            plt.xticks([N*0,N*1,N*2,N*3,N*4,N*5,N*6,N*7 ], [0,3,6,9,12,15,18,21,0])
            plt.legend(fontsize=fs-2,frameon=True,loc='upper left')
            plt.ylabel(r'Solar irradiance [$kWm^{-2}$]', fontsize=fs)
            plt.xlabel('Local time (UTC+9)', fontsize=fs)
        plt.savefig(title+'_RF_patterns.png', dpi=250)    
        
        
    def comp_scatter(true_data, estimated_data, title='Title'):
        plt.figure(figsize=(10, 9))
        plt.grid()
        plt.title(title, fontsize=30)
        plt.ylim(0,1200)
        plt.xlim(0,1200)
        plt.tick_params(labelsize=20)
        t_est = np.array([1,1200]); t_true = np.array([1,1200]);
        plt.scatter(true_data, estimated_data, alpha=0.7, s=15, color='black')
        plt.plot(t_est,t_true, color='black');
        plt.ylabel(r'Ground measured solar irradiance [$kWm^{-2}$]', fontsize=30)
        plt.xlabel(r'Estimated solar irradiance [$kWm^{-2}$]', fontsize=30)
        plt.title(title, fontsize=30)
        plt.text(100,1100,r'r=%f '%(np.corrcoef((true_data, estimated_data))[0,1]) , fontsize=20)
        np.correlate(t_est,t_true)
        plt.savefig(title+"_scatter.png", dpi=250)
        
    def separate_list(arr, start, end,from_idx, end_idx):
        if type(arr) is np.ndarray:
            arr_ext = arr[start:end]
            arr_res = np.r_[arr[from_idx:start], arr[end:end_idx]]
        elif type(arr) is list:
            arr_ext = arr[start:end]    
            arr_res = arr[from_idx:start] + arr[end:end_idx]
        return [arr_ext, arr_res]    

    # load files
    data = [np.loadtxt(f,delimiter="\t", skiprows=1, dtype={'names':('Time','Value'),'formats':('S18','f8')}) for f in load_files]

    data2 = data
    DATA_NUM = len(data)

    # add time column
    data = [ d['Value'] for d in data ]

    time_data = [dt.strptime(str(d)[3:20], '%d-%b-%Y %H:%M') for d in data2[0]['Time']]    
    data.append(np.transpose([d.day for d in time_data]))
    data.append(np.transpose([d.hour for d in time_data]))

    DATA_NUM = len(data)

    dataset = [];
    for i in range(0,len(test_idx)):
        ti = test_idx[i]
        [train_data, train_label, test_data, true_data] = create_traindata(data, ti, DATA_NUM, from_idx=from_idx, end_idx=end_idx)
        dict = {'train_data': train_data, 'train_label': train_label, 'test_data':test_data, 'true_data': true_data}
        dataset.append(dict)

    for i in range(0,len(test_idx)):
        estimated = [];
        estimator = RandomForestRegressor()
        model = RandomForestRegressor(n_estimators=N_est,n_jobs=-1)
        model.fit(np.transpose(dataset[i]['train_data']), np.array(dataset[i]['train_label'], dtype=np.float64))
        dataset[i].update({'estimated':model.predict(np.transpose(dataset[i]['test_data']))})

    comp_plot(dataset, days, title=pointname)    

