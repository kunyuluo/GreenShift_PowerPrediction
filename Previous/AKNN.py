'''
AKNN prediction model

Feb 15 2022

@author: HuangChe
'''

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import timeit
import json
import os
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d  %H:%M:%S %a',
    filename=r'./info/AKNN_PY.log',
    filemode='a'
    )


def read_json(path):
    '''
    Get model parameters from json file

    Parameters
    ----------
    path : String
        Path of the model file.

    Returns
    -------
    param : dict
        Model parameters.

    '''
    with open(path, 'r') as file:
        param = json.load(file)
    file.close()

    return param


def save_json(path, param, indent=4):
    '''
    Save model parameters in to json file

    Parameters
    ----------
    path : String
        Path to save the model file.
    param : dict
        Model parameters.
    indent : int, optional
        Indent to write the dict into json file. The default is 4.

    Returns
    -------
    None.

    '''
    with open(path, 'w') as file:
        json.dump(param, file, indent=indent)


def DQC(data, interval='H'):
    '''
    Data quality checking and time/date components

    Parameters
    ----------
    data : DataFrame
        Dataframe contains variables with time.
    interval : DateOffset, Timedelta or str, optional
        The offset string or object representing target conversion.
        The default is 'H'.

    Returns
    -------
    data : DataFrame
        Dataframe contains variables, time and time/date components, including
        "year", "month", "weekofyear", "dayofyear", "dayofweek",
        "hour", "min", "minofday"
    flag : int or bool
        if 0, good data quality;
        else if 1, missing data or wrong data formating.

    '''
    flag = 0
    data.index = pd.Series(
        pd.to_datetime(data['time'], utc=True)  # , format='%m/%d/%Y %H:%M')
        )
    data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    data = data.resample(interval).interpolate(method='time')
    err = np.where(data.isna())[0]
    if err.size > 0:
        flag = 1
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['weekofyear'] = data.index.weekofyear
    data['dayofyear'] = data.index.dayofyear
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    data['min'] = data.index.minute
    data['minofday'] = data['min']+data['hour']*60
    print(f'full_time_lenth: {len(data)}')
    data.dropna(inplace=True)
    print(f'non-nan_data_lenth: {len(data)}')

    data.reset_index(inplace=True)

    return data, flag


def normalize01(X):
    '''
    Normalize data to range 0~1

    Parameters
    ----------
    X : Dataframe
        Input and output variables.

    Returns
    -------
    S : Dataframe
        Normalized variables.
    PS : Dataframe
        Maximum and minimum of each varaible column in training data.

    '''
    xmax = X.max()  # maximum of each column
    xmin = X.min()  # minimum of each column

    S = pd.DataFrame()
    for name, col in X.iteritems():
        s = (col - xmin[name]) / (xmax[name] - xmin[name])
        S[name] = s

    PS = pd.DataFrame()
    PS['xmax'] = xmax
    PS['xmin'] = xmin

    return S, PS


def NormProject(X, PS):
    '''
    Normalize data to range 0~1

    Parameters
    ----------
    X : Series
        Input (and output) variables.
    PS : Dataframe
        Maximum and minimum of each varaible column in training data.

    Returns
    -------
    S : Series
        Normalized variables.

    '''
    xmin = PS['xmin']
    xmax = PS['xmax']
    S = pd.Series()
    for name, col in X.iteritems():
        s = (col - xmin[name]) / (xmax[name] - xmin[name])
        S[name] = s

    return S


def HEGaussian(x, mu, covariance):
    '''
    Correlated Hyper Ellipsoidal Gaussian kernel (
        i.e., no restriction is imposed on the covariance terms
        )

    Parameters
    ----------
    x : Series
        Including input and output.
    mu : Series
        Position vector of a kernel.
    covariance : Matrix
        Covariance matrix of a kernel.

    Returns
    -------
    Float
        Clustering criteria.

    '''
    m = x.shape[0]
    x_mu = np.matrix(x.values - mu.values)
    K = np.exp(
        (-0.5) * x_mu @ np.linalg.inv(covariance) @ x_mu.T
        ) / (((2 * np.pi)**(m / 2)) * np.sqrt(np.linalg.det(covariance)))
    if K[0, 0] == np.inf:
        Kr = 1e6
    elif K[0, 0] == -np.inf:
        Kr = -1e6
    else:
        Kr = K[0, 0]

    return Kr


def LoadDiffEstiamtion(data, LimSams):
    '''
    Get the average load difference

    Parameters
    ----------
    data : DataFrame
        Including variables, time and time/date components.
    LimSams : int
        The parameter of how much days select to calculate.

    Returns
    -------
    lambdahat : Array
        Average load difference at the same moment of all days.
    DailyTime : list
        Different moment of every day.
    AverDifL : Average load difference
        Average load difference at the same moment of selected days.
    NumPoints : dict
        number of days in each moment.
    MomentLoad : dict
        load difference at the same moment of all days.

    '''
    n = data.shape[0]
    data['DiffLoad'] = data['load'].diff()
    data['DiffLoad'] = data['DiffLoad'].shift(-1)

    data['hour_min'] = data['hour'] + data['min'] / 60
    DailyTime = data['hour_min'].unique().tolist()

    lambdahat = {}
    MomentLoad = {}
    NumPoints = {}
    for name, grp in data.groupby('hour_min'):
        MomentLoad[name] = grp['DiffLoad']
        lambdahat[name] = grp['DiffLoad'].mean()
        NumPoints[name] = len(MomentLoad[name])

    for i in range(n):
        if i != n - 1:
            UpNowDiff = data.loc[:i, :]
            currMomLoad = UpNowDiff[
                UpNowDiff['hour_min'] == UpNowDiff.loc[i, 'hour_min']
                ]
            if len(currMomLoad) > 1:
                if len(currMomLoad) > LimSams:
                    currMomLoad.reset_index(inplace=True, drop=True)
                    data.loc[i, 'AverDifL'] = currMomLoad.loc[
                        len(currMomLoad) - LimSams:, 'DiffLoad'
                        ].mean()
                else:
                    data.loc[i, 'AverDifL'] = currMomLoad.loc[
                        :, 'DiffLoad'
                        ].mean()
            else:
                data.loc[i, 'AverDifL'] = lambdahat[
                    UpNowDiff.loc[i, 'hour_min']
                    ]

        else:
            lastMonLoad = MomentLoad[data.loc[i, 'hour_min']]
            if len(lastMonLoad) <= LimSams:
                data.loc[i, 'AverDifL'] = lastMonLoad.mean()
            else:
                lastMonLoad.reset_index(inplace=True, drop=True)
                data.loc[i, 'AverDifL'] = lastMonLoad.loc[
                    len(lastMonLoad)-LimSams:
                        ].mean()

    AverDifL = data['AverDifL']

    return lambdahat, DailyTime, AverDifL, NumPoints, MomentLoad


def AKNNTrain(S, Sigma, mu_ker, nk, numK):
    '''
    Train the AKNN model by updating the kernels

    Parameters
    ----------
    S : Series
        Contains the normalized input vector and ScalarOutput ([X y]).
    Sigma : dict
        Covariance matrix of each kernels.
    mu_ker : dict
        Position vector (mean value) of the kernels.
    nk : list
        Number of samples clustered in each kernel.
    numK : int
        Number of kernels.

    Returns
    -------
    Sigma : dict
        Updated covariance matrix after training by the sample S.
    mu_ker : dict
        Updated position vector (mean value) after training by the sample S.
    nk : list
        Updated number of samples clustered in each kernel \
            after training by the sample S.
    numK : int
        Updated number of kernels after training by the sample S.

    '''
    e = 2.71828
    m = S.shape[0]

    Psk = []
    Pk = []
    for i in range(numK):
        Psk.append(HEGaussian(S, mu_ker[i], Sigma[i]))
        Pk.append(nk[i] / sum(nk))
    Ps = sum(np.multiply(Psk, Pk)) + 1e-9
    ProbK = np.multiply(Psk, Pk) / Ps  # Bayes's theorem  #p(φj|s)
    # kenerl nomination

    IndexOfNomination = 1  # if 1, keep Re-nominate
    while (IndexOfNomination):
        whichK = np.argmax(ProbK)  # Index of the maximum value
        # Nomination the kernel with highest probability
        HJold = 0.5 * np.log(
            ((2 * np.pi * e) ** m) * np.linalg.det(Sigma[whichK])
            )
        nkJ = nk[whichK] + 1
        muJ = (mu_ker[whichK] * nk[whichK] + S) / nkJ
        SigmaJ = (
            Sigma[whichK] * nk[whichK]
            + np.matrix(S - muJ).T @ np.matrix(S - muJ)
            + nk[whichK] * np.matrix(mu_ker[whichK] - muJ).T @ np.matrix(
                mu_ker[whichK] - muJ
                )
            ) / nkJ
        HJnew = 0.5 * np.log(((2 * np.pi * e)**m) * np.linalg.det(SigmaJ))
        # kenerl confirmation
        if HJnew < HJold:
            # If uncertainty of kernel J is reduced after including new sample
            # Update the kernel
            IndexOfNomination = 0  # jump out of while next time
            nk[whichK] = nkJ
            mu_ker[whichK] = muJ
            Sigma[whichK] = np.asarray(SigmaJ)
        else:
            ProbK[whichK] = 0
            # Let the probability of this kernel be 0
            # Re-nominate with highest probability in rest kernels

        if sum(abs(ProbK)) == 0:
            # If none of the kernels satisfies the criteria
            # Create a new kernal
            numK = numK + 1
            nk.append(1)
            mu_ker[numK - 1] = S
            Agama = np.zeros((2, numK - 1))
            for j in range(numK - 1):
                # determine the kernel width(gama) autonomously
                Agama[0, j] = np.linalg.norm(
                    (Sigma[j] + np.linalg.norm(
                        mu_ker[j] - S
                        ) * np.emath.sqrt(Sigma[j])), ord=2
                    )
                Agama[1, j] = np.linalg.norm(
                    (Sigma[j] - np.linalg.norm(
                        mu_ker[j] - S
                        ) * np.emath.sqrt(Sigma[j])), ord=2
                    )

            gama = Agama.min()
            Sigma[numK - 1] = gama * np.identity(m)
            IndexOfNomination = 0
            # logging.info(
            #     "create kernel {}, sample {}: {}".format(numK, sum(nk), S)
            #     )
            print("create kernel {}, sample {}".format(numK, sum(nk)))

    return Sigma, mu_ker, nk, numK


def AKNNPrediction(tempS, Sigma, mu_ker, nk, numK):
    '''
    Predict the output by giving input vector based on the created kernels

    Parameters
    ----------
    tempS : Series
        Given normalized input vector (X) for prediction.
    Sigma : dict
        Covariance matrix of each kernel.
    mu_ker : dict
        Position vector (mean value) of each kernel.
    nk : list
        Number of samples clustered in each kernel.
    numK : int
        Number of kernels.

    Returns
    -------
    ECM : float
        the predicted result (
            Expected conditional mean of the output space
            with the given input vector
            ).

    '''
    m = tempS.shape[0]
    sumK = 0
    muK = 0

    for j in range(numK):
        tempmean = mu_ker[j][:m]
        tempcov = Sigma[j][:m, :m]
        nkJ = nk[j] + 1
        muJ = (tempmean * nk[j] + tempS) / nkJ
        SigmaJ = (
            tempcov * nk[j]
            + np.matrix(tempS - muJ).T @ np.matrix(tempS - muJ)
            + nk[j] * np.matrix(tempmean - muJ).T @ np.matrix(tempmean - muJ)
            ) / nkJ
        tempK = HEGaussian(tempS, muJ, SigmaJ)

        tempSigma = np.linalg.inv(Sigma[j])
        aj = tempSigma[-1].reshape(1, -1)
        tempmu = mu_ker[j][-1] - (
            np.sum(np.multiply(np.matrix(tempS - tempmean), aj[0, :m]))
            ) / aj[0, -1]

        sumK = sumK + tempK
        muK = muK + tempmu * tempK

    ECM = muK / sumK

    return ECM


def EngineInitialization(Strain, StorePath=None, StoreName=None):
    '''
    Initialize the model based on training samples
    Save all the kernels in a file named "Patterns"

    Parameters
    ----------
    Strain : DataFrame
        Variables to train the model, contains inputs and outputs.
    StorePath : str or None, optional
        Location to save the trained model. The default is None.
    StoreName : str or None, optional
        Name of the stored model. The default is None.

    Returns
    -------
    PS : DataFrame
        Maximum and minimum of each varaible column in training data.

    '''
    fs = [
        col for col in Strain.columns if col != 'ScalarOutput'
        ] + ['ScalarOutput']
    TrainS = Strain[fs]
    TrainS = TrainS.astype(float)
    S, PS = normalize01(TrainS)
    # Normalization

    TrainingS = S
    n, m = TrainingS.shape

    numK = 2  # number of kernels 存储存在的kernel数目
    nk = []  # number of sumples in each kernel 存储每个kernel的样本数
    Sigma = {}  # sigma of each kernel 存储每个kernel的sigma
    mu_ker = {}  # mean vector of each kernel 存储每个kernel的mean
    # initial parameter

    InitialSigma = 0.5 * (np.linalg.norm(
        (TrainingS.iloc[0, :] - TrainingS.iloc[1, :])
        )) * np.identity(m)

    nk.append(1)
    nk.append(1)
    Sigma[0] = InitialSigma
    Sigma[1] = InitialSigma
    mu_ker[0] = TrainingS.iloc[0, :]
    mu_ker[1] = TrainingS.iloc[1, :]

    for i in range(2, n):
        Sigma, mu_ker, nk, numK = AKNNTrain(
            TrainingS.iloc[i, :], Sigma, mu_ker, nk, numK
            )

    Sigma_dict = {}
    for Key, Value in Sigma.items():
        Sigma_dict[Key] = Value.tolist()
    mu_ker_dict = {}
    for Key, Value in mu_ker.items():
        mu_ker_dict[Key] = Value.to_dict()

    Patterns = {
        'Sigma': Sigma_dict,
        'mu_ker': mu_ker_dict,
        'nk': nk,
        'numK': numK}
    if not os.path.exists(StorePath):
        try:
            os.mkdir(StorePath)
        except:
            pass
    save_json(StorePath + StoreName + '_Patterns.json', Patterns)
    save_json(StorePath + StoreName + '_Patterns_train.json', Patterns)

    print(f'kernel number of model trained: {numK}', nk)

    return PS


def EngineUpdate(Stest, PS, Sigma, mu_ker, nk, numK):
    '''
    Update kernels with test sample

    Parameters
    ----------
    Stest : Series
        Given input and output varaibles for kernel updating.
    PS : DataFrame
        Maximum and minimum of each varaible column in training data.
    Sigma : dict
        Covariance matrix of each kernel.
    mu_ker : dict
        Position vector (mean value) of each kernel.
    nk : list
        Number of samples clustered in each kernel.
    numK : int
        Number of kernels.

    Returns
    -------
    Sigma : dict
        Updated covariance matrix after training by the sample Stest.
    mu_ker : dict
        Updated position vector (mean value)\
            after training by the sample Stest.
    nk : list
        Updated number of samples clustered in each kernel \
            after training by the sample Stest.
    numK : int
        Updated number of kernels after training by the sample Stest.

    '''
    Supdate = Stest
    UpdatingS = NormProject(Supdate, PS)

    Sigma, mu_ker, nk, numK = AKNNTrain(UpdatingS, Sigma, mu_ker, nk, numK)

    return Sigma, mu_ker, nk, numK


def EnginePrediction(
        Stest, PS, Loadupper, Loadlower, Sigma, mu_ker, nk, numK
        ):
    '''
    Predict load

    Parameters
    ----------
    Stest : Series
        Sample of test data.
    PS : DataFrame
        Maximum and minimum of each varaible column in training data.
    Loadupper : float
        Maximum of ScalarOutput in training data.
    Loadlower : float
        Minimum of ScalarOutput in training data.
    Sigma : dict
        Covariance matrix of each kernel.
    mu_ker : dict
        Position vector (mean value) of each kernel.
    nk : list
        Number of samples clustered in each kernel.
    numK : int
        Number of kernels.

    Returns
    -------
    onepredict : float
        Predicted next load.

    '''
    TestS = Stest
    TestingS = NormProject(TestS, PS)

    ECM = AKNNPrediction(TestingS, Sigma, mu_ker, nk, numK)
    onepredict = Loadlower + ECM * (Loadupper - Loadlower)

    return max(onepredict, 0)


def Model_Offline_Train(
        data, target, features=None, StorePath=None, StoreName=None,
        k_previ=24, step_previ=1, k_next=24, step_next=1,
        interval='H', previWflag=0
        ):
    '''
    Train the model offline

    Parameters
    ----------
    data : DataFrame
        Training data set.
    target : str
        Target feature name of the prediction model
    features : list or None
        Features used to train the model except for previous loads.
        The default is None.
    StorePath : str, optional
        Location to save the trained model. The default is None.
    StoreName : str, optional
        Name of the stored model. The default is None.
    k_previ : int, optional
        Number of previous load used. The default is 24.
    step_previ : int, optional
        Step lenth of previous load. The default is 1.
    k_next : int, optional
        Number of next load to be predicted. The default is 24.
    step_next : int, optional
        Step lenth of next load to be predicted. The default is 1.
    interval : DateOffset, Timedelta or str, optional
        The offset string or object representing target conversion.
        The default is 'H'.
    previWflag : int, optional
        If 0, only present weather is used.
        If not 0, the number of weather feature using previous k_previ value.
        The default is 0.

    Returns
    -------
    TrainResult : str
        if 'Not enough quantity of training set',
        training set < 300 data points, too few;
        if 'Done', training completed.
    flag : int or bool
        if 0, good data quality;
        else if 1, missing data or wrong data formating.

    '''
    start = timeit.default_timer()

    data, flag = DQC(data, interval=interval)
    if data.shape[0] < 300:
        TrainResult = 'Not enough quantity of training set'

    else:
        TrainResult = 'Done'

        fs_list = []
        if features:
            if previWflag == 0:
                fs_list = fs_list + features
            else:
                for fs_train in features[:-previWflag]:
                    fs_list.append(fs_train)
                for fs_train in features[-previWflag:]:
                    for l in range(k_previ):
                        data['Previ' + fs_train + str(l)] = (
                            data[fs_train].shift(periods=l*step_previ)
                            )
                        fs_list.append('Previ' + fs_train + str(l))
        else:
            pass

        for i in range(k_previ):
            data['Previ' + target + str(i)] = data[target].shift(periods=i*step_previ)
            fs_list.append('Previ' + target + str(i))

        data['ScalarOutput'] = data[target].shift(periods=-1*step_next)
        # load at next step
        fs_list.append('ScalarOutput')

        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)

        Strain = data.copy()
        for colname in Strain.columns.tolist():
            if colname not in fs_list:
                del Strain[colname]
        logging.info("training samples: {}, features:{}".format(
            len(Strain), Strain.columns.tolist()
            ))

        print(f"train data length: {len(Strain)}")
        PS = EngineInitialization(
            Strain, StorePath=StorePath, StoreName=StoreName
            )

        Loadlower = data['ScalarOutput'].min()
        Loadupper = data['ScalarOutput'].max()
        PS_dict = PS.to_dict()
        Model = {
            'PS': PS_dict,
            'Loadlower': Loadlower,
            'Loadupper': Loadupper
            }

        save_json(StorePath + StoreName + '_Model.json', Model)
        save_json(StorePath + StoreName + '_Model_train.json', Model)

        end = timeit.default_timer()
        print('training time: %s Seconds' % (end - start))
        logging.info("#" * 60)

    return TrainResult, flag, (end-start)


def Model_Online_RunOnce(
        data, target, features=None, StorePath=None, StoreName=None,
        k_previ=24, step_previ=1, k_next=24, step_next=1,
        interval='H', previWflag=0, updateflag=1
        ):
    '''
    Predict load

    Parameters
    ----------
    data : DataFrame
        Test data set.
    target : str
        Target feature name of the prediction model
    features : list or None
        List of features used in the model except for previous loads.
        The default is None.
    StorePath : str, optional
        Location to save the trained model. The default is None.
    StoreName : str, optional
        Name of the stored model. The default is None.
    k_previ : int, optional
        Number of previous load used. The default is 24.
    step_previ : int, optional
        Step lenth of previous load. The default is 1.
    k_next : int, optional
        Number of next load to be predicted. The default is 24.
    step_next : int, optional
        Step lenth of next load to be predicted. The default is 1.
    interval : DateOffset, Timedelta or str, optional
        The offset string or object representing target conversion.
        The default is 'H'.
    previWflag : int, optional
        If 0, only present weather is used.
        If not 0, the number of weather feature using previous k_previ value.
        The default is 0.
    updateflag: bool, 0 or 1, optional
        if 0, not update with newly recieved data,
        if 1, update with each newly recieved data.
        The default is 1.

    Returns
    -------
    Stest : DataFrame
        Contains test data inputs and outputs.
    flag : int or bool
        if 0, good data quality;
        else if 1, missing data or wrong data formating.

    '''
    start = timeit.default_timer()
    Modeljson = read_json(StorePath + StoreName + '_Model.json')
    Patternsjson = read_json(StorePath + StoreName + '_Patterns.json')
    PS = pd.DataFrame(Modeljson['PS'])
    Loadupper = Modeljson['Loadupper']
    Loadlower = Modeljson['Loadlower']
    nk = Patternsjson['nk']
    numK = Patternsjson['numK']
    Sigma_dict = Patternsjson['Sigma']
    Sigma = {}
    for Key, Value in Sigma_dict.items():
        Sigma[int(Key)] = np.matrix(Value)
    mu_ker_dict = Patternsjson['mu_ker']
    mu_ker = {}
    for Key, Value in mu_ker_dict.items():
        mu_ker[int(Key)] = pd.Series(Value)
    fs = PS.index.tolist()

    data, flag = DQC(data, interval=interval)
    # data['ScalarOutput'] = data[target].shift(periods=-1*step_next)  # load at next step
    # features_temp = features.copy()
    fs_list = []
    if features:  # _temp:
        if previWflag == 0:
            fs_list = fs_list + features  # features_temp.copy()
        else:
            fs_list = []
            for fs_test in features[:-previWflag]:
                fs_list.append(fs_test)
            for fs_test in features[-previWflag:]:
                for l in range(k_previ):
                    data['Previ' + fs_test + str(l)] = (
                        data[fs_test].shift(periods=l*step_previ)
                        )
                    fs_list.append('Previ' + fs_test + str(l))
    else:
        fs_list = []

    for l in range(k_previ):
        data['Previ' + target + str(l)] = data[target].shift(periods=l*step_previ)
        fs_list.append('Previ' + target + str(l))

    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)

    Stest = data.copy()
    print(f"test data length: {len(Stest)}")
    if updateflag==1:
        Stest['ScalarOutput'] = Stest[target].shift(periods=-1*step_next)
        # load at next step

    predictload = {}
    for i in range(len(Stest)):
        if (updateflag==1)&(i>0)&(i+step_next<len(Stest)):
            Sigma, mu_ker, nk, numK = EngineUpdate(
                Stest.iloc[i-1][fs], PS, Sigma, mu_ker, nk, numK,
                )
            # update kernal with actual data before prediction

        predictknext = {}
        timestamp = Stest.loc[i, 'time']
        onepredict = 0
        TestS = Stest.iloc[i][fs[:-1]]
        for j in range(k_next):
            if i+(j+1)*step_next >= len(Stest):
                break
            onepredict = EnginePrediction(
                TestS, PS, Loadupper, Loadlower,
                Sigma, mu_ker, nk, numK
                )
            predictknext['Predi' + target + str(j+1)] = onepredict

            TestS[fs_list] = Stest.iloc[i+(j+1)*step_next][fs_list]
            for kk in range((j*step_next)//step_previ+1):
                TestS['Previ'+target+str(kk)] = predictknext[
                    'Predi' + target + str(int(j-kk*(step_previ/step_next)+1))
                    ]

        predictload[timestamp] = predictknext

    if updateflag == 1:
        Sigma_dict = {}
        for Key, Value in Sigma.items():
            Sigma_dict[Key] = Value.tolist()
        mu_ker_dict = {}
        for Key, Value in mu_ker.items():
            mu_ker_dict[Key] = Value.to_dict()
    
        Patterns = {
            'Sigma': Sigma_dict,
            'mu_ker': mu_ker_dict,
            'nk': nk,
            'numK': numK}
        save_json(StorePath + StoreName + '_Patterns.json', Patterns)

    PredictedLoad = pd.DataFrame(predictload)
    PredictedLoad = PredictedLoad.T
    PredictedLoad[PredictedLoad<0] = 0
    PredictedLoad.reset_index(inplace=True)
    PredictedLoad.rename(columns={'index': 'time'}, inplace=True)
    PredictedLoad['time'] = pd.to_datetime(
        PredictedLoad['time'], errors='coerce'
        )
    if features:
        TestResult = Stest[['time'] + features + [target]].merge(
            PredictedLoad, on='time')
    else:
        TestResult = Stest[['time'] + [target]].merge(
            PredictedLoad, on='time')
        
    for j in range(k_next):
        TestResult['Next' + target + str(j+1)] = (
            TestResult[target].shift(periods=-(j+1)*step_next)
            )

    print(f'kernel number of updated model: {numK}', nk)

    end = timeit.default_timer()
    print('testing time: %s Seconds' % (end - start))

    return TestResult, flag, (end-start)


if __name__ == '__main__':

    path = r'.\Data\InfDataClean.csv'
    fs = [
        'time',  # 'HourOfDayX','HourOfDayY','DayOfWeekX','DayOfWeekY',
        'outdoor_tempF','delta_cool','setback_off','cool_svg_49_2'
        ]
    # usable features, ordered 'time' + features to train the model + target
    data = pd.read_csv(path)
    dataa = data[fs]

    StorePath = r'.\json\\'
    StoreName = r'\InfDataClean_cool_svg_49_2_drop0_dow_mod'
    print(os.path.split(StoreName)[-1])
    features = [
        # 'HourOfDayX','HourOfDayY','DayOfWeekX','DayOfWeekY',
        'dayofweek', 'minofday',
        'outdoor_tempF','delta_cool','setback_off',]
    # features to train the model except for target
    # ordered time component + features (may use previous values)
    target = 'cool_svg_49_2'
    interval = "5Min"
    k_previ = 36
    step_previ = 24
    k_next = 24  # k_next<=k_previ
    step_next = 12  # step_next<=step_previ
    previWflag = 3  # number of lagged input features except for target

    dataa = dataa[27:]
    dataa = dataa[
        (dataa['cool_svg_49_2'].cumsum()!=0)
        &(dataa.sort_index(ascending=False).cool_svg_49_2.cumsum()!=0)
        ]
    # drop continuous zero values in winter 
    
    traindata = dataa.iloc[40000-2016-576:40000-576]#[0:int(len(dataa)*0.9)]  # [step_previ:(53001-step_next+step_previ)]  # 
    testdata = dataa.iloc[40000-step_next-step_previ*k_previ+step_previ:40000]#[int(len(dataa)*0.9)+step_next-k_previ*step_previ:]  # [(53001-(k_previ-1)*step_previ):(57001+step_next)]  # 
    # traindays = 7
    # testdays = 3
    # if interval == 'H':
    #     stepperhour = 1
    # else:
    #     stepperhour = 60/int("".join(list(filter(str.isdigit, interval))))
    # trainidx = [
    #     int(-(traindays+testdays)*24*stepperhour-step_previ*k_previ+step_next*2),
    #     int(-testdays*24*stepperhour+step_next)
    #     ]
    # testidx = int(-testdays*24*stepperhour-step_previ*k_previ+step_next*2)
    # traindata = dataa[trainidx[0]:trainidx[1]]
    # testdata = dataa[testidx:]

    TrainResult, flag, traintime = Model_Offline_Train(
        traindata, target, features=features,
        StorePath=StorePath, StoreName=StoreName,
        k_previ=k_previ, step_previ = step_previ,
        k_next=k_next, step_next = step_next, interval=interval,
        previWflag=previWflag
        )
    print(
        f'TrainResult: {TrainResult}, \
    data quality: {flag} (0 -- good, 1 -- low)'
    )
    # train offline model

    TestResult, flag, testtime = Model_Online_RunOnce(
        testdata, target, features=features,
        StorePath=StorePath, StoreName=StoreName,
        k_previ=k_previ, step_previ = step_previ,
        k_next=k_next, step_next = step_next, interval=interval,
        previWflag=previWflag, updateflag=1
        )
    print(f'Test done, data quality: {flag} (0 -- good, 1 -- low)')

    TestMSE = []
    TestAAC = []
    for i in range(1, k_next+1):
        SqureError = (
            TestResult['Predi' + target + str(i)]
            - TestResult['Next' + target + str(i)]
            )**2
        TestMSE.append(SqureError.mean())
        Accuracy = 1 - ((
            TestResult['Predi' + target + str(i)]
            - TestResult['Next' + target + str(i)]
            ).abs() / np.clip(TestResult['Next'+target+str(i)], 1e-6, None))
        TestAAC.append(np.clip(Accuracy, 0, 1).mean())

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.title('Test Set AAC vs Forecast Hour')
    plt.xlabel('Hour')
    plt.ylabel('Average Accuracy')
    plt.plot(range(1, 25), TestAAC, color='b', lw=2)
    plt.show()

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.title('Test Set MSE vs Forecast Hour')
    plt.xlabel('Hour')
    plt.ylabel('Mean Squred Error')
    plt.plot(range(1, 25), TestMSE, color='b', lw=2)
    plt.show()

    evaluate = pd.DataFrame({
        "TestMSE": TestMSE,
        "TestAAC": TestAAC})
    evaluate.to_csv(r"Data" + StoreName + "_AKNN_PY_MSE_AAC.csv", index=False)


    idx = 0
    ytrue = []
    ypred = []

    for i in range(1, k_next+1):
        ytrue.append(TestResult.loc[idx, 'Next' + target + str(i)])
        ypred.append(TestResult.loc[idx, 'Predi' + target + str(i)])
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.title('AKNN_PY_Forecast')
    plt.xlabel('Runtime')
    plt.ylabel(target)
    plt.plot(range(1, 25), ytrue, color='b', lw=2, label='True')
    plt.plot(range(1, 25), ypred, 'r--', lw=2, label='Predicted')
    plt.legend()

    idx = 0
    TimeSteps = k_next*3*step_next
    data_x = []
    data_ypred = []
    data_ytrue = []
    x = range(1, 25)
    for i in range(TimeSteps):
        ytrue = []
        ypred = []
        for j in range(1, k_next+1):
            ytrue.append(TestResult.loc[idx+i, 'Next' + target + str(j)])
            ypred.append(TestResult.loc[idx+i, 'Predi' + target + str(j)])
        data_x.append(x)
        data_ypred.append(ypred)
        data_ytrue.append(ytrue)

    fig, ax = plt.subplots()
    ax.set_xlim([0, 25])
    ax.set_ylim([0, 310])
    ax.set_xlabel('Hour')
    ax.set_ylabel('Runtime')
    ax.set_title('AKNN_PY_Forecast_24HR')
    linetrue, = ax.plot(data_x[0], data_ytrue[0], 'b', lw=2, label='True')
    linepred, = ax.plot(
        data_x[0], data_ypred[0], 'r--', lw=2, label='Predicted'
        )
    ax.legend(handles=[linetrue, linepred], loc='upper left')

    def update(i):
        ax.texts.clear()
        linetrue.set_data(data_x[i], data_ytrue[i])
        linepred.set_data(data_x[i], data_ypred[i])
        Day = ((i+1) // (24*12))
        Hour = (i*5//60)
        ax.text(
            10, 4.5,
            "Day: "+str(Day)+" Hour: "+str(Hour)
            )
    anim = animation.FuncAnimation(fig, update, frames=TimeSteps, interval=100,
                                   blit=False, repeat=True, repeat_delay=3000)
    anim.save(r'fig' + StoreName + '_AKNN_PY_Forecast_24HRwindows_3days.gif', writer='imagemagick')
    plt.show()

    TestResult.to_csv(r"Data" + StoreName + "_TestResult.csv")
