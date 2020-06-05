import argparse
import glob
import gower
import heapq
import joblib
import logging
import os.path
import random
import numpy as np
import pandas as pd
import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.explain.model._internal.explanation_client import ExplanationClient
from datetime import datetime


def select_top_RNs(subCluster_idx, RN_cnt_per_subcluster):
    ''' Return indecies of most reliable negatives in the given list (subCluster_idx)
        with the size of RN_cnt_per_subcluster '''
    avg_dist_from_positives = [gower_dist_avg[i] for i in subCluster_idx]
    top_idx = sorted(range(len(avg_dist_from_positives)), key=lambda sub: avg_dist_from_positives[sub])[
              -RN_cnt_per_subcluster:]
    top_RN_idx_per_subCluster = [subCluster_idx[i] for i in top_idx]
    return top_RN_idx_per_subCluster


def reliable_negative(test_data, raw_positives, subclusterCount, rep):
    raw_positives_idx = [test_data['SpecId'].tolist().index(x) for x in raw_positives['SpecId'].tolist() if
                         x in test_data['SpecId'].tolist()]
    non_positive_idx = np.setdiff1d(range(0, len(test_data)), raw_positives_idx).tolist()
    random.seed(rep)
    random.shuffle(non_positive_idx)
    subcluster_size = round(len(non_positive_idx) / subclusterCount)
    RN_cnt_per_subcluster = round(len(raw_positives) / subclusterCount) + 1
    RN_idx = []
    i = 0
    while True:
        random.seed(i)
        i += 1
        subCluster_idx = np.random.choice(non_positive_idx, size=subcluster_size, replace=False).tolist()
        RN_idx = np.append(RN_idx, select_top_RNs(subCluster_idx, raw_positives_idx, RN_cnt_per_subcluster))
        RN_idx = pd.unique(RN_idx)
        if len(RN_idx) >= len(raw_positives):
            return RN_idx[0:len(raw_positives)]
            break


def peptide_identification(args):
    print(datetime.now(), ': Peptid identification starts...')
    print('Settings: ')
    print(args)

    # PLATO setting
    subclusterCount = args.subclusterCount
    spy = args.spy
    spy_portion = args.spy_portion
    RN = args.RN
    rnd_all = args.rnd_all  # If random method, include all decoys
    rnd_portion = args.rnd_portion  # If random method, include rnd.portion of positive set, default 1: pos set = neg set
    replicates_cnt = args.replicates_cnt
    include_label = args.include_label
    AML_preprocess = args.AML_preprocess
    output_folder = args.output_folder

    # AutoML parameter setting
    autoML_best_model_selection = args.autoML_best_model_selection
    autoML_iterations = args.autoML_iterations

    metric = args.metric  # Other metrics: azureml.train.automl.utilities.get_primary_metrics('classification')
    cv_fold = args.cv_fold

    # Input, output
    file_name = args.sample_name
    input_path = args.input_folder
    output_path = output_folder + '/' + file_name
    log_file = output_path + '_autoML_errors_log.html'

    # Instantiate AutoML config and create an experiment in autoML workspace
    ws = Workspace.from_config()
    experiment_name = file_name
    experiment = Experiment(ws, experiment_name)
    print(datetime.now(), ': Assigned experiment ' + experiment_name + ' on Azure portal ')

    output = {}
    output['SDK version'] = azureml.core.VERSION
    output['Workspace Name'] = ws.name
    output['Resource Group'] = ws.resource_group
    output['Location'] = ws.location
    outputDf = pd.DataFrame(data=output, index=[''])
    print(outputDf)

    print(datetime.now(), ': Reading inputs')
    # Read POSITIVES and ALL inputs
    positives_path = glob.glob(input_path + file_name + '*POSITIVES*')
    raw_positives = pd.read_csv(positives_path[0], sep='\t')

    if AML_preprocess == True:
        all_path = glob.glob(input_path + file_name + '-ALL.txt')
        raw_all = pd.read_csv(all_path[0], sep='\t')
        # Extract new features
        # First and last three amino acides of peptide sequences as features - If NA then B category
        raw_all['Peptide'] = raw_all.Peptide.str.replace(r'([\(\[]).*?([\)\]])', r'B', regex=True)
        raw_all['P1'] = raw_all['Peptide'].str[0]
        raw_all['P2'] = raw_all['Peptide'].str[2]
        raw_all['P3'] = raw_all['Peptide'].str[3]
        raw_all['P4'] = raw_all['Peptide'].str[-4]
        raw_all['P5'] = raw_all['Peptide'].str[-3]
        raw_all['P6'] = raw_all['Peptide'].str[-1]

    else:
        all_path = glob.glob(input_path + file_name + '_percolator_feature.txt')
        raw_all = pd.read_csv(all_path[0], sep='\t')

    raw_all['Class'] = 0

    # Make positive and test set
    test_data = raw_all.drop(['ScanNr', 'Proteins'], axis=1)
    positive_set = pd.merge(left=pd.DataFrame(raw_positives['SpecId']), right=pd.DataFrame(test_data), how='left',
                            left_on='SpecId', right_on='SpecId')
    positive_set['Class'] = 1

    # Remove decoys in positive set, if there is any
    decoys_in_positive_idx = positive_set.index[positive_set['Label'] == -1].tolist()
    positive_set = positive_set[positive_set['Label'] != -1]

    # Dataframe to store predictions
    all_predictions = pd.DataFrame({
        'SpecId': list(test_data['SpecId']),
        'Peptide': list(test_data['Peptide']),
        'Label': list(test_data['Label'])})
    prediction_summary = all_predictions

    # Prepare test set for modeling
    y_test = test_data['Class']
    if include_label == True:
        X_test = test_data.drop(['SpecId', 'Peptide', 'Class'], axis=1)
    else:
        X_test = test_data.drop(['SpecId', 'Peptide', 'Label', 'Class'], axis=1)

    # Prepare positive set for modeling
    positive_set_idx = [test_data['SpecId'].tolist().index(x) for x in positive_set['SpecId'].tolist() if
                        x in test_data['SpecId'].tolist()]

    # Used to create the negative set
    decoys_idx = np.setdiff1d(test_data.index[test_data['Label'] == -1].tolist(), decoys_in_positive_idx).tolist()

    global gower_dist_avg
    if RN == True:
        if os.path.exists(input_path + file_name + 'gower_dist_avg.npy') == False:
            print(datetime.now(), ': Calculating Gower distance')
            gower_dist = gower.gower_matrix(test_data)
            selected_rows = gower_dist[positive_set_idx]
            gower_dist_avg = np.mean(selected_rows, axis=0)
            print(datetime.now(), ': Saving Gower distance matrix')
            np.save(input_path + '/' + file_name + 'gower_dist_avg.npy', gower_dist_avg)  # save
        else:
            print(datetime.now(), ': Loading Gower distance matrix from ', input_path + file_name + 'gower_dist_avg.npy')
            gower_dist_avg = np.load(input_path + file_name + 'gower_dist_avg.npy')  # load

    if spy == True:
        all_spies = pd.DataFrame()

    '''
    Create train set by concatinating positive and negative set, build model(s) using autoML
    and store predictions based on the best model
    '''
    for rep in range(0, replicates_cnt):
        print(datetime.now(), ': Replicate #', rep + 1)
        if spy == True:
            # Exclude spy_portion of training data to be the spies
            positive_set = positive_set.sample(n=len(positive_set), random_state=rep * 100).reset_index(drop=True)
            spySet_size = round(len(positive_set) * spy_portion)
            spies_ID = positive_set.loc[1:spySet_size, ['SpecId']]
            positive_set_wSpy = positive_set.iloc[spySet_size + 1:len(positive_set)]

        if RN == False:
            if rnd_all == True:
                # Negative set includes all decoys
                negative_set_idx = decoys_idx
            else:
                # Negative set idx includes rnd_portion times of |positive_set| indecies
                random.seed(rep)
                random.shuffle(decoys_idx)
                negative_set_idx = decoys_idx[0:rnd_portion * len(positive_set)]
        else:
            print(datetime.now(), ': Starts estimating RNs')
            negative_set_idx = reliable_negative(test_data, positive_set, subclusterCount, rep)
            print(datetime.now(), ': Ends estimating RNs')

        negative_set = test_data.iloc[negative_set_idx]

        if spy == True:
            train_data = pd.concat([positive_set_wSpy, negative_set], axis=0)
        else:
            train_data = pd.concat([positive_set, negative_set], axis=0)

        y_train = train_data['Class']
        if include_label == True:
            X_train = train_data.drop(['SpecId', 'Peptide', 'Class'], axis=1)
        else:
            X_train = train_data.drop(['SpecId', 'Peptide', 'Class', 'Label'], axis=1)


        print('Training set size:', len(y_train), '\nTest set size:', len(y_test))

        automl_config = AutoMLConfig(task='classification',
                                     debug_log=log_file,
                                     primary_metric=metric,
                                     iteration_timeout_minutes=200,
                                     iterations=autoML_iterations,
                                     verbosity=logging.INFO,
                                     preprocess=AML_preprocess,
                                     X=X_train,
                                     y=y_train,
                                     n_cross_validations=cv_fold,
                                     model_explainability=True)

        print(datetime.now(), ': modeling replicate #' + str(rep + 1) + '...')
        local_run = experiment.submit(automl_config, show_output=True)

        if autoML_best_model_selection == False:
            # Retrieve the Best Model based on bunch of metrics
            children = list(local_run.get_children())
            metricslist = {}
            for run in children:
                properties = run.get_properties()
                metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
                metricslist[int(properties['iteration'])] = metrics

            rundata = pd.DataFrame(metricslist).sort_index(1)
            tmp = rundata.T.sort_values(
                ['AUC_weighted', 'f1_score_weighted', 'precision_score_weighted', 'recall_score_weighted',
                 'weighted_accuracy'], ascending=False)
            rundata = tmp.sort_values('log_loss', ascending=True).T
            best_run_iteration = rundata.columns.values[0]
            rundata.to_csv(output_path + '_metrics_list_' + str(rep) + '.txt')
            best_run, fitted_model = local_run.get_output(iteration=best_run_iteration)
        else:
            best_run, fitted_model = local_run.get_output()

        print('Best run: ', best_run)
        print(datetime.now(), ': Saving best model and predictions')
        # Save the best model, prediction value and probability
        modelname = output_path + '_model_' + str(rep) + '.sav'
        joblib.dump(fitted_model, modelname)
        y_pred_val = fitted_model.predict(X_test)
        y_pred_prob = fitted_model.predict_proba(X_test)

        # Add the results of the replicate to all predictions table
        all_predictions['pred_rep' + str(rep)] = list(y_pred_val)
        all_predictions['prob_rep' + str(rep)] = list([item[1] for item in y_pred_prob])

        # Overwrite prediction values based on the spies cutoff
        if spy == True:
            threshold = min(pd.merge(spies_ID, all_predictions, on='SpecId')['prob_rep' + str(rep)])
            all_predictions['pred_rep' + str(rep)] = np.where(all_predictions['prob_rep' + str(rep)] >= threshold, 1, 0)
            all_spies['SpecId' + str(rep)] = spies_ID['SpecId']
            all_spies['Prob_rep' + str(rep)] = list(
                pd.merge(spies_ID, all_predictions, on=['SpecId'])['prob_rep' + str(rep)])

        print(datetime.now(), ': Replicate #' + str(rep + 1) + ' processed!')
        all_predictions.to_csv(output_path + '_all_predictions.csv', index=False)

    if spy == True:
        all_spies.to_csv(output_path + '_all_spies.csv', index=False)

    print(datetime.now(), ': Generate prediction summary of all replicates')
    pred_col_indecies = [col for col in all_predictions.columns if 'pred' in col]
    prob_col_indecies = [col for col in all_predictions.columns if 'prob' in col]

    prediction_summary['Std'] = all_predictions[prob_col_indecies].std(skipna=True, axis=1)
    prediction_summary['Min'] = all_predictions[prob_col_indecies].min(skipna=True, axis=1)
    prediction_summary['Max'] = all_predictions[prob_col_indecies].max(skipna=True, axis=1)
    prediction_summary['Avg'] = all_predictions[prob_col_indecies].mean(skipna=True, axis=1)
    prediction_summary['Median'] = all_predictions[prob_col_indecies].median(skipna=True, axis=1)
    prediction_summary['Vote'] = all_predictions[pred_col_indecies].sum(skipna=True, axis=1)
    prediction_summary.to_csv(output_path + '_prediction_summary.txt', sep='\t', index=False)

    # Feature importance
    print(datetime.now(), ': Output feature importance of the best run')
    client = ExplanationClient.from_run(best_run)
    raw_explanations = client.download_model_explanation(top_k=len(X_test.columns))
    print('Raw feature importance')
    print(raw_explanations.get_feature_importance_dict())
    d = raw_explanations.get_feature_importance_dict()
    raw_feature_importance = pd.DataFrame(list(d.items()))
    raw_feature_importance.to_csv(output_path + '_raw_feature_importance.csv', index=False)
    # Engineered
    engineered_explanations = client.download_model_explanation(top_k=len(X_test.columns))
    print('Engineered feature importance')
    print(engineered_explanations.get_feature_importance_dict())
    d = engineered_explanations.get_feature_importance_dict()
    engineered_feature_importance = pd.DataFrame(list(d.items()))
    engineered_feature_importance.to_csv(output_path + '_engineered_feature_importance.csv', index=False)

    now = datetime.now()
    print(datetime.now(), ': Program end')


def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError('{0} does not exist'.format(f))
    return f


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False


def parse_command_line():
    # Get the arguments passed through command line interface
    parser = argparse.ArgumentParser(
        description='''
           Peptide identification from MS/MS data
           Example:
                   python3 PLATO_v20.06.py --input_folder ./mel3/ --output_folder ./mel3_result/ --sample_name human-mel3-20140304-1  --autoML_iterations 15 > ./mel3/human-mel3-20140304.log 
       ''')

    parser.add_argument(
        '--sample_name',
        default=None,
        help='Sample name',
    )
    parser.add_argument(
        '--input_folder',
        default=None,
        help='Input folder should include two tsv files per sample: sample_name-POSITIVES.txt and sample_name-ALL.txt',
    )
    parser.add_argument(
        '--output_folder',
        default=None,
        help='Output folder',
    )
    parser.add_argument(
        '--RN',
        default=False,
        type=str2bool,
        help='Use reliable negative method to create negative set, default: False',
    )
    parser.add_argument(
        '--rnd_all',
        default=True,
        type=str2bool,
        help='Select all decoys as negative set, default: True',
    )
    parser.add_argument(
        '--rnd_portion',
        default=1,
        type=int,
        help='The portion of randomly selected decoys as negative set, default: 1',
    )
    parser.add_argument(
        '--include_label',
        default=False,
        type=str2bool,
        help='Include Label column as a feature, default: False',
    )
    parser.add_argument(
        '--replicates_cnt',
        default=20,
        type=int,
        help='Number of replicates, default: 20',
    )
    parser.add_argument(
        '--spy',
        default=False,
        type=str2bool,
        help='Incorporate spies in modeling, default: False',
    )
    parser.add_argument(
        '--spy_portion',
        default=0.1,
        help='Exclude the spy_portion of training set to be used as spies, default: 0.1',
    )
    parser.add_argument(
        '--subclusterCount',
        default=10,
        type=int,
        help='Number of subsamples, default: 10',
    )
    parser.add_argument(
        '--AML_preprocess',
        default=True,
        type=str2bool,
        help='AutoML does preprocessing before modeling, default: True',
    )
    parser.add_argument(
        '--cv_fold',
        default=10,
        type=int,
        help='How many cross validations to perform when user validation data is not specified, default: 10',
    )
    parser.add_argument(
        '--metric',
        default='accuracy',
        help='The metric that َAutoML will optimize for model selection, default: accuracy',
    )
    parser.add_argument(
        '--autoML_iterations',
        default=10,
        type=int,
        help='The total number of different piplines to test during an autoML experiment, default: 10',
    )
    parser.add_argument(
        '--autoML_best_model_selection',
        default=True,
        type=str2bool,
        help='Best run per is selected by AutoML based on given metric, default: True',
    )

    cmd_args = parser.parse_args()
    return cmd_args


if __name__ == '__main__':
    # Parse command line and get arguments
    args = parse_command_line()
    peptide_identification(args)
