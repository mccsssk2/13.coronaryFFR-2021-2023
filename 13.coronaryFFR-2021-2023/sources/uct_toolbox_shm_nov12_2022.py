#!/usr/bin/env python

## Imports
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
# Import Uncertainty Toolbox
import uncertainty_toolbox as uct
from simple_uq.models.pnn import PNN
import sys

'''
Author: Shambhavi Malik
Institute: Indian Institute of Technology (BHU) Varanasi
email: shambhavimalik.bme18@itbhu.ac.in
Date: 12 November 2022.

1 input, 1 output:
real	10m27.257s
user 10m35.880s
sys	0m10.885s

'''

## Specify file directories
# Training data.
parent_dir = 'modelUQ_training300Instances'
# Test data.
parent_dir_test = 'modelUQ_testing30Instances'
files = os.listdir(parent_dir)
# Input files
all_ffr_input = np.round_(np.loadtxt(os.path.join(parent_dir, 'All_FFRInputs.dat'), delimiter=" ", unpack=False), 6)
all_ffr_input_test = np.round_(np.loadtxt(os.path.join(parent_dir_test, 'All_FFRInputs.dat'), delimiter=" ", unpack=False), 6)
# Get index of all DAT files in parent directory
idx_opt = [] # list of model output files
idx_rcrcr = [] # list of RCRCR DAT files
idx_cfr = [] # list of CFR DAT files
for idx in range(len(files)):
    if files[idx].startswith('outputsmodel'):
        idx_opt.append(files[idx])
    elif files[idx].startswith('outputsRCRCR'):
        idx_rcrcr.append(files[idx])
    elif files[idx].startswith('outputsCFR'):
        idx_cfr.append(files[idx])

## Loop over all files and store numpy arrays in a list for easy access
train_input = []
train_output = []
test_input = []
test_output = []
for i in range(len(idx_opt)):
    # Train
    input_rcrcr = np.loadtxt(os.path.join(parent_dir, idx_rcrcr[i]), delimiter=" ", unpack=False)
    output_model = np.loadtxt(os.path.join(parent_dir, idx_opt[i]), delimiter=" ", unpack=False)
    output_cfr = np.loadtxt(os.path.join(parent_dir, idx_cfr[i]), delimiter=" ", unpack=False)
    input = np.concatenate((all_ffr_input,input_rcrcr),axis=1)
    train_input.append(input)
    output = np.concatenate((output_model,output_cfr),axis=1)
    train_output.append(output)
    # Test
    input_rcrcr_test = np.loadtxt(os.path.join(parent_dir_test, idx_rcrcr[i]), delimiter=" ", unpack=False)
    output_model_test = np.loadtxt(os.path.join(parent_dir_test, idx_opt[i]), delimiter=" ", unpack=False)
    output_cfr_test = np.loadtxt(os.path.join(parent_dir_test, idx_cfr[i]), delimiter=" ", unpack=False)
    input_test = np.concatenate((all_ffr_input_test,input_rcrcr_test),axis=1)
    test_input.append(input_test)
    output_test = np.concatenate((output_model_test,output_cfr_test),axis=1)
    test_output.append(output_test)

'''input_columns = list(range(train_input[0].shape[1])) # all input columns
drop_col = [3,5,6,10] # unwanted columns in input
for elem in drop_col: 
    input_columns.remove(elem)
output_columns = list(range(train_output[0].shape[1])) # all output columns

"""Loop over all columns in input and output files"""
for inp_col in input_columns:
    for opt_col in output_columns:'''

# Take input and output columns
inp_col = int(sys.argv[1]) # 0 # input('Enter input column')
opt_col = int(sys.argv[2]) # 1 # input('Enter output column')
# Create directory to store results
result_dir = './'+str(inp_col)+'_vs_'+str(opt_col)
os.makedirs(result_dir)
## Loop over all geometries
for geom in range(len(train_input)):
    input = train_input[geom]
    output = train_output[geom]
    input_test = test_input[geom]
    output_test = test_output[geom]
    ## Pre process data
    # Remove data rows where simulation failed (all column values = -1)
    # Get row index from output data files
    idx = []
    for row in range(len(output)):
        if [-1.]*25 == output[row].tolist():
            idx.append(row)
    # Drop failed simulation values from input and output files
    input_processed = np.delete(input, idx, 0)
    output_processed = np.delete(output, idx, 0)

    TrainValsplit = int(0.9 * len(input_processed[:, 0])) # split train data into training and validation sets (90% train, 10% validation)

    # Load data into pytorch dataloaders
    np_data, dataloaders = {}, {}
    for data_type in ['Train', 'Validation', 'Test']:
        if data_type == 'Train':
            xpts = input_processed[:TrainValsplit, inp_col]
            ypts = output_processed[:TrainValsplit, opt_col]
        elif data_type == 'Validation':
            xpts = input_processed[TrainValsplit:, inp_col]
            ypts = output_processed[TrainValsplit:, opt_col]
        elif data_type == 'Test':
            xpts = input_test[:, inp_col]
            ypts = output_test[:, opt_col]
        # Remove input and output elements from dataloaders where output value is -1
        idx_rem = [] # indexes to be removed
        for ir in range(len(ypts)):
            if ypts[ir]==-1:
                idx_rem.append(ir)
        ypts = np.delete(ypts,idx_rem)
        if len(ypts)==0: # all values -1 --> no data for model to train on 
            break 
        xpts = np.delete(xpts,idx_rem)
        np_data[data_type] = (xpts, ypts)
        dataloader = DataLoader(
            TensorDataset(
                torch.Tensor(xpts).reshape(-1, 1),
                torch.Tensor(ypts).reshape(-1, 1),
            ),
            batch_size=16,
            shuffle=True,
        )
        dataloaders[data_type] = dataloader

    if dataloaders == {}:  # if no data is available for training loop jumps to the next iteration 
        # print(idx_opt[model]+'_No data')
        continue

    else:
        """Make the PNN model."""
        pnn = PNN(
            input_dim=1,
            output_dim=1,
            encoder_hidden_sizes=[64, 64],
            encoder_output_dim=64,
            mean_hidden_sizes=[],
            logvar_hidden_sizes=[],
        )

        """Train the model with a pytorch-lightning trainer."""
        trainer = pl.Trainer(max_epochs=500, check_val_every_n_epoch=25)
        trainer.fit(pnn, dataloaders['Train'], dataloaders['Validation'])
        # Save model weights
        torch.save(pnn.state_dict(), os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'_pnn_weights.pth'))
        model = torch.load(os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'_pnn_weights.pth'))
        torch.set_printoptions(profile="full")
        string = str(model)
        with open(os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'_pnn_weights.txt'), 'w') as fp:
            fp.write(string)
        # Get the test output.
        test_results = trainer.test(pnn, dataloaders['Test'])

        # Get PNN predictive uncertainties
        te_x, te_y = np_data['Test']
        pred_mean, pred_std = pnn.get_mean_and_standard_deviation(te_x.reshape(-1, 1))
        pred_mean = pred_mean.flatten()
        pred_std = pred_std.flatten()

        # Confidence Interval Plot Function
        # Adopted and modified from uncertainity toolox function
        def plot_xy(
            y_pred: np.ndarray,
            y_std: np.ndarray,
            y_true: np.ndarray,
            x: np.ndarray,
            num_stds_confidence_bound: int = 2):
            """Plot one-dimensional inputs with associated predicted values, predictive
            uncertainties, and true values.

            Args:
                y_pred: 1D array of the predicted means for the held out dataset.
                y_std: 1D array of the predicted standard deviations for the held out dataset.
                y_true: 1D array of the true labels in the held out dataset.
                num_stds_confidence_bound: width of confidence band, in terms of number of
                    standard deviations.

            Returns:
                matplotlib.axes.Axes object with plot added.
            """

            font = {'name' : 'Times New Roman', 
                'size' : 24}
            # Create ax if it doesn't exist
            fig, ax = plt.subplots(figsize=(10, 10))

            # Order points in order of increasing x
            order = np.argsort(x)
            y_pred, y_std, y_true, x = (
                y_pred[order],
                y_std[order],
                y_true[order],
                x[order],
            )

            intervals = num_stds_confidence_bound * y_std

            h1 = ax.plot(x, y_true, ".", mec="#ff7f0e", mfc="None", mew=12)
            h2 = ax.plot(x, y_pred, "-", c="#1f77b4", linewidth=5)
            h3 = ax.fill_between(
                x,
                y_pred - intervals,
                y_pred + intervals,
                color="lightsteelblue",
                alpha=0.4,
            )
            ax.legend(
                [h1[0], h2[0], h3],
                ["Observations", "Predictions", "$95\%$ Interval"]
            )

            # Format plot
            '''ax.set_xlim(50, 150)
            ax.set_xticks([60, 100, 140], **font)
            ax.set_xticklabels([60, 100, 140], **font)'''
            ax.set_xlabel('Input Column '+str(inp_col+1), **font)
            '''ax.set_ylim(0, 1.1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], **font)'''
            ax.set_ylabel('Output Column '+str(opt_col+1), **font)
            ax.set_title(idx_opt[geom].removeprefix('outputs').removesuffix('.dat'), **font)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            for axis in ['bottom','left']:
                ax.spines[axis].set_linewidth(5)
            ax.tick_params(width=5, length = 15)
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

            return ax

        # Confidence Interval Plot
        plot_xy(pred_mean, pred_std, te_y, te_x)
        # plt.gcf().set_size_inches(5, 5)
        # plt.show()
        plt.savefig(os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'.png'))

        # Plot average calibration
        uct.viz.plot_calibration(pred_mean, pred_std, te_y) # function from uct toolbox
        plt.gcf().set_size_inches(4, 4)
        # plt.show()
        plt.savefig(os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'_calibration.png'))

        # Get all metrics
        pnn_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, te_y)
        # print(type(pnn_metrics))
        # Save metrics in a text file
        with open(os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'_metrics.txt'), 'w') as f:
            for key, value in pnn_metrics.items(): 
                f.write('%s:%s\n' % (key, value))

        ## Recalibration
        # Get recalibration data. (used test data)
        recal_x  = input_test[:, inp_col]
        recal_y = output_test[:, opt_col]

        # Get the predictive uncertainties in terms of expected proportions and observed proportions on the recalibration set.
        recal_pred_mean, recal_pred_std = pnn.get_mean_and_standard_deviation(
            recal_x.reshape(-1, 1)
        )
        recal_pred_mean = recal_pred_mean.flatten()
        recal_pred_std = recal_pred_std.flatten()
        exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
            recal_pred_mean, recal_pred_std, recal_y,
        )

        # Train a recalibration model.
        recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

        # Get the expected props and observed props using the new recalibrated model
        te_recal_exp_props, te_recal_obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
            pred_mean, pred_std, te_y, recal_model=recal_model
        )

        # Show the updated average calibration plot
        uct.viz.plot_calibration(pred_mean, pred_std, te_y,
                                exp_props=te_recal_exp_props,
                                obs_props=te_recal_obs_props)
        
        # plt.show()
        plt.savefig(os.path.join(result_dir, idx_opt[geom].removeprefix('outputs').removesuffix('.dat')+'_after_recalibration.png'))


