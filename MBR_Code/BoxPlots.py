import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def gteoutliers(beamSearch):
    Q1 = np.percentile(beamSearch, 25)
    Q3 = np.percentile(beamSearch, 75)

    # Compute the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = beamSearch[(beamSearch < lower_bound) | (beamSearch > upper_bound)]

    # Number of outliers
    num_outliers = len(outliers)

    print("Number of outliers beamSearch:", num_outliers)
def plot(beamSearch,beamSearchDPO,mbr,path,metric):
    # Step 2: Create the data arrays
    beamSearch = pd.read_csv(beamSearch)[f"{metric}_score"]*100  # Example array of 100 random numbers from a normal distribution
    beamSearchDPO = pd.read_csv(beamSearchDPO)[f"{metric}_score"]*100 # Another example array of 100 random numbers from a normal distribution
    #eamSearchKTO = pd.read_csv(BeamSearchKTO)[f"{metric}_score"]*100
    mbr= pd.read_csv(mbr)['ed_score']*100
    print('mbr',mbr.mean())
    print('beamSearch',beamSearch.mean())
    #print('beamSearchKTO',BeamSearchKTO.mean())
    print('beamSearchDPO',beamSearchDPO.mean())

    beamSearch = np.array(beamSearch)
    beamSearchDPO = np.array(beamSearchDPO)
    #BeamSearchKTO = np.array(BeamSearchKTO)
    mbr = np.array(mbr)
    
    median = np.median(beamSearch)
    std_dev = np.std(beamSearch)
    print("Standard Deviation beamSearch:", std_dev)
    print("Median beamSearch:", median)

    median = np.median(beamSearchDPO)
    std_dev = np.std(beamSearchDPO)
    print("Standard Deviation beamSearchDPO:", std_dev)
    print("Median: beamSearchDPO", median)

    median = np.median(mbr)
    std_dev = np.std(mbr)
    print("Standard Deviation mbr:", std_dev)
    print("Median: mbr", median)


    Q1 = np.percentile(beamSearch, 25)
    Q3 = np.percentile(beamSearch, 75)

    # Compute the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = beamSearch[(beamSearch < lower_bound) | (beamSearch > upper_bound)]

    # Number of outliers
    num_outliers = len(outliers)

    print("Number of outliers beamSearch:", num_outliers)
    

    Q1 = np.percentile(beamSearchDPO, 25)
    Q3 = np.percentile(beamSearchDPO, 75)

    # Compute the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = beamSearchDPO[(beamSearchDPO < lower_bound) | (beamSearchDPO > upper_bound)]

    # Number of outliers
    num_outliers = len(outliers)

    print("Number of outliers beamSearchDPO:", num_outliers)
    

    # Step 3: Create the box plots
    plt.figure(figsize=(10, 6))  # Create a figure with a specific size

    # Creating box plots for the two data arrays
    plt.boxplot([beamSearch,beamSearchDPO], patch_artist=True, boxprops=dict(facecolor='skyblue'))

    # Step 4: Customize the plot
    plt.xlabel("model type")
    plt.ylabel(metric)
    plt.xticks([1, 2,], ['BeamSearch', 'beamSearchDPO'])  # Label the x-axis with the names of the data sets

    # Step 5: Save the plot
    plt.savefig(path)  # Save the figure as a .png file
    plt.show()  # Display the plot

if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = argparse.ArgumentParser()


    parser.add_argument('--beamSearch', help="beamSearch path", default="../model-based-mbr/resultsMetrics/BeamZephyrSquadV2/['bertscore'].csv")
    parser.add_argument('--beamSearchDPO', help="beamSearchDPO path", default="../model-based-mbr/DPO_results/squad_v2T/bw/BETA0.5/['bertscore']_1000.csv")
    parser.add_argument('--MBR', default='../model-based-mbr/results/squad_v2_zephyr-7b-beta_032_0.01_00_1.00_bertscore_bertscore_0-1000.csv')
    #parser.add_argument('--beamSearchKTO',default="../model-based-mbr/KTO_results/strategyqaT/1:1/BETA0.1/['meteor', 'bertscore', 'rouge', 'rouge1', 'rouge2', 'sacrebleu', 'bleu_bp', 'bleu_lr']_457.csv")
    parser.add_argument('--path',default='../model-based-mbr/BoxPlotsMetrics/squadV2_bertscore_new.png')
    parser.add_argument('--metric', default='bertscore')

    
    
    args = parser.parse_args()
    beamSearch = args.beamSearch
    beamSearchDPO = args.beamSearchDPO
    #beamSearchKTO = args.beamSearchKTO
    MBR = args.MBR
    path = args.path
    metric = args.metric

    plot(beamSearch,beamSearchDPO,MBR,path,metric)