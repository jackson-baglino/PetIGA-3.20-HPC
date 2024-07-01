from fitter import Fitter, get_common_distributions, get_distributions
from PIL import Image
from time import sleep
from prettytable import PrettyTable

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

input           = "./input/DIWICE-147-000_20x_Diameters.xlsx"
outDir          = "/Users/jacksonbaglino/LSDEM/LSDEM_Fluid_2D/input/"

outFile         = 'fitParams.dat'
outFilePath     = outDir + outFile

plotFlag = 1

np.random.seed(21)

def main():
        # Import the data from the .csv file
    data = pd.read_excel(input, index_col=0, 
                         dtype={'No.' : float, 'Measure' : str, 
                                'Result' : float, 'Unit' : str})
    
    # Extract the 'Result' column
    diameters = data['Result']

    minDiam = min(diameters)
    maxDiam = 1.2*max(diameters)

    # Display a histogram of diamters
    if plotFlag == 1:
        sns.set_style('white')
        sns.set_context("paper", font_scale=2)

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        sns.histplot(data=diameters, kde=True, bins=12)

        plt.xlabel('Diameters (microns)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Diameters')

        plt.show()

    # Fit the data to determine the distribution type
    f = Fitter(diameters,
            distributions=[
                'beta',
                'cauchy',
                'chi2',
                'expon',
                'exponpow',
                'gamma',
                'lognorm',
                'norm',
                'powerlaw',
                'rayleigh',
                'uniform'])
    f.fit()

    # Display the summary using PrettyTable for a nicely formatted output
    summary = f.summary()
    pretty_summary = PrettyTable()
    pretty_summary.field_names = summary.columns
    for row in summary.iterrows():
        pretty_summary.add_row(row[1])
    
    if plotFlag == 1:
        print(pretty_summary)

        # Plot the fitted PDFs
        plt.figure()
        for distribution_name in f.get_best(method='sumsquare_error'):
            distribution = getattr(stats, distribution_name)
            params = f.fitted_param[distribution_name]  # Corrected this line

            x = np.linspace(diameters.min(), diameters.max(), 1000)
            pdf = distribution.pdf(x, *params)

            plt.plot(x, pdf, label=f'Fitted {distribution_name}')

        plt.xlabel('Diameters (microns)')
        plt.ylabel('Probability Density')
        plt.title('Fitted PDFs')
        plt.legend()
        plt.show()
        plt.close()

    # Determine the best fit
    bestFit = f.get_best(method='sumsquare_error')

    # Get the name of the best-fit distribution
    bestFitName = next(iter(bestFit))
    print(f"The best-fit distribution is: {bestFitName}.")

    # Identify the distribution parameters
    fitParams = f.fitted_param[bestFitName]
    print(f"Best fit parameters are {fitParams}.")

    with open(outFilePath, 'w') as file:
        for param in fitParams:
            file.write(str(param) + '\n')

        file.write(str(minDiam) + '\n')
        file.write(str(maxDiam) + '\n')

    # Print a message to confirm that the data has been written
    print(f'fitParams has been written to {outFilePath}')

    return 0


if __name__ == "__main__":
    plt.ion()
    main()
    plt.pause(1)
    plt.ioff()