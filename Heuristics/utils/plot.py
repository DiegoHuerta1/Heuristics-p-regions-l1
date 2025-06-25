import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np


def plot_record_f(record_f, name = "Multi-start Local search", figsize = (5, 3)):
    # best found solution
    min_value = min([f for hist_f in record_f for f in hist_f])

    sns.set_theme()
    fig, ax = plt.subplots(figsize=figsize)

    # iterate in each ls
    for hist in record_f:
        # if it is the best
        if np.isclose(min(hist), min_value):
            ax.plot(range(1, len(hist)+1), hist, '.-', alpha = 0.8, color = 'red')
        else:
            ax.plot(range(1, len(hist)+1), hist, '.-', alpha = 0.3, color = 'slategray')


    # personalization
    ax.set_title(f"Record of objective functions\nIterations of LS in {name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()
    plt.close()


def format_time(seconds):

    # first round to an integer
    seconds = int(round(seconds, 0))
    # calculate hours and minutes
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    # see how many seconds are left
    remaining_seconds = seconds % 60

    # start with an empty string
    formatted_time = ""

    # if there are hours
    if hours > 0:
        formatted_time += f"{hours} hours "
    # if there are minutes
    if minutes > 0:
        formatted_time += f"{minutes} minutes "
    # the remaining seconds
    if remaining_seconds > 0 or formatted_time == "":
        formatted_time += f"{remaining_seconds} seconds"
    # return the string
    return formatted_time.strip()
