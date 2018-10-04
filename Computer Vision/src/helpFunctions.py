def drawHistogram(newIm, title):
    import matplotlib.pyplot as plt
    import numpy as np
    frequencies, intensities = np.histogram(newIm)
    intensities = intensities[1:]
    plt.plot(intensities, frequencies)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

def pdf(x, mu, sig):
    import numpy as np
    return np.exp(-(x - mu)**2 / (sig**2)) / (np.sqrt(2*np.pi)*sig)
