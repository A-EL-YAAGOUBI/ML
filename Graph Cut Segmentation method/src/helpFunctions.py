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

def segmentImage(image, threshold='image', imageName='image'):
    from scipy.misc import imsave
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    try:
        length, width, _ = image.shape
    except:
        length, width = image.shape

    background = []
    foreground = []


    ## In order to handle 3Bytes pixels
    try:
        image = np.array([[pixel[0] for pixel in line] for line in image])
    except:
        pass

    for line in image:
        for pixel in line:
            if pixel > threshold:
                foreground.append(pixel)
            else:
                background.append(pixel)

    # Sampling 10% of the pixels
    background = np.array(background[::10])
    foreground = np.array(foreground[::10])

    ## Parametre estimation (MLE for Gaussian model)
    print('Estimating parameters')
    muB = background.mean()
    muF = foreground.mean()
    sigmaB = background.std()
    sigmaF = foreground.std()

    ############################
    ## Visualizing the Gaussians
    print('Visualizating Gaussians')
    xAxis = np.linspace(-20, 255, 250)

    pdfB = [pdf(x, muB, sigmaB) for x in xAxis]
    pdfF = [pdf(x, muF, sigmaF) for x in xAxis]

    plt.figure(figsize=(13,5))
    plt.subplot(1, 2, 1)
    plt.plot(xAxis, pdfB, 'b')
    plt.title('Background distribution$(\mu , \sigma) = $({0:.2f}, {1:.2f})'.format(muB, sigmaB))
    plt.xlabel('Intensity')
    plt.ylabel('Probability density')

    plt.subplot(1, 2, 2)
    plt.plot(xAxis, pdfF, 'r')
    plt.title('Foreground distribution$(\mu , \sigma) = $({0:.2f}, {1:.2f})'.format(muF, sigmaF))
    plt.xlabel('Intensity')
    plt.ylabel('Probability density')
    plt.show()

    #####################
    ## Building the graph
    ## Setting the weights for non-terminal edges
    print('Building the Graph')
    myGraph = nx.grid_2d_graph(length, width)
    myGraph.add_node('F')
    myGraph.add_node('B')


    # Building links between non-terminal nodes
    for (xA, yA), (xB, yB) in myGraph.edges:
        capacity = (int(image[xA][yA]) - int(image[xB][yB]))**2
        myGraph.add_edge((xA, yA), (xB, yB), capacity=capacity)

    ## Setting the weights for terminal edges

    # in order to avoir numerical error with the log
    epsilon = 0.0001

    print('Adding weight on terminal edges')
    for node in myGraph.nodes:
        if node != 'F' and node != 'B':
            xNode, yNode = node
            pixel = image[xNode][yNode]
            myGraph.add_edge((xNode, yNode), 'F', capacity=-np.log(pdf(pixel, muB, sigmaB) + epsilon))
            myGraph.add_edge((xNode, yNode), 'B', capacity=-np.log(pdf(pixel, muF, sigmaF) + epsilon))

    ## Realizing the graph cut using Networkx built in minimum_cut function
    print('Cutting the Graph')
    _, (pixelsB, pixelsF) = nx.minimum_cut(G=myGraph, s='F', t='B')

    pixelsB = list(pixelsB)
    pixelsF = list(pixelsF)

    back  = np.zeros((length,width))
    front = np.zeros((length,width))

    for node in myGraph.nodes:
        if node != 'F' and node != 'B':
            x = node[0]
            y = node[1]
            if (x, y) in pixelsB:
                front[x][y] = image[x][y]
            else:
                back[x][y] = image[x][y]

    print('Visualizing the cut')
    plt.figure(figsize=(13,5))
    plt.subplot(1, 2, 1)
    plt.imshow(front)

    plt.subplot(1, 2, 2)
    plt.imshow(back)
    plt.show()

    imsave('./images/background/background_{}'.format(imageName), back)
    imsave('./images/foreground/foreground_{}'.format(imageName), front)
